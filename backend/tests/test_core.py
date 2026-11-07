import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("PILLSPARK_DATA_DIR", str(Path(__file__).parent / ".tmpdata"))
os.environ["MEMORY_BACKEND"] = "local"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import export, memory, registries, safety  # noqa: E402
from app.schemas import FieldResult, Order  # noqa: E402


def f(key="med1.drug_name", name="drug_name", **kw):
    return FieldResult(key=key, name=name, label="Drug", group="Line 1", status="checking", **kw)


# ---------------------------------------------------------------- safety invariant
@pytest.mark.parametrize("source", ["memory", "sie_model", "reranker", "consensus", "pharmacist_suggestion", ""])
def test_untrusted_sources_cannot_verify(source):
    fr = f(value="cytarabine")
    with pytest.raises(safety.UnsafeVerification):
        safety.set_status(fr, "verified", source)
    assert fr.status == "checking"


def test_single_reader_can_never_be_verified():
    fr = f(value="cytoxan")
    fr.evidence["agreeing_readers"] = ["GLM-OCR"]          # one OCR source, even if two models agree on it
    with pytest.raises(safety.UnsafeVerification):
        safety.set_status(fr, "verified", "triage_rules")
    fr2 = f(value="cytoxan")                                # no reader record at all
    with pytest.raises(safety.UnsafeVerification):
        safety.set_status(fr2, "verified", "triage_rules")


def test_triage_rules_can_verify_only_without_open_reasons():
    fr = f(value="daunorubicin")
    fr.evidence["agreeing_readers"] = ["GLM-OCR", "LightOnOCR"]
    safety.set_status(fr, "verified", "triage_rules")
    assert fr.status == "verified" and fr.status_source == "triage_rules"
    fr2 = f(value="x", reasons=["Readers disagree"])
    fr2.evidence["agreeing_readers"] = ["GLM-OCR", "LightOnOCR"]
    with pytest.raises(safety.UnsafeVerification):
        safety.set_status(fr2, "verified", "triage_rules")


def test_only_pharmacist_resolves():
    fr = f(value="x")
    with pytest.raises(safety.UnsafeVerification):
        safety.set_status(fr, "resolved", "memory")
    safety.set_status(fr, "resolved", "pharmacist")
    assert fr.status == "resolved"


def test_unreadable_withholds_value():
    fr = f(value="Dr. Smith")
    safety.set_status(fr, "unreadable", "triage_rules")
    assert fr.value is None


# ---------------------------------------------------------------- PHI never reaches memory
def test_phi_stripped_before_memory_write():
    mem = memory.Memory()
    rec = mem.write("pharmacist_corrections", "pharmacist", {
        "field": "dose", "final_value": "80 mg", "raw_readings": {"Qwen+LightOnOCR": "80 mg"},
        "patient_name": "Mujibur Rahman", "patient_dob": "1958-03-04", "patient": {"name": "Mujibur Rahman"},
        "reason": "matches day 3"})
    stored = (memory.MEM_DIR / "pharmacist_corrections.jsonl").read_text()
    assert "Mujibur" not in stored and "1958-03-04" not in stored
    assert rec["final_value"] == "80 mg"


class FakePII:
    def pii(self, model, text, labels):
        out = []
        for needle, label in (("Mujibur Rahman", "person"), ("03/04/1958", "date of birth")):
            i = text.find(needle)
            if i >= 0:
                out.append({"text": needle, "label": label, "start": i, "end": i + len(needle), "score": 1.0})
        return out, {"ms": 1, "model": model}


def test_free_text_redacted_with_pii_model():
    text, meta, n = memory.redact_text(FakePII(), "Called Mujibur Rahman, DOB 03/04/1958, confirmed 80 mg")
    assert "Mujibur" not in text and "03/04/1958" not in text and n == 2 and "80 mg" in text


def test_memory_write_permissions():
    mem = memory.Memory()
    with pytest.raises(memory.MemoryPermissionError):
        mem.write("pharmacist_corrections", "triage-agent", {"field": "dose"})
    with pytest.raises(memory.MemoryPermissionError):
        mem.write("prescriber_patterns", "pharmacist", {"field": "dose"})
    mem.write("agent_notes", "extraction-agent", {"field": "drug_name", "note": "readers disagreed"})


def test_recall_finds_similar_reading_and_retraction_hides_it():
    mem = memory.Memory()
    rec = mem.write("pharmacist_corrections", "pharmacist", {"field": "drug_name", "raw_readings": {"a": "Cytosar-U"}, "final_value": "cytarabine"})
    hits = mem.recall("drug_name", ["cytosar-u 150mg"])
    assert any(h["id"] == rec["id"] for h in hits) or mem.recall("drug_name", ["Cytosar-U"])
    mem.retract(rec["id"], "pharmacist", "wrong pattern")
    assert not any(h["id"] == rec["id"] for h in mem.recall("drug_name", ["Cytosar-U"]))


# ---------------------------------------------------------------- validators
def test_npi_luhn():
    assert registries.npi_valid("1234567893")
    assert not registries.npi_valid("1234567890")


def test_dea_checksum():
    assert registries.dea_valid("AB1234563")["ok"]
    assert not registries.dea_valid("AB1234564")["ok"]


def test_bsa_dose_paused_without_bsa():
    assert registries.bsa_dose_check("dauno", "80 mg", None)["paused"]           # handwritten abbreviation
    assert registries.bsa_dose_check("cytoxan", "150 mg", None) is None           # cyclophosphamide is not in the 7+3 table
    r = registries.bsa_dose_check("daunorubicin", "80 mg", None)
    assert r["paused"] and r["ok"] is None
    assert registries.bsa_dose_check("daunorubicin", "80 mg", 1.6)["ok"] is True


def test_ismp_qd_flagged():
    assert registries.parse_sig("1 tab po q.d.")["error_prone_abbrev"]


# ---------------------------------------------------------------- FHIR export shape
def test_fhir_medicationrequest_r4_shape():
    o = Order(order_id="PS-TEST01", filename="x.jpg", sha256="0" * 64, created_at=0, status="approved")
    for name, val in (("drug_name", "cytarabine"), ("dose", "150 mg"), ("directions", "IV over 24 h"), ("date", "2/9/24")):
        fr = FieldResult(key=f"med1.{name}", name=name, label=name, group="Line 1", status="resolved", resolved_value=val)
        o.fields[fr.key] = fr
    b = export.fhir_bundle(o)
    assert b["resourceType"] == "Bundle" and b["type"] == "collection"
    mr = b["entry"][0]["resource"]
    assert mr["resourceType"] == "MedicationRequest"
    assert mr["status"] in {"active", "on-hold", "cancelled", "completed", "entered-in-error", "stopped", "draft", "unknown"}
    assert mr["intent"] in {"proposal", "plan", "order", "original-order", "reflex-order", "filler-order", "instance-order", "option"}
    assert "subject" in mr and "medicationCodeableConcept" in mr
    q = mr["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
    assert q["value"] == 150.0 and q["system"] == "http://unitsofmeasure.org"
    json.dumps(b)


# ---------------------------------------------------------------- field location boxes
def test_box_conversion():
    from app import pipeline
    assert pipeline._box_pct([333, 402, 518, 425]) == [33.3, 40.2, 18.5, 2.3]
    assert pipeline._box_pct([0, 0, 0, 0]) is None                    # not found
    assert pipeline._box_pct([0, 0, 1000, 1000]) is None              # whole page is not a location
    assert pipeline._box_pct([518, 425, 333, 402]) == [33.3, 40.2, 18.5, 2.3]   # corners swapped


def test_feedback_reinforces_and_fades():
    mem = memory.Memory()
    good = mem.write("pharmacist_corrections", "pharmacist", {"field": "dose", "raw_readings": {"a": "8omg"}, "final_value": "80 mg"})
    bad = mem.write("pharmacist_corrections", "pharmacist", {"field": "dose", "raw_readings": {"a": "8omg"}, "final_value": "90 mg"})
    mem.feedback([good["id"]], True, "dose")
    ids = [h["id"] for h in mem.recall("dose", ["8omg"])]
    assert ids[0] == good["id"]                                  # accepted suggestion ranks first
    mem.feedback([bad["id"]], False, "dose"); mem.feedback([bad["id"]], False, "dose")
    assert bad["id"] not in [h["id"] for h in mem.recall("dose", ["8omg"])]   # ignored twice: fades out
