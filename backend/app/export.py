"""Exports for approved orders: JSON, CSV and an HL7 FHIR R4 Bundle of MedicationRequest resources."""
import csv
import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from .schemas import Order


def final_value(f) -> Any:
    return f.resolved_value if f.status == "resolved" else f.value if f.status == "verified" else None


def order_json(o: Order) -> Dict[str, Any]:
    return {"order_id": o.order_id, "status": o.status, "approved": o.status == "approved",
            "fields": {k: {"value": final_value(f), "status": f.status, "resolved_by": f.resolved_by, "reason": f.resolution_reason}
                       for k, f in o.fields.items()}}


def csv_text(o: Order) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["order_id", "group", "field", "value", "status", "resolved_by", "reason"])
    for k, f in o.fields.items():
        w.writerow([o.order_id, f.group, f.label, final_value(f), f.status, f.resolved_by or "", f.resolution_reason or ""])
    return buf.getvalue()


def _lines(o: Order) -> List[int]:
    return sorted({int(k[3:].split(".")[0]) for k in o.fields if k.startswith("med")})


def fhir_bundle(o: Order) -> Dict[str, Any]:
    """FHIR R4 Bundle (type collection) with one MedicationRequest per medication line."""
    entries = []
    pres = o.fields.get("prescriber_name")
    pres_v = final_value(pres) if pres else None
    for i in _lines(o):
        g = lambda f: final_value(o.fields[f"med{i}.{f}"])  # noqa: E731
        drug, dose, sig, date = g("drug_name"), g("dose"), g("directions"), g("date")
        mr: Dict[str, Any] = {
            "resourceType": "MedicationRequest",
            "id": f"{o.order_id.lower()}-{i}",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": str(drug) if drug else "unknown"},
            "subject": {"display": "Patient (see source document)"},
            "dosageInstruction": [{"text": " ".join(str(x) for x in (dose, sig) if x) or "see source document"}],
        }
        m = re.match(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)", str(dose or "").lower())
        if m:
            mr["dosageInstruction"][0]["doseAndRate"] = [{"doseQuantity": {"value": float(m.group(1)), "unit": m.group(2),
                                                                             "system": "http://unitsofmeasure.org", "code": m.group(2)}}]
        if pres_v:
            mr["requester"] = {"display": str(pres_v)}
        if date:
            mr["note"] = [{"text": f"Row date as written: {date}"}]
        entries.append({"fullUrl": f"urn:uuid:{mr['id']}", "resource": mr})
    return {"resourceType": "Bundle", "type": "collection", "timestamp": datetime.now(timezone.utc).isoformat(), "entry": entries}
