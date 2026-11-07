"""Pharmacy memory: what pharmacists corrected and what agents noticed, shared across agents.

Rules:
- No patient identifiers ever enter memory. Patient fields are stripped structurally and free text is
  passed through SIE's pii-redact model first. Memory is keyed on prescriber, drug and field patterns.
- Write permissions per dataset (read is open to every agent and role).
- Memory can add candidates, add a "Seen before" reason, or raise a flag. It can never verify
  (see safety.set_status, which rejects source="memory").

Backends: cognee (when MEMORY_BACKEND allows and it initialises) mirrors every record into a cognee
dataset; the local JSONL index is always written and is what recall matches on, so recall stays
deterministic and fast during a live run. Events name the backend that actually stored the record.
"""
import json
import re
import threading
import time
import uuid
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from . import config

DATASETS = ("pharmacist_corrections", "prescriber_patterns", "agent_notes")
WRITE_PERMS = {
    "pharmacist_corrections": {"pharmacist"},
    "prescriber_patterns": {"verification-agent"},
    "agent_notes": {"extraction-agent", "verification-agent", "triage-agent"},
}
PII_LABELS = ["person", "date of birth", "medical record number", "phone number", "street address",
              "national id number", "health plan beneficiary number", "email address"]

MEM_DIR = config.DATA_DIR / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


class MemoryPermissionError(PermissionError):
    pass


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./-]", " ", str(s or "").lower())).strip()


def strip_phi(record: Dict[str, Any]) -> Dict[str, Any]:
    """Structural strip: drop every patient field and any key that looks like a patient identifier."""
    out = {}
    for k, v in record.items():
        if k in config.PHI_FIELDS or k.startswith("patient"):
            continue
        if isinstance(v, dict):
            v = strip_phi(v)
        out[k] = v
    return out


def redact_text(provider, text: str) -> Tuple[str, Optional[Dict[str, Any]], int]:
    """Replace PII spans found by SIE pii-redact with [label]. Returns (text, meta, n_spans)."""
    if not text or provider is None:
        return text, None, 0
    ents, meta = provider.pii(config.MODELS["pii"], text, PII_LABELS)
    spans = sorted({(e["start"], e["end"], e["label"]) for e in ents if e.get("start") is not None}, reverse=True)
    for start, end, label in spans:
        text = text[:start] + f"[{label}]" + text[end:]
    return text, meta, len(spans)


class Memory:
    def __init__(self):
        self.backend = "local"
        self.cognee_error: Optional[str] = None
        self._cognee = None
        if config.MEMORY_BACKEND in ("auto", "cognee"):
            try:
                from . import cognee_store
                self._cognee = cognee_store.CogneeStore()
                self.backend = "cognee"
            except Exception as e:  # noqa: BLE001
                self.cognee_error = f"{type(e).__name__}: {str(e)[:160]}"
                if config.MEMORY_BACKEND == "cognee":
                    raise

    # ------------------------------------------------------------------ write
    def write(self, dataset: str, actor: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset}")
        if actor not in WRITE_PERMS[dataset]:
            raise MemoryPermissionError(f"{actor} has no write permission on {dataset}")
        rec = strip_phi(record)
        rec.update({"id": f"{dataset[:3]}-{uuid.uuid4().hex[:6]}", "dataset": dataset, "written_by": actor, "ts": time.time()})
        with _lock, open(MEM_DIR / f"{dataset}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        stored_in = "local"
        if self._cognee is not None:
            try:
                stored_in = self._cognee.remember(dataset, actor, rec)
            except Exception as e:  # noqa: BLE001
                self.cognee_error = f"{type(e).__name__}: {str(e)[:160]}"
        rec["stored_in"] = stored_in
        return rec

    def retract(self, record_id: str, actor: str, reason: str) -> Dict[str, Any]:
        return self.write("pharmacist_corrections", actor, {"kind": "retraction", "retracts": record_id, "reason": reason})

    # ------------------------------------------------------------------ read
    def all(self, dataset: str) -> List[Dict[str, Any]]:
        p = MEM_DIR / f"{dataset}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        retracted = {r.get("retracts") for r in rows if r.get("kind") == "retraction"}
        return [r for r in rows if r.get("id") not in retracted and r.get("kind") not in ("retraction", "feedback")]

    def feedback_weights(self) -> Dict[str, int]:
        """+1 for each time a pharmacist accepted a memory suggestion, -1 each time it was not used."""
        p = MEM_DIR / "pharmacist_corrections.jsonl"
        w: Dict[str, int] = {}
        if p.exists():
            for l in p.read_text().splitlines():
                r = json.loads(l) if l.strip() else {}
                if r.get("kind") == "feedback":
                    for mid in r.get("feedback_for") or []:
                        w[mid] = w.get(mid, 0) + (1 if r.get("accepted") else -1)
        return w

    def feedback(self, memory_ids: List[str], accepted: bool, field: str) -> Dict[str, Any]:
        return self.write("pharmacist_corrections", "pharmacist", {"kind": "feedback", "feedback_for": memory_ids,
                                                                    "accepted": accepted, "field_kind": field})

    def recall(self, field_name: str, readings: List[str], prescriber: Optional[str] = None) -> List[Dict[str, Any]]:
        """Corrections and notes whose raw readings look like these readings for the same field."""
        want = [_norm(r) for r in readings if r]
        hits = []
        for ds in DATASETS:
            for rec in self.all(ds):
                if rec.get("field") != field_name:
                    continue
                raws = [_norm(x) for x in (rec.get("raw_readings") or {}).values() if x] + [_norm(rec.get("pattern", ""))]
                score = max((SequenceMatcher(None, a, b).ratio() for a in want for b in raws if a and b), default=0)
                if score >= 0.85:
                    hits.append({**rec, "match": round(score, 2)})
        weights = self.feedback_weights()
        hits = [h for h in hits if weights.get(h["id"], 0) > -2]          # repeatedly unused suggestions fade out
        for h in hits:
            h["feedback"] = weights.get(h["id"], 0)
        hits.sort(key=lambda r: (-r["feedback"], -r["match"], -r["ts"]))  # accepted suggestions are reinforced
        return hits[:3]

    def stats(self) -> Dict[str, Any]:
        out = {"backend": self.backend, "cognee_error": self.cognee_error,
               "counts": {ds: len(self.all(ds)) for ds in DATASETS}}
        if self._cognee is not None:
            out["cognee"] = self._cognee.stats()
        return out


_memory: Optional[Memory] = None


def get() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory
