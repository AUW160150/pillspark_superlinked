"""Six agents: Intake -> Extraction -> Consensus -> Verification -> Triage -> Order.

Every step emits an event in the shape the review screen renders:
  {t, stage, src: sie|registry|rules|agent|memory, who, model?, ms?, credits?, level?: ok|flag|block|info, msg, ...}
Adapted from the original PillSpark prototype; the model layer now runs on Superlinked SIE open models.
"""
import io
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import config, memory as memmod, registries as reg
from .config import MED_FIELDS, MODELS, ORDER_FIELDS
from .safety import UnsafeVerification, is_flag, set_status
from .schemas import Candidate, FieldResult, Order
from .sie import ModelUnavailable, Provider, to_jpeg, to_png

Emit = Callable[[Dict[str, Any]], None]
STAGES = ["intake", "extraction", "consensus", "verification", "triage", "order"]
SHORT = {MODELS["ocr_primary"]: "LightOnOCR", MODELS["ocr_secondary"]: "GLM-OCR", MODELS["field_tagger"]: "GLiNER2",
         MODELS["structured_parse"]: "Qwen", MODELS["reranker"]: "Qwen3-Reranker", MODELS["pii"]: "pii-redact"}

PARSE_PROMPT = """You are transcribing a prescription or medication chart for a pharmacist. Return JSON only.
Rules:
- Transcribe exactly what is written. Do not correct spelling, expand abbreviations, or guess.
- A field not on the page: present=false, value="".
- Writing you can see but cannot read with certainty: legible=false, put your possible readings in alternatives.
- A signature without a printed or clearly written name is NOT a legible name.
- confidence (0 to 1) is how sure you are of the exact characters.
order: patient_name, diagnosis, regimen (named protocol), body_surface_area (with units), prescriber_name, prescriber_npi (10 digits).
medications: one entry per drug administration row, top to bottom. drug_name exactly as written (brand or generic),
dose with unit, directions = route, diluent, volume and infusion time as written, date = the row's date as written."""

LOCATE_PROMPT = """Locate fields on this prescription/medication chart. For each item below give a tight bounding box
[x1, y1, x2, y2] around where it is written, in coordinates normalised to 0-1000 relative to the image width and height
(0,0 = top-left). Use [0, 0, 0, 0] if you cannot find it. For a blank form line (for example an empty "Body Surface area"),
box the printed label and its empty line. For a prescriber who only signed, box the signature.
Items:
"""
_BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}


def _fspec() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["value", "present", "legible", "confidence", "alternatives"],
            "properties": {"value": {"type": "string"}, "present": {"type": "boolean"}, "legible": {"type": "boolean"},
                           "confidence": {"type": "number"}, "alternatives": {"type": "array", "items": {"type": "string"}}}}


PARSE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["order", "medications"],
    "properties": {
        "order": {"type": "object", "additionalProperties": False, "required": list(ORDER_FIELDS), "properties": {k: _fspec() for k in ORDER_FIELDS}},
        "medications": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": list(MED_FIELDS),
                                                   "properties": {k: _fspec() for k in MED_FIELDS}}},
    },
}
TAG_SCHEMA = {"type": "object", "properties": {
    "patient_name": {"type": "string", "description": "full name of the patient"},
    "diagnosis": {"type": "string", "description": "the patient's diagnosis"},
    "regimen": {"type": "string", "description": "named chemotherapy or treatment protocol"},
    "body_surface_area": {"type": "string", "description": "body surface area in square metres"},
    "prescriber_name": {"type": "string", "description": "printed name of the prescribing doctor"},
    "prescriber_npi": {"type": "string", "description": "10-digit NPI number of the prescriber"},
}}


# ------------------------------------------------------------------ normalisation
def _plain(md: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", md)
    t = re.sub(r"</t[dh]>", " | ", t)
    t = re.sub(r"</tr>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[ \t]+", " ", t)


def norm(field: str, v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if field == "dose":
        return reg.norm_strength(s)
    if field == "date":
        m = re.findall(r"\d+", s)
        return "/".join(str(int(x)) for x in m) if m else reg.norm_text(s)
    if field == "drug_name":
        t = reg.norm_text(s)
        t = re.sub(r"^(?:→\s*)?(?:inj\.?|i\.?v\.?|iv|injection)\s+", "", t)
        return re.sub(r"\s*\(.*$", "", t).strip()
    if field == "prescriber_name":
        t = reg.norm_text(s)
        return re.sub(r"\b(dr|md|mo|do|mbbs)\b\.?", "", t).replace("( )", "").strip(" .()")
    if field == "prescriber_npi":
        return re.sub(r"\D", "", s)
    return reg.norm_text(s)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, order: Order, emit: Optional[Emit]):
        self.order = order
        self.t0 = time.perf_counter()
        self.emit_cb = emit
        self.m = {"sie_calls": 0, "credits": 0.0, "registry": 0, "mem_reads": 0, "mem_writes": 0, "rejected": 0, "latency": []}

    def emit(self, msg: str, src: str = "agent", **kw) -> Dict[str, Any]:
        ev = {"type": "event", "t": int((time.perf_counter() - self.t0) * 1000), "stage": self.order.stage, "src": src, "msg": msg}
        ev.update({k: v for k, v in kw.items() if v is not None})
        if self.emit_cb:
            self.emit_cb(ev)
        return ev

    def stage(self, idx: int, note: str):
        self.order.stage = idx
        self.emit(note, "agent", who=f"{STAGES[idx]}-agent", type="stage", state="running")

    def model_event(self, meta: Dict[str, Any], msg: str, level: str = "ok", who: Optional[str] = None):
        self.m["sie_calls"] += 1
        self.m["credits"] += meta.get("credits") or 0
        self.m["latency"].append({"name": SHORT.get(meta["model"], meta["model"]), "ms": meta["ms"], "kind": "sie"})
        self.emit(msg, "sie", model=SHORT.get(meta["model"], meta["model"]), model_id=meta["model"], ms=meta["ms"],
                  credits=meta.get("credits"), replayed=meta.get("replayed") or None, level=level, who=who)

    def check(self, cid: str, state: str, detail: str):
        self.emit(detail, "rules", type="check", check=[cid, state, detail], level=None, quiet=True)

    def snapshot(self):
        self.emit("fields", "agent", type="fields", fields=[f.model_dump() for f in self.order.fields.values()], quiet=True)


# ------------------------------------------------------------------ 0. intake
def agent_intake(ctx: Ctx, image_bytes: bytes) -> Dict[str, Any]:
    ctx.stage(0, "Intake started")
    im = Image.open(io.BytesIO(image_bytes))
    w, h = im.size
    g = np.asarray(im.convert("L").resize((min(w, 1200), int(h * min(w, 1200) / w))), dtype=np.float32)
    lap = (np.abs(np.diff(g, 2, axis=0))[:, :-2] + np.abs(np.diff(g, 2, axis=1))[:-2, :])
    sharp, contrast, ink = float(lap.var()), float(g.std()), float((g < 110).mean())
    quality = max(0.0, min(1.0, 0.5 * min(sharp / 900.0, 1.0) + 0.3 * min(contrast / 70.0, 1.0) + 0.2 * (1.0 if 0.01 < ink < 0.35 else 0.3)))
    q = {"width": w, "height": h, "sharpness": round(sharp, 1), "contrast": round(contrast, 1), "quality": round(quality, 2)}
    ctx.order.metrics["image_quality"] = q
    ctx.emit(f"Document received: {ctx.order.filename}, {w}×{h} px. Idempotency key checked.", who="intake-agent")
    ok = quality >= 0.45
    ctx.emit(f"Image quality {quality:.2f} (sharpness {q['sharpness']}, contrast {q['contrast']})." + ("" if ok else " Low quality: expect more fields to need review."),
             "rules", who="intake-agent", level="ok" if ok else "flag")
    ctx.check("quality", "pass" if ok else "fail", f"Quality {quality:.2f}")
    ctx.emit("Image bytes kept out of logs, traces and events.", who="logger")
    return q


# ------------------------------------------------------------------ 1. extraction
PLAUSIBLE = {
    "body_surface_area": lambda v: bool(re.search(r"\d", v)),
    "prescriber_npi": lambda v: bool(re.fullmatch(r"\D*\d{10}\D*", v)),
    "prescriber_name": lambda v: bool(re.search(r"[A-Za-z]{2}", v)) and not re.search(r"\d{1,2}/\d{1,2}", v),
    "patient_name": lambda v: bool(re.search(r"[A-Za-z]{2}", v)),
}


def _plausible(field: str, value: str) -> bool:
    check = PLAUSIBLE.get(field)
    return check(value) if check else True


def _windows(text: str, size: int = 170, overlap: int = 30) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return out or [""]


def agent_extraction(ctx: Ctx, prov: Provider, image_bytes: bytes) -> List[Dict[str, Any]]:
    ctx.stage(1, "Extraction started on Superlinked SIE")
    png, jpeg = to_png(image_bytes), to_jpeg(image_bytes, 1600)   # SIE counts an image as ~3.1k tokens at any size
    readers = [MODELS["ocr_primary"], MODELS["ocr_secondary"]]
    ctx.emit(f"Sent the page to {SHORT[readers[0]]} and {SHORT[readers[1]]} in parallel as two independent readers.", "sie", who="extraction-agent", type="scan", scan=True)
    transcripts: Dict[str, str] = {}
    with ThreadPoolExecutor(4) as ex:
        futs = {m: ex.submit(prov.ocr, m, png) for m in readers}
        for m, f in futs.items():
            try:
                text, meta = f.result()
                transcripts[m] = text
                ctx.model_event(meta, f"{SHORT[m]} read {len(text.splitlines())} lines ({len(text)} characters).", who="extraction-agent")
            except ModelUnavailable as e:
                ctx.emit(f"{SHORT[m]} failed: {e}. Continuing with the other reader.", "sie", model=SHORT[m], level="flag", who="extraction-agent")
    ctx.emit("OCR finished.", "agent", type="scan", scan=False, quiet=True)
    ctx.order.ocr = {SHORT[m]: t for m, t in transcripts.items()}
    if not transcripts:
        return []
    passes: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(8) as ex:
        jobs = []
        for m, text in transcripts.items():
            jobs.append(("parse", m, ex.submit(prov.parse, MODELS["structured_parse"], jpeg, text, PARSE_SCHEMA, PARSE_PROMPT, SHORT[m])))
            for wi, win in enumerate(_windows(_plain(text))):
                jobs.append(("tag", m, ex.submit(prov.tag_fields, MODELS["field_tagger"], win, TAG_SCHEMA)))
        tags: Dict[str, Dict[str, List[str]]] = {}
        for kind, m, fut in jobs:
            try:
                value, meta = fut.result()
            except ModelUnavailable as e:
                ctx.emit(f"{SHORT[MODELS['structured_parse' if kind == 'parse' else 'field_tagger']]} failed on the {SHORT[m]} transcript: {e}", "sie", level="flag", who="extraction-agent")
                continue
            if kind == "parse":
                meds = value.get("medications") or []
                ctx.model_event(meta, f"Qwen read the image plus the {SHORT[m]} transcript into the strict order schema: {len(meds)} medication rows.", who="extraction-agent")
                passes.append({"name": f"Qwen+{SHORT[m]}", "kind": "parse", "data": value})
            else:
                found = ", ".join(sorted(value)) or "nothing"
                ctx.model_event(meta, f"GLiNER2 tagged order fields in the {SHORT[m]} text: {found}.", who="extraction-agent")
                for k, v in value.items():
                    tags.setdefault(m, {}).setdefault(k, []).append(v)
        for m, fields in tags.items():
            kept = {k: v[0] for k, v in fields.items() if _plausible(k, v[0])}
            dropped = sorted(set(fields) - set(kept))
            if dropped:
                ctx.emit(f"GLiNER2 spans that can't be valid for their field were dropped ({', '.join(dropped)}): a tagger can point, it can't decide.", "rules", who="extraction-agent")
            passes.append({"name": f"GLiNER2+{SHORT[m]}", "kind": "tag", "data": kept})
    return passes


# ------------------------------------------------------------------ 2. consensus
def _new_field(key: str, name: str, label: str, group: str) -> FieldResult:
    return FieldResult(key=key, name=name, label=label, group=group, status="checking")


def _box_pct(bb: Any) -> Optional[List[float]]:
    """[x1,y1,x2,y2] in 0-1000 -> [x%, y%, w%, h%]; None for empty or implausible boxes."""
    try:
        x1, y1, x2, y2 = (float(v) for v in bb)
    except Exception:
        return None
    x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
    y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4 or w * h > 250_000:          # empty, a dot, or a quarter of the page: not a field location
        return None
    return [round(x1 / 10, 2), round(y1 / 10, 2), round(w / 10, 2), round(h / 10, 2)]


def _iou(a: List[float], b: List[float]) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def jpeg_for(image_bytes: bytes) -> bytes:
    return to_jpeg(image_bytes, 1600)


def _desc(fr: FieldResult) -> str:
    v = fr.value if fr.value is not None else (fr.candidates[0].value if fr.candidates else None)
    return f"{fr.label}: {v!r}" if v else f"{fr.label} (blank or unreadable)"


def locate_fields(ctx: "Ctx", prov: Provider, jpeg: bytes):
    """One Qwen grounding call: where each agreed (or disputed) value sits on the page. Boxes are evidence for the
    pharmacist's eyes only; they never change a status."""
    fields = ctx.order.fields
    order_keys = [k for k in ORDER_FIELDS if k != "prescriber_npi"]
    lines = sorted({int(k[3:].split(".")[0]) for k in fields if k.startswith("med")})
    items = [f"order.{k}: {_desc(fields[k])}" for k in order_keys]
    for i in lines:
        items.append(f"line {i}: " + "; ".join(_desc(fields[f'med{i}.{f}']) for f in MED_FIELDS))
    schema = {"type": "object", "additionalProperties": False, "required": ["order", "lines"], "properties": {
        "order": {"type": "object", "additionalProperties": False, "required": order_keys, "properties": {k: _BOX for k in order_keys}},
        "lines": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": list(MED_FIELDS),
                                             "properties": {f: _BOX for f in MED_FIELDS}}}}}
    try:
        data, meta = prov.locate(MODELS["structured_parse"], jpeg, LOCATE_PROMPT + "\n".join(items), schema)
    except ModelUnavailable as e:
        ctx.emit(f"Could not locate fields on the page ({e}); values are still shown with their OCR text.", "sie", model="Qwen", level="flag", who="consensus-agent")
        return
    located = 0
    def put(fr: FieldResult, bb):
        nonlocal located
        bx = _box_pct(bb)
        if bx:
            v = fr.value if fr.value is not None else (fr.candidates[0].value if fr.candidates else "blank")
            fr.evidence["boxes"] = [bx + [f"{fr.label}: {str(v)[:24]}"]]
            located += 1
    for k in order_keys:
        put(fields[k], (data.get("order") or {}).get(k))
    for idx, i in enumerate(lines):
        row = (data.get("lines") or [])[idx] if idx < len(data.get("lines") or []) else {}
        for f in MED_FIELDS:
            put(fields[f"med{i}.{f}"], row.get(f))
    ctx.model_event(meta, f"Qwen located {located} of {len(fields)} fields on the page (boxes are evidence for the pharmacist, never a reason to verify).", who="consensus-agent")
    ctx.emit("regions", "agent", type="regions", quiet=True, regions=[b[:4] for fr in fields.values() for b in fr.evidence.get("boxes", [])])


def _vote(fr: FieldResult, readings: List[Tuple[str, Optional[Dict[str, Any]], Optional[str]]]):
    """readings: (pass name, parse-field dict or None, tagger value or None)."""
    votes: Counter = Counter()
    raw: Dict[str, Any] = {}
    srcs: Dict[str, List[str]] = {}
    confs: Dict[str, List[float]] = {}
    for pname, pf, tagv in readings:
        if pf is not None:
            if not pf.get("present"):
                k = "__absent__"
            elif not pf.get("legible") or not str(pf.get("value") or "").strip():
                k = "__illegible__"
            else:
                k = norm(fr.name, pf["value"])
            votes[k] += 1
            raw.setdefault(k, pf.get("value"))
            srcs.setdefault(k, []).append(pname)
            confs.setdefault(k, []).append(float(pf.get("confidence") or 0))
            for alt in pf.get("alternatives") or []:
                ak = norm(fr.name, alt)
                if ak and ak not in raw:
                    raw[ak] = alt
                    srcs.setdefault(ak, []).append(f"{pname} (alternative)")
        elif tagv:
            k = norm(fr.name, tagv)
            votes[k] += 1
            raw.setdefault(k, tagv)
            srcs.setdefault(k, []).append(pname)
            confs.setdefault(k, []).append(0.6)
    n = sum(votes.values())
    fr.evidence["readings"] = {s: raw[k] for k, ss in srcs.items() for s in ss if "alternative" not in s}
    fr.candidates = [Candidate(value=raw[k], votes=c, sources=srcs.get(k, [])) for k, c in votes.most_common() if not k.startswith("__")]
    fr.candidates += [Candidate(value=raw[k], votes=0, sources=srcs.get(k, [])) for k in raw if k not in votes and not k.startswith("__")]
    if n == 0:
        fr.value, fr.agreement = None, 0
        fr.evidence["vote"] = "no readings"
        return
    prio = {"__illegible__": 1, "__absent__": 0}
    top, top_votes = max(votes.items(), key=lambda kv: (kv[1], prio.get(kv[0], 2)))
    fr.agreement = round(top_votes / n, 3)
    fr.evidence["vote"] = {("absent" if k == "__absent__" else "illegible" if k == "__illegible__" else str(raw[k])): c for k, c in votes.items()}
    if top == "__absent__" and top_votes == n:
        fr.evidence["absent"] = True
        fr.value, fr.confidence = None, 1.0
        return
    if top in ("__absent__", "__illegible__"):
        fr.value, fr.self_confidence, fr.confidence = None, 0.3, 0.3
        illeg = votes.get("__illegible__", 0)
        fr.reasons.append(f"{illeg} of {n} readers could not read it" if illeg else f"{top_votes} of {n} readers found nothing here; others read {', '.join(repr(c.value) for c in fr.candidates[:2])}")
        fr.evidence["unreadable"] = True
        return
    fr.value = raw[top]
    fr.evidence["agreeing_readers"] = sorted({s.split("+")[-1] for s in srcs.get(top, []) if "alternative" not in s})
    fr.self_confidence = round(sum(confs.get(top, [0])) / max(1, len(confs.get(top, []))), 3)
    fr.confidence = round(0.5 * fr.self_confidence + 0.5 * fr.agreement, 3)
    if top_votes < n:
        others = [f"{c.value!r} ({', '.join(c.sources)})" for c in fr.candidates if c.votes and norm(fr.name, c.value) != top]
        fr.reasons.append(f"Readers disagree: {top_votes} of {n} read {fr.value!r}" + (f"; others read {'; '.join(others)}" if others else "")
                          + (f"; {votes['__illegible__']} could not read it" if votes.get("__illegible__") else ""))
    if votes.get("__illegible__"):
        fr.confidence = min(fr.confidence, config.REVIEW_MIN - 0.01)


def _evidence_line(transcripts: Dict[str, str], value: Any) -> Dict[str, str]:
    out = {}
    v = reg.norm_text(value)[:8]
    if not v:
        return out
    for name, t in transcripts.items():
        for line in _plain(t).splitlines():
            if v and v in reg.norm_text(line):
                out[name] = line.strip()[:160]
                break
    return out


def agent_consensus(ctx: Ctx, passes: List[Dict[str, Any]]):
    ctx.stage(2, "Consensus started")
    parse_passes = [p for p in passes if p["kind"] == "parse"]
    tag_passes = [p for p in passes if p["kind"] == "tag"]
    if len(parse_passes) < 2:
        ctx.emit(f"Only {len(parse_passes)} structured reading came back. Without a second independent reader nothing on this order can be verified; every field goes to the pharmacist.",
                 who="consensus-agent", level="block")
    ctx.emit(f"Voting field by field across {len(passes)} model passes ({len(parse_passes)} structured parses, {len(tag_passes)} GLiNER2 taggings).", who="consensus-agent")
    fields: Dict[str, FieldResult] = {}
    for f, label in ORDER_FIELDS.items():
        fr = _new_field(f, f, label, "Order")
        readings = [(p["name"], (p["data"].get("order") or {}).get(f), None) for p in parse_passes]
        readings += [(p["name"], None, p["data"].get(f)) for p in tag_passes if p["data"].get(f)]
        _vote(fr, readings)
        fields[f] = fr
    n_lines = max((len(p["data"].get("medications") or []) for p in parse_passes), default=0)
    if parse_passes and len({len(p['data'].get('medications') or []) for p in parse_passes}) > 1:
        counts = ", ".join(f"{p['name']}: {len(p['data'].get('medications') or [])}" for p in parse_passes)
        ctx.emit(f"Readers found different numbers of medication rows ({counts}). Rows are compared in order; mismatches stay flagged.", who="consensus-agent", level="flag")
    for i in range(n_lines):
        for f, label in MED_FIELDS.items():
            key = f"med{i + 1}.{f}"
            fr = _new_field(key, f, label, f"Line {i + 1}")
            readings = []
            for p in parse_passes:
                meds = p["data"].get("medications") or []
                readings.append((p["name"], meds[i][f] if i < len(meds) else {"present": False, "legible": True, "value": "", "confidence": 1, "alternatives": [], "bbox": [0, 0, 0, 0]}, None))
            _vote(fr, readings)
            fields[key] = fr
    for fr in fields.values():
        fr.evidence["ocr"] = _evidence_line(ctx.order.ocr, fr.value or (fr.candidates[0].value if fr.candidates else ""))
    ctx.order.fields = fields
    for i in range(n_lines):
        d, dt = fields[f"med{i + 1}.drug_name"], fields[f"med{i + 1}.date"]
        name = d.value or (d.candidates[0].value if d.candidates else "unreadable drug")
        for f in MED_FIELDS:
            fields[f"med{i + 1}.{f}"].group = f"Line {i + 1} · {name}" + (f" · {dt.value}" if dt.value else "")
    split = [fr for fr in fields.values() if fr.reasons]
    for fr in split[:12]:
        ctx.emit(f"{fr.group} · {fr.label}: {fr.reasons[0]}", who="consensus-agent", level="flag", field=fr.key)
    ctx.snapshot()
    ctx.emit(f"{len(fields)} fields voted, {len(split)} with disagreement or unreadable.", who="consensus-agent", level="ok")


# ------------------------------------------------------------------ 3. verification
ISMP_PATTERNS = [(r"\bq\.?d\.?\b", "q.d."), (r"\bq\.?o\.?d\.?\b", "q.o.d."), (r"\d+\.0\s*mg\b", "trailing zero"),
                 (r"(?<![\d.])\.\d+\s*mg\b", "naked decimal"), (r"µg", "µg"), (r"\bI\.?U\.?\b", "IU")]


def _bsa(fields: Dict[str, FieldResult]) -> Optional[float]:
    v = fields.get("body_surface_area")
    if not v or not v.value:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(v.value))
    return float(m.group(1)) if m else None


def agent_verification(ctx: Ctx, prov: Provider):
    ctx.stage(3, "Verification started")
    fields = ctx.order.fields
    lines = sorted({int(k[3:].split(".")[0]) for k in fields if k.startswith("med")})
    bsa = _bsa(fields)
    rx_ok, rx_bad, lasa_hits, dose_states, sig_bad = 0, [], [], [], []
    match_cache: Dict[str, Dict[str, Any]] = {}
    match_ms: Dict[str, int] = {}
    durations: Dict[str, Dict[str, List[str]]] = {}
    # Prefetch every RxNorm lookup in parallel (disk-cached), so verification isn't a chain of round trips
    keys = sorted({norm("drug_name", c.value) for i in lines for c in fields[f"med{i}.drug_name"].candidates if c.value}
                  | {norm("drug_name", fields[f"med{i}.drug_name"].value) for i in lines if fields[f"med{i}.drug_name"].value})

    def _fetch(k):
        t0 = time.perf_counter()
        m = reg.rxnav_drug_match(k)
        reg.rxnav_candidates(k, 5)
        return k, m, int((time.perf_counter() - t0) * 1000)
    with ThreadPoolExecutor(8) as ex:
        for k, m, ms in ex.map(_fetch, keys):
            match_cache[k], match_ms[k] = m, ms
    for i in lines:
        drug, dose, sig = fields[f"med{i}.drug_name"], fields[f"med{i}.dose"], fields[f"med{i}.directions"]
        readings = [c.value for c in drug.candidates if c.value] or ([drug.value] if drug.value else [])
        # --- RxNorm match on every distinct reading
        top_match = None
        for r in readings[:3]:
            key = norm("drug_name", r)
            if key not in match_ms or match_ms[key] >= 0:
                if key not in match_cache:
                    t0 = time.perf_counter()
                    match_cache[key] = reg.rxnav_drug_match(key)
                    match_ms[key] = int((time.perf_counter() - t0) * 1000)
                ctx.m["registry"] += 1
                ms = match_ms[key]
                match_ms[key] = -1          # reported once
                mm = match_cache[key]
                ctx.m["latency"].append({"name": "RxNorm", "ms": ms, "kind": "registry"})
                if mm.get("ok"):
                    ctx.emit(f"RxNorm approximateTerm \"{r}\" → {mm['match']['name']} (similarity {mm['score']:.2f}).", "registry", model="RxNorm", ms=ms, level="ok", who="verification-agent")
                elif mm.get("ok") is None:
                    ctx.emit(f"RxNorm unreachable for \"{r}\"; the drug stays unverified.", "registry", model="RxNorm", ms=ms, level="flag", who="verification-agent")
                else:
                    best = (mm.get("match") or {}).get("name", "none")
                    ctx.emit(f"RxNorm has no confident match for \"{r}\" (best: {best}, {mm.get('score') or 0:.2f}).", "registry", model="RxNorm", ms=ms, level="flag", who="verification-agent")
            if top_match is None and drug.value and norm("drug_name", r) == norm("drug_name", drug.value):
                top_match = match_cache[key]
        if drug.value and (top_match or {}).get("ok"):
            drug.registry_score = 1.0
            rx_ok += 1
        elif drug.value or readings:
            drug.registry_score = 0.3
            rx_bad.append(drug.value or readings[0])
            if drug.value:
                drug.reasons.append(f"RxNorm has no confident match for {drug.value!r}")
        # --- rerank RxNorm candidates against the OCR line (ranking only)
        if readings and (len(readings) > 1 or not (top_match or {}).get("ok")):
            names: List[str] = []
            for r in readings[:3]:
                for c in reg.rxnav_candidates(norm("drug_name", r), 5):
                    if c["name"] not in names:
                        names.append(c["name"])
            names = names[:8]
            if names:
                ctx_line = next(iter(drug.evidence.get("ocr", {}).values()), " ".join(str(x) for x in readings))
                try:
                    ranked, meta = prov.rerank(MODELS["reranker"], ctx_line, names)
                    for rk in ranked[:3]:
                        drug.candidates.append(Candidate(value=rk["text"], votes=0, sources=[f"RxNorm, ranked #{rk['rank'] + 1} by Qwen3-Reranker"]))
                    drug.reasons.append(f"Qwen3-Reranker ranked {len(names)} RxNorm candidates; top: {ranked[0]['text']}. Ranking only, a pharmacist confirms.")
                    ctx.model_event(meta, f"Qwen3-Reranker ranked {len(names)} RxNorm candidates for line {i} against the OCR text: {ranked[0]['text']} first. Ranking only.", level="flag", who="verification-agent")
                except ModelUnavailable as e:
                    ctx.emit(f"Reranker unavailable ({e}); candidates left unranked.", "sie", model="Qwen3-Reranker", level="flag")
        # --- LASA
        if drug.value:
            partner = reg.lasa_partner(norm("drug_name", drug.value))
            if partner:
                unanimous = (drug.agreement or 0) == 1.0 and drug.registry_score == 1.0
                lasa_hits.append(drug.value)
                if not unanimous:
                    drug.reasons.append(f"LASA (look-alike/sound-alike): {drug.value} ↔ {partner}, and the readings are not unanimous")
                ctx.emit(f"LASA check: {drug.value} ↔ {partner}. " + ("Readings agree and RxNorm confirms, advisory only." if unanimous else "Readings disagree: flagged."),
                         "rules", level="ok" if unanimous else "flag", who="verification-agent", field=drug.key)
        # --- dose range (BSA-dosed agents need a body surface area)
        if dose.value and drug.value:
            chk = reg.bsa_dose_check(norm("drug_name", drug.value), str(dose.value), bsa)
            if chk is None:
                rng = reg.dose_in_range(norm("drug_name", drug.value), str(dose.value))
                if rng and rng.get("ok") is False:
                    dose.reasons.append(f"{rng['mg']} mg is outside the usual single-dose range {rng['range_mg']} mg")
                    dose_states.append("fail")
                elif rng:
                    dose_states.append("pass")
            elif chk.get("paused"):
                dose.reasons.append(f"Dose-range check paused: {chk['reason']} ({chk['drug']} {chk['range_mg_m2'][0]}–{chk['range_mg_m2'][1]} mg/m² a day)")
                dose_states.append("paused")
            elif chk.get("ok") is False:
                dose.reasons.append(f"{chk['mg_m2']} mg/m² is outside {chk['range_mg_m2'][0]}–{chk['range_mg_m2'][1]} mg/m² for {chk['drug']}")
                dose_states.append("fail")
            else:
                dose_states.append("pass")
        # --- directions
        if sig.value:
            p = reg.parse_sig(sig.value)
            if p["error_prone_abbrev"]:
                sig.reasons.append(f"ISMP error-prone abbreviation: {', '.join(p['error_prone_abbrev'])}")
                sig_bad.append(sig.key)
            m = re.search(r"(\d+)\s*(?:hours?|hrs?|h)\b", str(sig.value).lower())
            if m and drug.value:
                durations.setdefault(norm("drug_name", drug.value)[:6], {}).setdefault(m.group(1), []).append(sig.key)
        # --- date
        dt = fields[f"med{i}.date"]
        if dt.value and not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", norm("date", dt.value)):
            dt.reasons.append("Date not parseable as day/month/year")
    for drugkey, by_h in durations.items():
        if len(by_h) > 1:
            desc = " vs ".join(f"{h} h" for h in sorted(by_h, key=int, reverse=True))
            for keys in by_h.values():
                for k in keys:
                    fields[k].reasons.append(f"Infusion time differs across days for this drug: {desc}")
                    sig_bad.append(k)
            ctx.emit(f"Directions conflict for {drugkey}…: infusion over {desc}.", "rules", level="flag", who="verification-agent")
    ctx.check("rxnorm", "pass" if not rx_bad else "fail", f"{rx_ok} matched" + (f"; unconfirmed: {', '.join(map(str, rx_bad[:3]))}" if rx_bad else ""))
    ctx.check("lasa", "pass" if not any(any("LASA" in r for r in fields[f'med{i}.drug_name'].reasons) for i in lines) else "fail",
              f"{len(lasa_hits)} LASA drug line(s) checked" if lasa_hits else "No LASA drugs found")
    ctx.check("dose", "fail" if "fail" in dose_states else "paused" if "paused" in dose_states else "pass" if dose_states else "paused",
              "Out of range" if "fail" in dose_states else "Needs body surface area" if "paused" in dose_states else "Within range" if dose_states else "No checkable dose")
    if "paused" in dose_states:
        ctx.emit("Dose-range check paused: body surface area is not on the chart, so mg/m² can't be computed.", "rules", level="flag", who="verification-agent")
    ctx.check("sig", "fail" if sig_bad else "pass", f"{len(set(sig_bad))} direction field(s) flagged" if sig_bad else "Parsed")
    # --- prescriber
    pres, npi = fields["prescriber_name"], fields["prescriber_npi"]
    if npi.value:
        ok = reg.npi_valid(npi.value)
        ctx.check("npi", "pass" if ok else "fail", "Checksum valid" if ok else "Checksum fails")
        if not ok:
            npi.reasons.append("NPI fails the Luhn check digit")
    else:
        ctx.check("npi", "fail", "No NPI on the document")
        ctx.emit("No NPI or DEA number on the document. Checksums can't run.", "rules", level="flag", who="verification-agent")
    name_for_lookup = pres.value or ""
    if npi.value and reg.npi_valid(npi.value) or len(norm("prescriber_name", name_for_lookup).split()) >= 2:
        ctx.m["registry"] += 1
        t0 = time.perf_counter()
        look = reg.nppes_lookup(norm("prescriber_name", name_for_lookup), npi.value if npi.value and reg.npi_valid(npi.value) else None)
        ms = int((time.perf_counter() - t0) * 1000)
        ctx.m["latency"].append({"name": "NPPES", "ms": ms, "kind": "registry"})
        if look.get("ok") and not look.get("ambiguous"):
            pres.registry_score = 1.0
            ctx.check("nppes", "pass", f"{look.get('found_name')} · NPI {look.get('npi')}")
            ctx.emit(f"NPPES confirms {look.get('found_name')} (NPI {look.get('npi')}).", "registry", model="NPPES", ms=ms, level="ok", who="verification-agent")
        else:
            why = look.get("reason") or (f"{look.get('result_count')} records share this name" if look.get("ambiguous") else "not confirmed")
            pres.reasons.append(f"NPPES: {why}")
            ctx.check("nppes", "fail", why)
            ctx.emit(f"NPPES could not confirm the prescriber: {why}.", "registry", model="NPPES", ms=ms, level="flag", who="verification-agent")
    else:
        if pres.value:
            pres.reasons.append("Only a partial name and no NPI: NPPES can't confirm who this is")
        ctx.check("nppes", "fail", "Skipped: no NPI and no full name")
        ctx.emit("NPPES lookup skipped: no NPI and no full printed name to query.", "registry", model="NPPES", level="flag", who="verification-agent")
    # --- ISMP scan over both transcripts
    found = sorted({label for t in ctx.order.ocr.values() for pat, label in ISMP_PATTERNS if re.search(pat, _plain(t))})
    ctx.check("ismp", "fail" if found else "pass", ", ".join(found) if found else "None found")
    ctx.emit("ISMP scan: " + (f"error-prone notation found: {', '.join(found)}." if found else "no error-prone abbreviations (q.d., trailing zero, naked decimal, µg, IU)."),
             "rules", level="flag" if found else "ok", who="verification-agent")
    if fields["patient_name"].value and "redact" in str(fields["patient_name"].value).lower():
        fields["patient_name"].reasons.append("Patient identifier is redacted on this copy")
    ctx.emit(f"Verification finished: RxNorm, NPPES, checksums, LASA, dose range, sig parser, ISMP.", who="verification-agent", level="ok")


# ------------------------------------------------------------------ 4. triage
def agent_triage(ctx: Ctx, mem: memmod.Memory):
    ctx.stage(4, "Triage started")
    fields = ctx.order.fields
    src = "memory"
    label = "memory index"   # recall matches the local index; writes are mirrored into cognee when available
    ctx.m["mem_reads"] += 1
    t0 = time.perf_counter()
    proposals = []
    hits_total = 0
    for fr in fields.values():
        if fr.name not in ("drug_name", "dose", "prescriber_name", "directions"):
            continue
        readings = [c.value for c in fr.candidates if c.value] + ([fr.value] if fr.value else [])
        hits = mem.recall(fr.name, [str(r) for r in readings])
        for h in hits:
            hits_total += 1
            if h["dataset"] == "pharmacist_corrections":
                raws = list(dict.fromkeys(str(v) for v in (h.get("raw_readings") or {}).values() if str(v or "").strip()))
                text = f"Seen before: {' / '.join(repr(v) for v in raws) or 'this reading'} was resolved to {h.get('final_value')!r} by a pharmacist ({h['id']})."
                fr.candidates.append(Candidate(value=h.get("final_value"), votes=0, sources=[f"memory {h['id']}"]))
                if fr.value and norm(fr.name, fr.value) == norm(fr.name, h.get("final_value")):
                    proposals.append((fr, h))
            else:
                text = f"Agent note ({h.get('written_by')}): {h.get('note')} ({h['id']})."
            fr.memory.append({"id": h["id"], "dataset": h["dataset"], "text": text, "stored_in": h.get("stored_in", "local")})
            fr.reasons.append(text)
            if hits_total <= 6:
                ctx.emit(f"{fr.group} · {fr.label}: {text}", src, model=label, who="triage-agent", level="flag", field=fr.key, memory=fr.key)
    ms = int((time.perf_counter() - t0) * 1000)
    ctx.m["latency"].append({"name": label, "ms": ms, "kind": "memory"})
    if hits_total > 6:
        ctx.emit(f"…and {hits_total - 6} more memory hints, shown on their fields.", src, model=label, who="triage-agent")
    ctx.emit(f"Recalled {hits_total} memory record(s) from pharmacist_corrections, prescriber_patterns and agent_notes"
             + (" (records mirrored in cognee datasets)." if mem.backend == "cognee" else "."), src, model=label, ms=ms, who="triage-agent", op="read")
    # A naive memory integration would mark these verified. The choke point refuses.
    for fr, h in proposals:
        try:
            set_status(fr, "verified", "memory")
        except UnsafeVerification as e:
            ctx.m["rejected"] += 1
            if ctx.m["rejected"] <= 2:
                ctx.emit(f"REJECTED: memory proposed verified for {fr.group} · {fr.label}. {e}.", "agent", who="set_status()", level="block")
    if ctx.m["rejected"] > 2:
        ctx.emit(f"REJECTED {ctx.m['rejected'] - 2} more memory proposals to verify. Memory may add doubt, never verify.", "agent", who="set_status()", level="block")
    ctx.check("invariant", "enforced", f"{ctx.m['rejected']} attempt(s) blocked at set_status()" if ctx.m["rejected"] else "No self-verification attempted")
    # A dose or direction only means something for a confirmed drug: it can't verify on its own line otherwise
    for fr in fields.values():
        if fr.key.startswith("med") and fr.name in ("dose", "directions"):
            drug = fields.get(fr.key.split(".")[0] + ".drug_name")
            if drug is not None and not (drug.agreement == 1.0 and not drug.reasons and len(set(drug.evidence.get("agreeing_readers") or [])) >= 2):
                fr.reasons.append("The drug on this line is not confirmed, so its " + ("dose" if fr.name == "dose" else "directions") + " can't be verified on their own")
    # Deterministic decision for every field
    for fr in fields.values():
        if fr.evidence.get("absent"):
            set_status(fr, "missing", "triage_rules")
            if fr.key == "body_surface_area":
                fr.reasons.append("Needed for mg/m² dose-range checks")
            continue
        if fr.evidence.get("unreadable") or fr.value is None:
            set_status(fr, "unreadable", "triage_rules")
            continue
        conf = fr.confidence * (0.6 + 0.4 * fr.registry_score) if fr.registry_score is not None else fr.confidence
        fr.confidence = round(conf, 3)
        readers = fr.evidence.get("agreeing_readers") or []
        if len(readers) < 2:
            fr.reasons.append(f"Only one reader produced this value ({', '.join(readers) or 'none'}): no independent cross-check, so it cannot be verified")
        if fr.agreement == 1.0 and fr.confidence >= config.VERIFIED_MIN and not fr.reasons:
            set_status(fr, "verified", "triage_rules")
        elif (fr.self_confidence or 0) < config.REVIEW_MIN and (fr.agreement or 0) < 1.0:
            set_status(fr, "unreadable", "triage_rules", f"Reading confidence {fr.self_confidence:.2f} below {config.REVIEW_MIN}: value withheld, candidates shown")
        else:
            if not fr.reasons:
                fr.reasons.append(f"Confidence {fr.confidence:.2f} is below the {config.VERIFIED_MIN} verification line")
            set_status(fr, "needs_review", "triage_rules")
    queue = [k for k, fr in fields.items() if is_flag(fr)]
    ctx.order.review_queue = queue
    verified = sum(1 for fr in fields.values() if fr.status == "verified")
    worst = next((fields[k] for k in queue if fields[k].status == "unreadable"), fields[queue[0]] if queue else None)
    ctx.snapshot()
    ctx.emit(f"{len(queue)} flags open, {verified} verified. Order status: {'Blocked' if queue else 'Ready to approve'}."
             + (f" Worst first: {worst.group} · {worst.label} ({worst.status.replace('_', ' ')})." if worst else ""),
             who="triage-agent", level="flag" if queue else "ok")


# ------------------------------------------------------------------ 5. order
def agent_order(ctx: Ctx, mem: memmod.Memory):
    ctx.stage(5, "Order started")
    o = ctx.order
    o.status = "blocked" if o.review_queue else "ready"
    o.completed_at = time.time()
    # Agent note: when the two OCR readers disagree on a drug, the extraction agent records it (no patient data)
    for fr in o.fields.values():
        if fr.name == "drug_name" and len({norm('drug_name', c.value) for c in fr.candidates if c.votes}) > 1:
            readings = fr.evidence.get("readings") or {}
            try:
                rec = mem.write("agent_notes", "extraction-agent", {"field": "drug_name", "raw_readings": readings,
                                "note": "The readers disagreed on this handwritten drug name: " + "; ".join(f"{k} read {v!r}" for k, v in readings.items())})
                ctx.m["mem_writes"] += 1
                ctx.emit(f"extraction-agent wrote a failure note to agent_notes ({rec['id']}, stored in {rec['stored_in']}): readers disagreed on {fr.group}.",
                         "memory", model="cognee" if rec["stored_in"].startswith("cognee") else "memory (local)", who="extraction-agent", op="write", level="ok")
            except Exception as e:  # noqa: BLE001
                ctx.emit(f"Could not write agent note: {e}", "memory", level="flag")
            break
    o.metrics.update({"sie_calls": ctx.m["sie_calls"], "credits": round(ctx.m["credits"], 2), "registry_lookups": ctx.m["registry"],
                      "memory_reads": ctx.m["mem_reads"], "memory_writes": ctx.m["mem_writes"], "rejected_self_verifications": ctx.m["rejected"],
                      "latency": ctx.m["latency"], "fields_total": len(o.fields),
                      "verified": sum(1 for f in o.fields.values() if f.status == "verified"), "flags": len(o.review_queue), "guesses": 0})
    ctx.emit(f"Order {o.status.upper()}: {o.metrics['verified']} of {len(o.fields)} fields verified, {len(o.review_queue)} to the pharmacist. "
             f"{ctx.m['sie_calls']} SIE calls, {o.metrics['credits']:g} credits.", who="order-agent", level="ok", type="done", status=o.status, metrics=o.metrics)


# ------------------------------------------------------------------ entry
def run(order: Order, image_bytes: bytes, emit: Optional[Emit] = None, replay_speed: float = 1.0) -> Order:
    ctx = Ctx(order, emit)
    order.status = "processing"
    mem = memmod.get()
    try:
        agent_intake(ctx, image_bytes)
        try:
            prov = Provider(order.sha256, order.model_backend, replay_speed=replay_speed)
            passes = agent_extraction(ctx, prov, image_bytes)
        except ModelUnavailable as e:
            prov, passes = None, []
            ctx.emit(f"Model service unavailable: {e}", "sie", level="block", who="extraction-agent")
        if not [p for p in passes if p["kind"] == "parse"]:
            # Never fall back to guessing: every field needs review
            ctx.stage(2, "Consensus skipped")
            order.fields = {f: FieldResult(key=f, name=f, label=l, group="Order", status="checking",
                                           reasons=["No model reading was available: every field needs review"]) for f, l in ORDER_FIELDS.items()}
            for fr in order.fields.values():
                set_status(fr, "unreadable", "triage_rules")
            order.review_queue = list(order.fields)
            ctx.snapshot()
            ctx.stage(5, "Order started")
            order.status = "blocked"
            ctx.emit("No readings: every field needs review. PillSpark never falls back to guessing.", who="order-agent", level="block", type="done", status="blocked", metrics={})
            return order
        agent_consensus(ctx, passes)
        with ThreadPoolExecutor(1) as ex:           # locating runs alongside the registry checks
            locating = ex.submit(locate_fields, ctx, prov, jpeg_for(image_bytes))
            agent_verification(ctx, prov)
            locating.result()
        agent_triage(ctx, mem)
        agent_order(ctx, mem)
    except Exception as e:  # noqa: BLE001
        order.status, order.error = "failed", f"{type(e).__name__}: {str(e)[:300]}"
        ctx.emit(f"Pipeline error: {order.error}", who="order-agent", level="block", type="done", status="failed", metrics={})
    return order
