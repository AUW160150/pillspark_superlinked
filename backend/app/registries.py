"""External verification: RxNav (drug/strength), NPPES (prescriber), checksums, sig parser, safety tables.
All network calls are cached on disk so gates and demos are repeatable and fast.
Adapted from the original PillSpark prototype (github.com/AUW160150/PillSpark)."""
import json
import re
import hashlib
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import httpx

from .config import CACHE_DIR

RXNAV = "https://rxnav.nlm.nih.gov/REST"
NPPES = "https://npiregistry.cms.hhs.gov/api/"
_TIMEOUT = 8.0


def _cached_get(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    try:
        r = httpx.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data))
        return data
    except Exception as e:  # network failure → caller treats as "unverifiable", not "wrong"
        return {"_error": str(e)}


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s./-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_strength(s: Any) -> str:
    """'500mg' / '500 MG' / '0.5 g' → '500 mg'; '50mcg' → '50 mcg'."""
    t = norm_text(s).replace("milligrams", "mg").replace("micrograms", "mcg").replace("µg", "mcg").replace("ug", "mcg")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu|%)", t)
    if not m:
        return t
    num, unit = m.group(1), m.group(2)
    if unit == "g":
        num, unit = str(float(num) * 1000).rstrip("0").rstrip("."), "mg"
    if unit.startswith("unit"):
        unit = "units"
    return f"{num} {unit}"


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


# ---------------------------------------------------------------- RxNav
def rxnav_drug_match(drug_name: str) -> Dict[str, Any]:
    """Fuzzy-match a drug name against RxNorm. Returns best candidate + similarity score (0-1)."""
    if not drug_name:
        return {"ok": False, "score": 0.0, "reason": "empty"}
    data = _cached_get(f"{RXNAV}/approximateTerm.json", {"term": drug_name, "maxEntries": 8})
    if not data or "_error" in (data or {}):
        return {"ok": None, "score": None, "reason": f"rxnav unreachable: {(data or {}).get('_error')}"}
    cands = (data.get("approximateGroup") or {}).get("candidate") or []
    best = None
    for c in cands:
        name = c.get("name") or ""
        # compare against the ingredient-ish head of the candidate name
        head = re.split(r"\s\d", name)[0]
        score = max(similarity(drug_name, head), similarity(drug_name, name))
        if best is None or score > best["score"]:
            best = {"rxcui": c.get("rxcui"), "name": name, "score": round(score, 3)}
    if not best:
        sugg = _cached_get(f"{RXNAV}/spellingsuggestions.json", {"name": drug_name}) or {}
        s = ((sugg.get("suggestionGroup") or {}).get("suggestionList") or {}).get("suggestion") or []
        return {"ok": False, "score": 0.0, "reason": "no RxNorm candidate", "suggestions": s[:3]}
    sugg = _cached_get(f"{RXNAV}/spellingsuggestions.json", {"name": drug_name}) or {}
    s = ((sugg.get("suggestionGroup") or {}).get("suggestionList") or {}).get("suggestion") or []
    return {"ok": best["score"] >= 0.85, "score": best["score"], "match": best, "suggestions": s[:3]}


def _to_mg(num: str, unit: str) -> Optional[float]:
    try:
        v = float(num)
    except Exception:
        return None
    return {"mg": v, "mcg": v / 1000.0, "g": v * 1000.0}.get(unit)


def rxnav_strength_exists(drug_name: str, strength: str, form_hint: str = "") -> Dict[str, Any]:
    """Does RxNorm list a clinical drug of this ingredient at this strength? Unit-aware (75 mcg == 0.075 MG)."""
    if not drug_name or not strength:
        return {"ok": False, "reason": "missing drug or strength"}
    data = _cached_get(f"{RXNAV}/drugs.json", {"name": drug_name})
    if not data or "_error" in data:
        return {"ok": None, "reason": "rxnav unreachable"}
    groups = (data.get("drugGroup") or {}).get("conceptGroup") or []
    names: List[str] = [p.get("name") or "" for g in groups for p in (g.get("conceptProperties") or [])]
    want = norm_strength(strength)
    m = re.match(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", want)
    if not m:
        return {"ok": None, "reason": f"strength '{strength}' has no mg/mcg unit — cannot check"}
    want_mg = _to_mg(m.group(1), m.group(2))
    hits, avail_mg = [], set()
    for n in names:
        for mm in re.finditer(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", n.lower()):
            mg = _to_mg(mm.group(1), mm.group(2))
            if mg is None:
                continue
            avail_mg.add(mg)
            if abs(mg - want_mg) < 1e-6 and n not in hits:
                hits.append(n)
    def fmt(mg: float) -> str:
        return f"{mg*1000:g} mcg" if mg < 1 else f"{mg:g} mg"
    out = {"ok": bool(hits), "matches": hits[:5], "available_strengths": [fmt(x) for x in sorted(avail_mg)][:12], "concepts_checked": len(names)}
    if form_hint and hits:
        out["form_match"] = any(norm_text(form_hint).split(" ")[0] in h.lower() for h in hits)
    return out


# ---------------------------------------------------------------- NPPES
def nppes_lookup(full_name: str, npi: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
    if npi and re.fullmatch(r"\d{10}", str(npi)):
        data = _cached_get(NPPES, {"version": "2.1", "number": npi, "limit": 1})
        if data and "_error" not in data and data.get("result_count"):
            r = data["results"][0]["basic"]
            found = f"{r.get('first_name','')} {r.get('last_name','')}".strip()
            score = similarity(full_name, found) if full_name else 1.0
            return {"ok": score >= 0.6, "by": "npi", "found_name": found, "score": round(score, 3), "npi": npi,
                    "taxonomy": (data["results"][0].get("taxonomies") or [{}])[0].get("desc")}
        if data and "_error" in data:
            return {"ok": None, "reason": "nppes unreachable"}
        return {"ok": False, "by": "npi", "reason": "NPI not found in NPPES"}
    if not full_name:
        return {"ok": False, "reason": "no prescriber name"}
    parts = norm_text(re.sub(r"\b(dr|md|do|np|pa|dds|phd)\b\.?", "", full_name.lower())).split()
    if len(parts) < 2:
        return {"ok": False, "reason": "need first and last name"}
    params = {"version": "2.1", "first_name": parts[0], "last_name": parts[-1], "limit": 5}
    if state:
        params["state"] = state
    data = _cached_get(NPPES, params)
    if not data or "_error" in data:
        return {"ok": None, "reason": "nppes unreachable"}
    n = data.get("result_count", 0)
    if n == 0:
        return {"ok": False, "by": "name", "reason": "no NPPES record for that name"}
    r = data["results"][0]
    return {"ok": True, "by": "name", "result_count": n, "npi": r.get("number"),
            "found_name": f"{r['basic'].get('first_name','')} {r['basic'].get('last_name','')}",
            "taxonomy": (r.get("taxonomies") or [{}])[0].get("desc"),
            "ambiguous": n > 1}


# ---------------------------------------------------------------- checksums
def npi_valid(npi: Any) -> bool:
    s = re.sub(r"\D", "", str(npi or ""))
    if len(s) != 10:
        return False
    digits = [int(c) for c in ("80840" + s)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def dea_valid(dea: Any, last_name: str = "") -> Dict[str, Any]:
    s = re.sub(r"[^A-Za-z0-9]", "", str(dea or "")).upper()
    if not re.fullmatch(r"[ABCDEFGHMPRX][A-Z9]\d{7}", s):
        return {"ok": False, "reason": "format"}
    d = [int(c) for c in s[2:]]
    check = ((d[0] + d[2] + d[4]) + 2 * (d[1] + d[3] + d[5])) % 10
    ok = check == d[6]
    out = {"ok": ok, "reason": "checksum" if not ok else "valid"}
    if last_name and s[1] != "9" and s[1] != last_name.strip()[:1].upper():
        out["name_initial_mismatch"] = True
    return out


# ---------------------------------------------------------------- sig parsing
_FREQ = [
    (r"\b(q\.?i\.?d\.?|four times (a |per )?day|4x daily)\b", "4 times daily", False),
    (r"\b(t\.?i\.?d\.?|three times (a |per )?day|3x daily|q8h)\b", "3 times daily", False),
    (r"\b(b\.?i\.?d\.?|twice (a |per )?day|twice daily|2x daily|q12h)\b", "twice daily", False),
    (r"\b(q\.?h\.?s\.?|at bedtime|nightly|at night)\b", "at bedtime", False),
    (r"\b(q\.?d\.?)\b", "once daily", True),          # ISMP error-prone abbreviation (qd ↔ qid)
    (r"\b(once (a |per )?day|once daily|daily|every day|q24h)\b", "once daily", False),
    (r"\b(every other day|q\.?o\.?d\.?)\b", "every other day", True),
    (r"\b(p\.?r\.?n\.?|as needed)\b", "as needed", False),
    (r"\bq\s*(\d+)\s*h(ours?)?\b", "every {0} hours", False),
]
_ROUTE = [(r"\b(p\.?o\.?|by mouth|orally)\b", "oral"), (r"\b(s\.?l\.?|sublingual)\b", "sublingual"),
          (r"\b(top(ically)?)\b", "topical"), (r"\b(inh|inhale|puffs?)\b", "inhaled"),
          (r"\b(s\.?c\.?|subq|subcutaneous)\b", "subcutaneous"), (r"\b(i\.?m\.?)\b", "intramuscular")]


def parse_sig(text: Any) -> Dict[str, Any]:
    t = norm_text(text)
    out: Dict[str, Any] = {"raw": text, "dose": None, "route": None, "frequency": None, "duration": None,
                           "error_prone_abbrev": [], "parsed": False}
    if not t:
        return out
    m = re.search(r"(\d+(?:\.\d+)?|one|two|three|half|1/2)\s*(tabs?|tablets?|caps?|capsules?|ml|puffs?|drops?|units?)", t)
    if m:
        out["dose"] = f"{m.group(1)} {m.group(2)}"
    for pat, label in _ROUTE:
        if re.search(pat, t):
            out["route"] = label
            break
    for pat, label, risky in _FREQ:
        mm = re.search(pat, t)
        if mm:
            out["frequency"] = label.format(*mm.groups()) if "{0}" in label else label
            if risky:
                out["error_prone_abbrev"].append(mm.group(0))
            break
    md = re.search(r"(?:x|for)\s*(\d+)\s*(days?|d|weeks?|wks?|wk|w|months?|mo)\b", t)
    if md:
        unit = {"d": "days", "day": "days", "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "mo": "months", "month": "months"}.get(md.group(2), md.group(2))
        out["duration"] = f"{md.group(1)} {unit}"
    if re.search(r"\bu\b", t):
        out["error_prone_abbrev"].append("U (units)")
    out["parsed"] = bool(out["dose"] or out["frequency"])
    return out


# ---------------------------------------------------------------- safety tables
# Look-alike / sound-alike pairs (ISMP-style, abbreviated list for the demo)
LASA_PAIRS = [
    ("hydroxyzine", "hydralazine"), ("clonidine", "clonazepam"), ("clonidine", "klonopin"),
    ("metformin", "metronidazole"), ("lamictal", "lamisil"), ("celebrex", "celexa"),
    ("zantac", "xanax"), ("tramadol", "trazodone"), ("prednisone", "prednisolone"),
    ("glipizide", "glyburide"), ("sertraline", "cetirizine"), ("amlodipine", "amiloride"),
    ("losartan", "lorazepam"), ("buspirone", "bupropion"), ("oxycodone", "oxycontin"),
    ("fluoxetine", "fluvoxamine"), ("cephalexin", "cefazolin"), ("lisinopril", "lipitor"),
    # ISMP tall-man anthracyclines
    ("daunorubicin", "doxorubicin"), ("daunorubicin", "idarubicin"),
]


def lasa_partner(drug: str) -> Optional[str]:
    d = norm_text(drug)
    for a, b in LASA_PAIRS:
        if d.startswith(a):
            return b
        if d.startswith(b):
            return a
    return None


# typical adult single-dose range in mg (min, max) — a dose outside → needs_review
DOSE_RANGES_MG = {
    "amoxicillin": (250, 1000), "metformin": (500, 1000), "lisinopril": (2.5, 40), "atorvastatin": (10, 80),
    "amlodipine": (2.5, 10), "omeprazole": (10, 40), "sertraline": (25, 200), "gabapentin": (100, 800),
    "prednisone": (1, 60), "azithromycin": (250, 600), "hydrochlorothiazide": (12.5, 50), "losartan": (25, 100),
    "metoprolol": (25, 200), "ibuprofen": (200, 800), "cephalexin": (250, 1000), "fluoxetine": (10, 80),
    "montelukast": (4, 10), "pantoprazole": (20, 80), "escitalopram": (5, 20), "hydroxyzine": (10, 100),
    "hydralazine": (10, 100), "tramadol": (50, 100), "clonidine": (0.1, 0.3), "levothyroxine": (0.025, 0.3),
    "simvastatin": (5, 80), "furosemide": (20, 80), "warfarin": (1, 10), "alprazolam": (0.25, 2),
}


def dose_in_range(drug: str, strength: str) -> Optional[Dict[str, Any]]:
    d = norm_text(drug).split(" ")[0] if drug else ""
    rng = DOSE_RANGES_MG.get(d)
    if not rng:
        return None
    s = norm_strength(strength)
    m = re.match(r"(\d+(?:\.\d+)?)\s*(mg|mcg)", s)
    if not m:
        return {"ok": None, "reason": "strength not parseable"}
    mg = float(m.group(1)) / (1000 if m.group(2) == "mcg" else 1)
    ok = rng[0] <= mg <= rng[1]
    return {"ok": ok, "mg": mg, "range_mg": rng}


# Body-surface-area dosed agents (mg/m² per day, 7+3 induction reference ranges). A per-m² dose can only be
# range-checked when the chart gives a body surface area; without one the check is paused, never skipped.
BSA_DOSED_MG_M2 = {"daunorubicin": (45, 90), "cytarabine": (100, 200), "idarubicin": (10, 12)}


def bsa_dose_check(drug: str, dose: str, bsa_m2: Optional[float]) -> Optional[Dict[str, Any]]:
    d = norm_text(drug).split(" ")[0] if drug else ""
    # match full names and handwritten abbreviations ("dauno", "cytara"), at least 4 letters
    key = next((k for k in BSA_DOSED_MG_M2 if len(d) >= 4 and (d.startswith(k[:6]) or k.startswith(d))), None)
    if not key:
        return None
    rng = BSA_DOSED_MG_M2[key]
    m = re.match(r"(\d+(?:\.\d+)?)\s*mg", norm_strength(dose or ""))
    if not m:
        return {"ok": None, "drug": key, "reason": "dose not parseable"}
    mg = float(m.group(1))
    if not bsa_m2:
        return {"ok": None, "drug": key, "paused": True, "range_mg_m2": rng, "mg": mg,
                "reason": "body surface area missing, so mg/m² cannot be computed"}
    per_m2 = mg / bsa_m2
    return {"ok": rng[0] <= per_m2 <= rng[1], "drug": key, "mg": mg, "mg_m2": round(per_m2, 1), "range_mg_m2": rng}


def rxnav_candidates(term: str, n: int = 6) -> List[Dict[str, Any]]:
    """RxNorm approximateTerm candidates (rxcui + name), for the reranker to order."""
    data = _cached_get(f"{RXNAV}/approximateTerm.json", {"term": term, "maxEntries": n})
    if not data or "_error" in data:
        return []
    out, seen = [], set()
    for c in (data.get("approximateGroup") or {}).get("candidate") or []:
        name = c.get("name")
        if not name:
            props = _cached_get(f"{RXNAV}/rxcui/{c.get('rxcui')}/properties.json", {}) or {}
            name = ((props.get("properties") or {}).get("name"))
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"rxcui": c.get("rxcui"), "name": name})
    return out
