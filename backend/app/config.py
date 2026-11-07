"""Settings. Secrets come only from the environment (or pillspark/.env for local runs)."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.environ.get("PILLSPARK_DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "registry_cache"
FIXTURE_DIR = BACKEND_DIR / "fixtures"
SAMPLES_DIR = BACKEND_DIR / "samples"
LANDING_DIR = ROOT_DIR / "landing"
DB_PATH = DATA_DIR / "pillspark.db"
for d in (DATA_DIR, UPLOAD_DIR, CACHE_DIR, FIXTURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# live = real SIE calls; mock = replay recorded SIE outputs (offline, deterministic, labelled in the UI)
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "sie" if os.environ.get("SIE_API_KEY") else "mock")
SIE_API_KEY = os.environ.get("SIE_API_KEY", "").strip()
SIE_BASE_URL = os.environ.get("SIE_BASE_URL", "").strip() or "https://api.superlinked.com"
SIE_TIMEOUT_S = float(os.environ.get("SIE_TIMEOUT_S", "180"))
RECORD_FIXTURES = os.environ.get("PILLSPARK_RECORD", "1") == "1"   # save live SIE outputs for offline replay

MODELS = yaml.safe_load((BACKEND_DIR / "models.yaml").read_text())

VERIFIED_MIN = float(os.environ.get("PILLSPARK_VERIFIED_MIN", "0.90"))
REVIEW_MIN = float(os.environ.get("PILLSPARK_REVIEW_MIN", "0.70"))

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

# Memory: cognee when it can be initialised, otherwise a local store (labelled in events)
MEMORY_BACKEND = os.environ.get("MEMORY_BACKEND", "auto")   # auto | cognee | local
APP_ORIGIN = os.environ.get("APP_ORIGIN", "http://127.0.0.1:8000")

# Order-level fields and per-medication-line fields
ORDER_FIELDS = {
    "patient_name": "Patient name",
    "diagnosis": "Diagnosis",
    "regimen": "Regimen",
    "body_surface_area": "Body surface area",
    "prescriber_name": "Prescriber",
    "prescriber_npi": "Prescriber NPI",
}
MED_FIELDS = {
    "drug_name": "Drug",
    "dose": "Dose",
    "directions": "Directions (sig)",
    "date": "Date",
}
PHI_FIELDS = {"patient_name", "patient_dob", "patient_id", "patient_address", "patient_phone"}
