from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field as PField

FIELD_STATUSES = ("pending", "checking", "verified", "needs_review", "unreadable", "missing", "resolved")
ORDER_STATUSES = ("queued", "processing", "blocked", "ready", "approved", "failed")


class Candidate(BaseModel):
    value: Any = None
    votes: int = 0
    sources: List[str] = PField(default_factory=list)


class FieldResult(BaseModel):
    key: str                                # e.g. "prescriber_name" or "med1.dose"
    name: str                               # base field name, e.g. "dose"
    label: str
    group: str                              # "Order" or "Medication 1"
    value: Optional[Any] = None             # withheld (None) when unreadable/missing: never a guess
    confidence: float = 0.0
    status: str = "pending"
    status_source: Optional[str] = None     # who set the current status (see safety.set_status)
    reasons: List[str] = PField(default_factory=list)
    candidates: List[Candidate] = PField(default_factory=list)
    evidence: Dict[str, Any] = PField(default_factory=dict)
    memory: List[Dict[str, Any]] = PField(default_factory=list)
    agreement: Optional[float] = None
    self_confidence: Optional[float] = None
    registry_score: Optional[float] = None
    resolved_value: Optional[Any] = None
    resolved_by: Optional[str] = None
    resolution_reason: Optional[str] = None


class Order(BaseModel):
    order_id: str
    filename: str
    sha256: str
    created_at: float
    status: str = "queued"
    stage: int = -1
    model_backend: str = "mock"
    fields: Dict[str, FieldResult] = PField(default_factory=dict)
    review_queue: List[str] = PField(default_factory=list)
    metrics: Dict[str, Any] = PField(default_factory=dict)
    ocr: Dict[str, str] = PField(default_factory=dict)     # model -> transcript (no image bytes)
    error: Optional[str] = None
    completed_at: Optional[float] = None


class ResolveRequest(BaseModel):
    value: Any
    reason: str


class ApproveRequest(BaseModel):
    note: Optional[str] = None
