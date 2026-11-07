"""The single choke point for field status changes.

Invariant (guesses = 0): models, the reranker and memory may add candidates, add reasons, raise flags
or rank options. They may never mark a field verified. Only the deterministic triage rules (agreement
between at least two independent readers, i.e. readings built on both OCR transcripts, plus every
registry/safety check passing) or a pharmacist can.
"""
from typing import Optional

from .schemas import FieldResult

# Sources allowed to set "verified". "triage_rules" = deterministic agreement + checks.
TRUSTED_VERIFIERS = {"triage_rules"}
# Sources allowed to set "resolved" (a human decision).
HUMAN = {"pharmacist"}
FLAG_STATUSES = {"needs_review", "unreadable", "missing"}


class UnsafeVerification(Exception):
    """Raised when an untrusted source tries to verify a field."""


def set_status(field: FieldResult, status: str, source: str, reason: Optional[str] = None) -> FieldResult:
    if status == "verified" and source not in TRUSTED_VERIFIERS:
        raise UnsafeVerification(f"{source} may add doubt, never verify ({field.key})")
    if status == "resolved" and source not in HUMAN:
        raise UnsafeVerification(f"only a pharmacist can resolve a flag ({field.key}); got {source}")
    if status == "verified" and field.reasons:
        raise UnsafeVerification(f"{field.key} has open reasons and cannot be verified")
    if status == "verified" and len(set(field.evidence.get("agreeing_readers") or [])) < 2:
        raise UnsafeVerification(f"{field.key} was read by fewer than two independent readers; no cross-check, so it cannot be verified")
    if status in ("unreadable", "missing"):
        field.value = None                      # withhold: never show a guess as the value
    field.status = status
    field.status_source = source
    if reason:
        field.reasons.append(reason)
    return field


def is_flag(field: FieldResult) -> bool:
    return field.status in FLAG_STATUSES
