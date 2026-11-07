"""PillSpark API: real-time document intake.

POST /api/intake                      upload an image (multipart) or {"sample": name}; Idempotency-Key header honoured
GET  /api/orders                      queue
GET  /api/orders/{id}                 order with fields
GET  /api/orders/{id}/events          server-sent events: replays stored events, then streams live ones
POST /api/orders/{id}/fields/{key}/resolve   pharmacist resolves a flag (reason required)
POST /api/orders/{id}/approve         refused while any flag is open
GET  /api/orders/{id}/audit           append-only audit trail
GET  /api/orders/{id}/export?format=json|fhir
GET  /api/memory                      what the agents have learned
GET  /healthz, /readyz
/                                     the landing page and live review screen
"""
import asyncio
import hashlib
import io
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, export, memory as memmod, pipeline, store
from .safety import UnsafeVerification, is_flag, set_status
from .schemas import ApproveRequest, Order, ResolveRequest

log = logging.getLogger("pillspark")
logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}')
for noisy in ("httpx", "httpcore", "sie_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)   # request URLs can carry names; keep them out of logs


def jlog(**kw):
    log.info(json.dumps(kw, default=str))


app = FastAPI(title="PillSpark", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=[config.APP_ORIGIN], allow_methods=["GET", "POST"], allow_headers=["*"])


# ------------------------------------------------------------------ live event broker
class Broker:
    def __init__(self):
        self.subs: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def publish(self, order_id: str, ev: Dict[str, Any]):
        """Called from pipeline threads: persist, then hand to the event loop."""
        store.add_event(order_id, ev)
        if self.loop:
            for q in list(self.subs.get(order_id, [])):
                self.loop.call_soon_threadsafe(q.put_nowait, ev)


broker = Broker()
jobs: "asyncio.Queue[str]" = asyncio.Queue()
_images: Dict[str, bytes] = {}
REPLAY_SPEED = float(os.environ.get("PILLSPARK_REPLAY_SPEED", "1.0"))  # mock mode: fraction of recorded latency


async def worker():
    while True:
        order_id = await jobs.get()
        o = store.get_order(order_id)
        try:
            img = _images.pop(order_id, None) or (config.UPLOAD_DIR / f"{o.sha256}.img").read_bytes()
            jlog(event="job_start", order_id=order_id, backend=o.model_backend)

            def emit(ev):
                if ev.get("type") in ("fields", "done"):
                    store.save_order(o)          # persist before announcing, so readers never see stale state
                broker.publish(order_id, ev)
            await asyncio.to_thread(pipeline.run, o, img, emit, REPLAY_SPEED)
            store.save_order(o)
            for k, f in o.fields.items():   # every field decision is an audit event
                store.audit(order_id, f.status_source or "triage-agent", "field_status", k, "checking", f.status,
                            "; ".join(f.reasons)[:300] or None)
            store.audit(order_id, "triage-agent", "status", None, "processing", o.status, f"{len(o.review_queue)} flags")
            jlog(event="job_done", order_id=order_id, status=o.status, flags=len(o.review_queue))
        except Exception as e:  # noqa: BLE001
            o.status, o.error = "failed", f"{type(e).__name__}"
            store.save_order(o)
            broker.publish(order_id, {"type": "done", "status": "failed", "msg": "Processing failed. The order needs manual entry.", "src": "agent", "level": "block"})
            jlog(event="job_failed", order_id=order_id, error=type(e).__name__)
        finally:
            jobs.task_done()


@app.on_event("startup")
async def _startup():
    store.init()
    broker.loop = asyncio.get_running_loop()
    memmod.get()
    asyncio.create_task(worker())


# ------------------------------------------------------------------ rate limit (per client IP, sliding window)
_hits: Dict[str, deque] = defaultdict(deque)
_live_runs: deque = deque()
MAX_LIVE_RUNS_PER_HOUR = int(os.environ.get("PILLSPARK_MAX_LIVE_RUNS_PER_HOUR", "30"))


def _rate_limit(request: Request, limit: int = 20, window: float = 60.0):
    ip = request.client.host if request.client else "?"
    q = _hits[ip]
    now = time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many uploads. Try again in a minute.")
    q.append(now)


def _actor(x_user: Optional[str]) -> str:
    return (x_user or "pharmacist.demo").strip()[:60]


# ------------------------------------------------------------------ intake
@app.post("/api/intake")
async def intake(request: Request, file: Optional[UploadFile] = File(None), idempotency_key: Optional[str] = Header(None)):
    _rate_limit(request)
    if idempotency_key:
        existing = store.find_by_idem(idempotency_key)
        if existing:
            return {"order_id": existing, "duplicate": True}
    if file is not None:
        data = await file.read()
        filename = file.filename or "upload"
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Send a file (multipart) or {\"sample\": name}.")
        name = str(body.get("sample") or "")
        path = config.SAMPLES_DIR / name
        if not name or "/" in name or ".." in name or not path.exists():
            raise HTTPException(404, "Unknown sample.")
        data, filename = path.read_bytes(), name
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 15 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").lower()
        im.verify()
    except Exception:
        raise HTTPException(415, "Not a readable image. Upload JPEG, PNG or WebP.")
    if fmt not in ("jpeg", "png", "webp"):
        raise HTTPException(415, f"Unsupported image type {fmt!r}. Upload JPEG, PNG or WebP.")
    sha = hashlib.sha256(data).hexdigest()
    backend = config.MODEL_BACKEND
    if backend == "sie" and not config.SIE_API_KEY:
        backend = "mock"
    # Public deployments: cap live SIE runs per hour so a public URL can't drain the credits.
    now = time.time()
    while _live_runs and now - _live_runs[0] > 3600:
        _live_runs.popleft()
    if backend == "sie" and len(_live_runs) >= MAX_LIVE_RUNS_PER_HOUR:
        if file is None and (config.FIXTURE_DIR / f"{sha[:16]}.json").exists():
            backend = "mock"           # the sample replays its recorded live run, labelled as a replay
        else:
            raise HTTPException(429, f"The live demo is limited to {MAX_LIVE_RUNS_PER_HOUR} live runs an hour. Try the sample chart, or again later.")
    if backend == "sie":
        _live_runs.append(now)
    (config.UPLOAD_DIR / f"{sha}.img").write_bytes(data)
    o = Order(order_id=f"PS-{uuid.uuid4().hex[:6].upper()}", filename=filename, sha256=sha, created_at=time.time(), model_backend=backend)
    store.save_order(o, idempotency_key)
    store.audit(o.order_id, "intake-agent", "created", None, None, "queued", f"{filename} ({len(data) // 1024} KB)")
    _images[o.order_id] = data
    await jobs.put(o.order_id)
    jlog(event="intake", order_id=o.order_id, bytes=len(data), backend=backend)
    return {"order_id": o.order_id, "status": "queued", "model_backend": backend, "queue_position": jobs.qsize()}


@app.get("/api/samples")
def samples():
    return {"samples": sorted(p.name for p in config.SAMPLES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))}


@app.get("/api/orders")
def orders():
    return {"orders": store.list_orders()}


def _get(order_id: str) -> Order:
    o = store.get_order(order_id)
    if not o:
        raise HTTPException(404, "Order not found.")
    return o


@app.get("/api/orders/{order_id}")
def order(order_id: str):
    return _get(order_id).model_dump()


@app.get("/api/orders/{order_id}/events")
async def events(order_id: str, request: Request, after: int = 0):
    _get(order_id)
    q: asyncio.Queue = asyncio.Queue()
    broker.subs[order_id].append(q)

    async def gen():
        last = after
        try:
            for ev in store.events_after(order_id, after):
                last = ev["seq"]
                yield f"data: {json.dumps(ev, default=str)}\n\n"
                if ev.get("type") == "done":
                    return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if ev.get("seq", 0) <= last:
                    continue
                last = ev.get("seq", last)
                yield f"data: {json.dumps(ev, default=str)}\n\n"
                if ev.get("type") == "done":
                    return
        finally:
            broker.subs[order_id].remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ pharmacist actions
def _ev(order_id: str, msg: str, src: str, **kw) -> Dict[str, Any]:
    ev = {"type": "event", "src": src, "msg": msg, **{k: v for k, v in kw.items() if v is not None}}
    broker.publish(order_id, ev)
    return ev


@app.post("/api/orders/{order_id}/fields/{key}/resolve")
async def resolve(order_id: str, key: str, body: ResolveRequest, x_user: Optional[str] = Header(None), x_role: Optional[str] = Header(None)):
    role = (x_role or "pharmacist").lower()
    if role != "pharmacist":
        raise HTTPException(403, "Only a pharmacist can resolve a flag.")
    if not body.reason or not body.reason.strip():
        raise HTTPException(422, "Choose a reason for the change.")
    o = _get(order_id)
    if o.status not in ("blocked", "ready"):
        raise HTTPException(409, f"Order is {o.status}; nothing to resolve.")
    fr = o.fields.get(key)
    if not fr:
        raise HTTPException(404, "Field not found.")
    if not is_flag(fr):
        raise HTTPException(409, "This field is not flagged.")
    actor = _actor(x_user)
    before = fr.value if fr.value is not None else f"[{fr.status}]"
    readings = dict(fr.evidence.get("readings") or {})
    set_status(fr, "resolved", "pharmacist")
    fr.resolved_value, fr.resolved_by, fr.resolution_reason = body.value, actor, body.reason.strip()
    o.review_queue = [k for k in o.review_queue if k != key]
    out = [_ev(order_id, f"Resolved {fr.group} · {fr.label} → {body.value} ({body.reason.strip()}).", "pharmacist", who=actor, level="ok", field=key)]
    store.audit(order_id, actor, "resolve", key, before, body.value, body.reason.strip())
    out.append(_ev(order_id, f"audit: {actor} {fr.label}: {before} → {body.value}", "agent", type="audit", quiet=True))
    # Memory write: patient fields never; free text through SIE pii-redact first
    mem = memmod.get()
    if fr.name in config.PHI_FIELDS:
        out.append(_ev(order_id, "Patient field: not written to memory (memory never holds patient identifiers).", "memory", level="ok", who="pharmacist"))
    else:
        reason_text = body.reason.strip()
        try:
            from .sie import ModelUnavailable, Provider
            prov = Provider(o.sha256, o.model_backend, replay_speed=0)
            reason_text, meta, n = await asyncio.to_thread(memmod.redact_text, prov, reason_text)
            if meta:
                out.append(_ev(order_id, f"pii-redact scrubbed the correction text before it reaches memory ({n} span(s) removed); patient fields stripped structurally.",
                               "sie", model="pii-redact", model_id=config.MODELS["pii"], ms=meta["ms"], credits=meta.get("credits"), replayed=meta.get("replayed") or None, level="ok"))
        except Exception as e:  # noqa: BLE001
            reason_text = "[reason withheld: PII redaction unavailable]"
            out.append(_ev(order_id, f"pii-redact unavailable ({type(e).__name__}); the free-text reason was withheld from memory.", "sie", model="pii-redact", level="flag"))
        prescriber = o.fields.get("prescriber_name")
        rec = await asyncio.to_thread(mem.write, "pharmacist_corrections", "pharmacist", {
            "field": fr.name, "raw_readings": readings, "final_value": body.value, "reason": reason_text,
            "prescriber": (prescriber.value or prescriber.resolved_value) if prescriber else None, "order_ref": order_id})
        label = "cognee" if rec["stored_in"].startswith("cognee") else "memory (local)"
        out.append(_ev(order_id, f"Wrote to pharmacist_corrections ({rec['id']}, stored in {rec['stored_in']}): field, each model's reading, final value, reason. No patient identifiers.",
                       "memory", model=label, who="pharmacist", op="write", level="ok"))
        if fr.memory:
            accepted = any(str(m.get("text", "")).find(repr(body.value)) >= 0 for m in fr.memory)
            fb = await asyncio.to_thread(mem.feedback, [m["id"] for m in fr.memory], accepted, fr.name)
            out.append(_ev(order_id, f"Feedback {fb['id']}: the memory suggestion was {'accepted' if accepted else 'not used'} by the pharmacist. "
                           "Accepted suggestions rank first next time; ones ignored repeatedly fade out.", "memory", model=label, who="feedback", op="write", level="ok"))
    if not o.review_queue:
        o.status = "ready"
        store.audit(order_id, "triage-agent", "status", None, "blocked", "ready", "all flags resolved")
        out.append(_ev(order_id, "All flags resolved. Order status: Ready to approve.", "agent", who="triage-agent", level="ok", order_status="ready"))
    store.save_order(o)
    return {"order": o.model_dump(), "events": out}


@app.post("/api/orders/{order_id}/approve")
async def approve(order_id: str, body: ApproveRequest = ApproveRequest(), x_user: Optional[str] = Header(None), x_role: Optional[str] = Header(None)):
    if (x_role or "pharmacist").lower() != "pharmacist":
        raise HTTPException(403, "Only a pharmacist can approve an order.")
    o = _get(order_id)
    open_flags = [k for k, f in o.fields.items() if is_flag(f)]
    if open_flags or o.status != "ready":
        raise HTTPException(409, f"Approve is disabled: {len(open_flags)} field(s) still need a pharmacist.")
    actor = _actor(x_user)
    o.status = "approved"
    store.save_order(o)
    store.audit(order_id, actor, "approve", None, "ready", "approved", body.note)
    out = [_ev(order_id, "Order approved.", "pharmacist", who=actor, level="ok", order_status="approved")]
    store.audit(order_id, "order-agent", "export", None, None, "json,csv,fhir", "MedicationRequest bundle generated")
    n = len(export.fhir_bundle(o)["entry"])
    out.append(_ev(order_id, f"Exports ready: JSON, CSV and a FHIR R4 Bundle with {n} MedicationRequest resource(s). Nothing was sent anywhere.", "agent", who="order-agent", level="ok"))
    return {"order": o.model_dump(), "events": out}


@app.get("/api/orders/{order_id}/audit")
def audit(order_id: str):
    _get(order_id)
    return {"audit": store.audit_for(order_id)}


@app.get("/api/orders/{order_id}/export")
def export_order(order_id: str, format: str = "json"):
    o = _get(order_id)
    if o.status != "approved":
        raise HTTPException(409, "Only approved orders can be exported.")
    if format == "fhir":
        return JSONResponse(export.fhir_bundle(o))
    if format == "csv":
        return StreamingResponse(iter([export.csv_text(o)]), media_type="text/csv")
    return JSONResponse(export.order_json(o))


@app.get("/api/memory")
def memory_view():
    mem = memmod.get()
    return {"stats": mem.stats(), "records": {ds: mem.all(ds)[-50:] for ds in memmod.DATASETS}}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    mem = memmod.get()
    return {"ok": True, "model_backend": config.MODEL_BACKEND, "sie_key": bool(config.SIE_API_KEY), "memory": mem.backend,
            "queue": jobs.qsize(), "models": config.MODELS}


app.mount("/", StaticFiles(directory=str(config.LANDING_DIR), html=True), name="landing")
