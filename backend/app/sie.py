"""Model layer on Superlinked SIE.

Every call returns (value, meta) where meta = {model, ms, credits, served_by, replayed}. Live calls are
recorded to fixtures/<image sha>.json so the same document can be replayed offline (MODEL_BACKEND=mock),
which is labelled as a replay in every event. Image bytes are never logged or put into events.
"""
import base64
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config

Meta = Dict[str, Any]
_FIXTURE_LOCK = threading.Lock()


class ModelUnavailable(Exception):
    pass


def _key(kind: str, model: str, payload: str = "") -> str:
    return f"{kind}|{model}|{hashlib.sha1(payload.encode()).hexdigest()[:12]}"


def _credits(resp: Any) -> Optional[float]:
    try:
        req = resp.get("request") or {}
        usage = req.get("usage") or resp.get("usage") or {}
        c = usage.get("credits_charged") or req.get("credits_debited")
        return float(c) if c is not None else None
    except Exception:
        return None


class Provider:
    def __init__(self, image_sha: str, backend: Optional[str] = None, replay_speed: float = 1.0):
        self.backend = backend or config.MODEL_BACKEND
        self.sha = image_sha
        self.replay_speed = replay_speed
        self.fixture_path = config.FIXTURE_DIR / f"{image_sha[:16]}.json"
        self._lock = threading.Lock()
        self._recorded: Dict[str, Any] = {}
        if self.backend == "mock":
            if not self.fixture_path.exists():
                raise ModelUnavailable("no recorded SIE outputs for this document (mock mode replays recorded live runs only)")
            self._recorded = json.loads(self.fixture_path.read_text())
            self.client = None
        else:
            if not config.SIE_API_KEY and not config.SIE_BASE_URL:
                raise ModelUnavailable("SIE_API_KEY is not set")
            from sie_sdk import SIEClient
            self.client = SIEClient(api_key=config.SIE_API_KEY or None, base_url=config.SIE_BASE_URL, timeout_s=config.SIE_TIMEOUT_S)

    # ----------------------------------------------------------------- plumbing
    def _call(self, key: str, model: str, fn) -> Tuple[Any, Meta]:
        if self.backend == "mock":
            rec = self._recorded.get(key)
            if rec is None:
                raise ModelUnavailable(f"no recorded output for {key}")
            if self.replay_speed:
                time.sleep(rec["meta"]["ms"] / 1000 * self.replay_speed)
            return rec["value"], {**rec["meta"], "replayed": True}
        last: Optional[Exception] = None
        for attempt in range(2):                      # one retry with backoff; SDK also retries 429/503
            t0 = time.perf_counter()
            try:
                value, raw = fn()
                meta = {"model": model, "ms": int((time.perf_counter() - t0) * 1000), "credits": _credits(raw),
                        "served_by": ((raw.get("request") or {}).get("served_by") if isinstance(raw, dict) else None), "replayed": False}
                if config.RECORD_FIXTURES:
                    with _FIXTURE_LOCK:   # merge with what is on disk: several providers may record one document
                        disk = json.loads(self.fixture_path.read_text()) if self.fixture_path.exists() else {}
                        disk[key] = {"value": value, "meta": {k: meta[k] for k in ("model", "ms", "credits")}}
                        self.fixture_path.write_text(json.dumps(disk, indent=1))
                return value, meta
            except Exception as e:  # noqa: BLE001 - surfaced as a model failure, never as a guess
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise ModelUnavailable(f"{model}: {type(last).__name__}: {str(last)[:200]}")

    # ----------------------------------------------------------------- jobs
    def ocr(self, model: str, png: bytes) -> Tuple[str, Meta]:
        def fn():
            from sie_sdk.types import Item
            r = self.client.extract(model, Item(id="page1", images=[png]), provision_timeout_s=config.SIE_TIMEOUT_S)
            r = dict(r)
            text = next((e.get("text", "") for e in (r.get("entities") or []) if e.get("label") == "markdown"), None)
            if text is None:
                raise RuntimeError("OCR reply had no markdown entity")
            return text, r
        return self._call(_key("ocr", model), model, fn)

    def tag_fields(self, model: str, text: str, schema: Dict[str, Any]) -> Tuple[Dict[str, str], Meta]:
        def fn():
            from sie_sdk.types import Item
            try:
                r = dict(self.client.extract(model, Item(id="w", text=text), output_schema=schema, provision_timeout_s=config.SIE_TIMEOUT_S))
            except Exception as e:  # SIE answers "nothing found" with HTTP 500 + this message (cookbook note)
                if "conserved billing settlement" in str(e):
                    return {}, {}
                raise
            data = r.get("data") if isinstance(r.get("data"), dict) else {}
            return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}, r
        return self._call(_key("tag", model, text), model, fn)

    def parse(self, model: str, jpeg: bytes, transcript: str, schema: Dict[str, Any], prompt: str, tag: str) -> Tuple[Dict[str, Any], Meta]:
        uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

        def fn():
            content = [{"type": "image_url", "image_url": {"url": uri}},
                       {"type": "text", "text": prompt + "\n\nOCR transcript (may contain errors; the image is the source of truth):\n" + transcript[:4500]}]
            r = self.client.chat_completions(model, [{"role": "user", "content": content}], max_completion_tokens=2800, temperature=0.0,   # 8,192-token window: image counts ~3.1k, prompt ~2k
                                             response_format={"type": "json_schema", "json_schema": {"name": "order", "schema": schema, "strict": True}},
                                             provision_timeout_s=config.SIE_TIMEOUT_S)
            r = dict(r)
            if r["choices"][0].get("finish_reason") == "length":
                raise RuntimeError("structured output was truncated (token limit)")
            text = r["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            return json.loads(text), r
        schema_v = hashlib.sha1(json.dumps(schema, sort_keys=True).encode() + prompt.encode()).hexdigest()[:8]
        return self._call(_key("parse", model, schema_v + tag + transcript[:4500]), model, fn)

    def locate(self, model: str, jpeg: bytes, prompt: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Meta]:
        uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

        def fn():
            r = dict(self.client.chat_completions(model, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": prompt}]}],
                                                  max_completion_tokens=1500, temperature=0.0,
                                                  response_format={"type": "json_schema", "json_schema": {"name": "locate", "schema": schema, "strict": True}},
                                                  provision_timeout_s=config.SIE_TIMEOUT_S))
            if r["choices"][0].get("finish_reason") == "length":
                raise RuntimeError("box output was truncated (token limit)")
            text = re.sub(r"^```(?:json)?|```$", "", r["choices"][0]["message"]["content"].strip()).strip()
            return json.loads(text), r
        return self._call(_key("locate", model, prompt), model, fn)

    def rerank(self, model: str, query: str, docs: List[str]) -> Tuple[List[Dict[str, Any]], Meta]:
        def fn():
            r = dict(self.client.score(model, {"text": query}, [{"id": str(i), "text": d} for i, d in enumerate(docs)], provision_timeout_s=config.SIE_TIMEOUT_S))
            out = [{"text": docs[int(s["item_id"])], "score": float(s["score"]), "rank": int(s["rank"])} for s in r["scores"]]
            return out, r
        return self._call(_key("rerank", model, query + "||" + "|".join(docs)), model, fn)

    def pii(self, model: str, text: str, labels: List[str]) -> Tuple[List[Dict[str, Any]], Meta]:
        def fn():
            from sie_sdk.types import Item
            r = dict(self.client.extract(model, Item(id="pii", text=text), labels=labels, provision_timeout_s=config.SIE_TIMEOUT_S))
            ents = [{"text": e.get("text"), "label": e.get("label"), "start": e.get("start"), "end": e.get("end"), "score": e.get("score")}
                    for e in (r.get("entities") or [])]
            return ents, r
        return self._call(_key("pii", model, text), model, fn)


def to_png(image_bytes: bytes, max_edge: int = 2048) -> bytes:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")   # match what browsers display
    im.thumbnail((max_edge, max_edge))
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue()


def to_jpeg(image_bytes: bytes, max_edge: int = 1600) -> bytes:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    im.thumbnail((max_edge, max_edge))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=88)
    return b.getvalue()
