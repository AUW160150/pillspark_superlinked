"""End-to-end check over HTTP against a real uvicorn process.

  --mode live : real Superlinked SIE calls (records outputs to fixtures/ for offline replay)
  --mode mock : replays the recorded SIE outputs (no network to SIE)

Uploads the redacted sample chart, reads the SSE stream, resolves every flag as a pharmacist,
approves, checks the audit trail, exports FHIR, and checks idempotency.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
SAMPLE = "chemo_chart_redacted.jpg"


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def fail(msg):
    print("FAIL:", msg); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "mock"], required=True)
    ap.add_argument("--keep", action="store_true", help="leave the server running and print its URL")
    a = ap.parse_args()
    port = free_port()
    data_dir = tempfile.mkdtemp(prefix="pillspark-e2e-")
    env = {**os.environ, "MODEL_BACKEND": "sie" if a.mode == "live" else "mock", "PILLSPARK_DATA_DIR": data_dir,
           "MEMORY_BACKEND": os.environ.get("MEMORY_BACKEND", "local"), "PILLSPARK_REPLAY_SPEED": "0.05",
           "APP_ORIGIN": f"http://127.0.0.1:{port}"}
    srv = subprocess.Popen([str(PY), "-m", "uvicorn", "app.main:app", "--app-dir", str(ROOT / "backend"), "--port", str(port), "--log-level", "warning"],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            try:
                if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            fail("server did not start: " + (srv.stdout.read() if srv.poll() is not None else ""))
        c = httpx.Client(base_url=base, timeout=60)
        ready = c.get("/readyz").json()
        print("readyz:", {k: ready[k] for k in ("model_backend", "memory", "sie_key")})
        r = c.post("/api/intake", json={"sample": SAMPLE}, headers={"Idempotency-Key": "e2e-1"})
        if r.status_code != 200:
            fail(f"intake {r.status_code} {r.text}")
        oid = r.json()["order_id"]
        print("order:", oid, r.json())
        evs = []
        t0 = time.time()
        with httpx.stream("GET", f"{base}/api/orders/{oid}/events", timeout=httpx.Timeout(600, read=600)) as s:
            for line in s.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    evs.append(ev)
                    if not ev.get("quiet") and ev.get("type") not in ("fields",):
                        tag = ev.get("model") or ev.get("who") or ""
                        print(f"  +{ev.get('t', 0) / 1000:6.2f}s [{ev.get('src')}{'·' + tag if tag else ''}] {ev.get('msg')[:150]}"
                              + (f"  ({ev['ms'] / 1000:.2f}s{', ' + str(ev['credits']) + ' cr' if ev.get('credits') else ''})" if ev.get("ms") else ""))
                    if ev.get("type") == "done":
                        break
        print(f"stream closed after {time.time() - t0:.1f}s, {len(evs)} events")
        stages = sorted({e["stage"] for e in evs if e.get("type") == "stage"})
        if stages != [0, 1, 2, 3, 4, 5]:
            fail(f"stages seen {stages}")
        done = evs[-1]
        if done.get("type") != "done":
            fail("no done event")
        o = c.get(f"/api/orders/{oid}").json()
        flags = [k for k, f in o["fields"].items() if f["status"] in ("needs_review", "unreadable", "missing")]
        print("status:", o["status"], "flags:", len(flags), "verified:", sum(f["status"] == "verified" for f in o["fields"].values()), "of", len(o["fields"]))
        bad = [k for k, f in o["fields"].items() if f["status"] == "verified" and ((f.get("agreement") or 0) < 1.0
               or len(set((f.get("evidence") or {}).get("agreeing_readers") or [])) < 2)]
        if bad:
            fail(f"guess: verified despite disagreement: {bad}")
        if o["status"] != "blocked" or not flags:
            fail("expected a blocked order with flags")
        located = [k for k, f in o["fields"].items() if (f.get("evidence") or {}).get("boxes")]
        meds_located = [k for k in located if k.startswith("med") and k.endswith(".drug_name")]
        print(f"located on page: {len(located)} of {len(o['fields'])} fields; drug lines boxed: {len(meds_located)}")
        if len(located) < 8 or not meds_located:
            fail("too few fields located on the page (grounding boxes missing)")
        models = sorted({e.get("model") for e in evs if e.get("src") == "sie" and e.get("ms")})
        print("SIE models called:", models, "| credits:", o["metrics"].get("credits"), "| replayed:", any(e.get("replayed") for e in evs))
        if a.mode == "live":
            need = {"LightOnOCR", "GLM-OCR", "GLiNER2", "Qwen"}
            if not need <= set(models):
                fail(f"missing live SIE calls: {need - set(models)}")
            if any(e.get("replayed") for e in evs):
                fail("live run used replayed outputs")
        # duplicate upload with same key -> same order
        r2 = c.post("/api/intake", json={"sample": SAMPLE}, headers={"Idempotency-Key": "e2e-1"}).json()
        if r2.get("order_id") != oid or not r2.get("duplicate"):
            fail(f"idempotency broken: {r2}")
        # approve refused while flagged
        if c.post(f"/api/orders/{oid}/approve", json={}).status_code != 409:
            fail("approve should be refused while flags are open")
        # reason required
        if c.post(f"/api/orders/{oid}/fields/{flags[0]}/resolve", json={"value": "x", "reason": " "}).status_code != 422:
            fail("resolve without reason should be 422")
        # technician cannot resolve
        if c.post(f"/api/orders/{oid}/fields/{flags[0]}/resolve", json={"value": "x", "reason": "r"}, headers={"X-Role": "technician"}).status_code != 403:
            fail("technician must not resolve")
        mem_writes = pii = 0
        for k in flags:
            f = o["fields"][k]
            val = (f["candidates"][0]["value"] if f["candidates"] else None) or "confirmed with prescriber"
            rr = c.post(f"/api/orders/{oid}/fields/{k}/resolve", json={"value": val, "reason": "Checked against the source document"},
                        headers={"X-User": "pharmacist.e2e"})
            if rr.status_code != 200:
                fail(f"resolve {k}: {rr.status_code} {rr.text}")
            for ev in rr.json()["events"]:
                mem_writes += ev.get("src") == "memory" and ev.get("op") == "write"
                pii += ev.get("model") == "pii-redact"
        o = c.get(f"/api/orders/{oid}").json()
        if o["status"] != "ready":
            fail(f"after resolving all flags status={o['status']}")
        ap_ = c.post(f"/api/orders/{oid}/approve", json={}, headers={"X-User": "pharmacist.e2e"})
        if ap_.status_code != 200 or ap_.json()["order"]["status"] != "approved":
            fail(f"approve failed {ap_.status_code} {ap_.text[:200]}")
        audit = c.get(f"/api/orders/{oid}/audit").json()["audit"]
        ph = [x for x in audit if x["actor"] == "pharmacist.e2e"]
        fhir = c.get(f"/api/orders/{oid}/export?format=fhir").json()
        mem = c.get("/api/memory").json()
        print(f"resolved {len(flags)} flags · pharmacist audit rows {len(ph)} · memory writes {mem_writes} · pii-redact calls {pii} · "
              f"FHIR MedicationRequests {len(fhir['entry'])} · memory {mem['stats']}")
        if len(ph) != len(flags) + 1:
            fail("expected one pharmacist audit row per resolution plus approval")
        non_phi = [k for k in flags if o["fields"][k]["name"] not in ("patient_name",)]
        if mem_writes < len(non_phi):
            fail(f"expected a memory write per non-patient resolution ({mem_writes} < {len(non_phi)})")
        stored = json.dumps(mem["records"]).lower()
        patient_readings = [str(v).strip().lower() for v in (o["fields"].get("patient_name", {}).get("evidence", {}).get("readings") or {}).values()
                            if str(v).strip() and str(v).strip().lower() not in ("[redacted]", "redacted")]
        patient_readings += [str(c["value"]).strip().lower() for c in o["fields"].get("patient_name", {}).get("candidates", [])
                             if c.get("value") and str(c["value"]).strip().lower() not in ("[redacted]", "redacted")]
        leaked = [v for v in patient_readings if len(v) >= 3 and v in stored]
        if leaked:
            fail(f"patient name reading found in memory: {leaked}")
        print("LIVE_OK" if a.mode == "live" else "E2E_OK")
        if a.keep:
            print("server:", base); srv.wait()
    finally:
        if not a.keep:
            srv.terminate()


if __name__ == "__main__":
    main()
