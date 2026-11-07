"""Learning-loop check (mock mode, no SIE credits): run the sample, resolve every flag as a pharmacist,
run it again on the same memory, and confirm memory adds "Seen before" hints while set_status() rejects
every memory attempt to verify. Starts its own server on a free port with a throwaway data folder."""
import json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parents[2]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
env = {**os.environ, "MODEL_BACKEND": "mock", "MEMORY_BACKEND": "local", "PILLSPARK_REPLAY_SPEED": "0",
       "PILLSPARK_DATA_DIR": tempfile.mkdtemp(prefix="pillspark-learn-"), "APP_ORIGIN": f"http://127.0.0.1:{port}"}
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "app.main:app", "--app-dir", str(ROOT / "backend"), "--port", str(port), "--log-level", "warning"],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(80):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200: break
        except Exception: time.sleep(0.25)
    base = f"http://127.0.0.1:{port}"
    c = httpx.Client(base_url=base, timeout=120)
    def run():
        oid = c.post("/api/intake", json={"sample": "chemo_chart_redacted.jpg"}).json()["order_id"]
        evs = []
        with httpx.stream("GET", f"{base}/api/orders/{oid}/events", timeout=300) as s:
            for line in s.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:]); evs.append(ev)
                    if ev.get("type") == "done": break
        return oid, evs
    oid, evs = run()
    o = c.get(f"/api/orders/{oid}").json()
    for k, f in o["fields"].items():
        if f["status"] in ("needs_review", "unreadable", "missing"):
            v = (f["candidates"][0]["value"] if f["candidates"] else None) or "confirmed with prescriber"
            c.post(f"/api/orders/{oid}/fields/{k}/resolve", json={"value": v, "reason": "Checked against the source document"})
    oid2, evs2 = run()
    seen = [e["msg"] for e in evs2 if e.get("src") == "memory" and "Seen before" in e.get("msg", "")]
    rej = [e["msg"] for e in evs2 if "REJECTED" in e.get("msg", "")]
    o2 = c.get(f"/api/orders/{oid2}").json()
    verified_by_memory = [k for k, f in o2["fields"].items() if f["status"] == "verified" and f.get("status_source") != "triage_rules"]
    print("run1 memory hits:", sum("Seen before" in e.get("msg","") for e in evs))
    print("run2 seen-before:", len(seen), "| rejected self-verifications:", len(rej), "| verified by non-rules source:", verified_by_memory)
    print(" e.g.", seen[:2]); print(" e.g.", rej[:2])
    print("run2 flags:", len(o2["review_queue"]), "status:", o2["status"])
    ok = bool(seen and rej and not verified_by_memory and o2["status"] == "blocked")
    print("LEARN_OK" if ok else "LEARN_FAIL")
    if not ok: sys.exit(1)
finally:
    srv.terminate()
