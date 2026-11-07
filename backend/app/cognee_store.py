"""cognee mirror for pharmacy memory, in multi-user mode with dataset permissions.

Users: one per agent (extraction-agent, verification-agent, triage-agent) and per human role (pharmacist,
technician). Datasets and permissions:
  pharmacist_corrections  owner pharmacist          write: pharmacist                 read: everyone
  prescriber_patterns     owner verification-agent  write: verification-agent         read: everyone
  agent_notes             owner triage-agent        write: every agent                read: everyone
cognee's LLM and embeddings both run on Superlinked SIE (OpenAI-compatible endpoints), so no frontier model
is involved. Records are added synchronously (cognee.add, no LLM call); the knowledge graph is built by
cognee.cognify in the background (Qwen3.8-27B on SIE: GLM-5.3-Flash lacks json_schema support there), debounced per dataset. Records arrive here already PHI-stripped.
"""
import asyncio
import json
import os
import secrets
import threading
from typing import Any, Dict, Optional

from . import config

USERS = ["extraction-agent", "verification-agent", "triage-agent", "pharmacist", "technician"]
OWNERS = {"pharmacist_corrections": "pharmacist", "prescriber_patterns": "verification-agent", "agent_notes": "triage-agent"}
EXTRA_WRITERS = {"agent_notes": ["extraction-agent", "verification-agent"]}


def configure_env() -> None:
    """cognee reads its settings from the environment at import time."""
    if not config.SIE_API_KEY:
        raise RuntimeError("cognee is configured to use SIE for its LLM and embeddings; SIE_API_KEY is not set")
    root = config.DATA_DIR / "cognee"
    sie_v1 = config.SIE_BASE_URL.rstrip("/") + "/v1"
    defaults = {
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"), "DATA_ROOT_DIRECTORY": str(root / "data"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "LLM_PROVIDER": "custom", "LLM_MODEL": "openai/Qwen/Qwen3.8-27B-FP8", "LLM_ENDPOINT": sie_v1,
        "LLM_API_KEY": config.SIE_API_KEY, "LLM_MAX_COMPLETION_TOKENS": "4096",
        "EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_ENDPOINT": sie_v1, "EMBEDDING_MODEL": "BAAI/bge-m3",
        "EMBEDDING_DIMENSIONS": "1024", "EMBEDDING_API_KEY": config.SIE_API_KEY,
        "COGNEE_SKIP_CONNECTION_TEST": "true", "IMPROVE_AUTO_ENABLED": "false", "TELEMETRY_DISABLED": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def record_text(rec: Dict[str, Any]) -> str:
    """A short natural-language record (no patient data: rec is already PHI-stripped)."""
    if rec.get("dataset") == "pharmacist_corrections":
        readings = "; ".join(f"{k} read {v!r}" for k, v in (rec.get("raw_readings") or {}).items())
        return (f"Pharmacist correction {rec['id']}: field {rec.get('field')}. Model readings: {readings or 'none'}. "
                f"Final value: {rec.get('final_value')!r}. Reason: {rec.get('reason')}. Prescriber: {rec.get('prescriber') or 'unknown'}.")
    if rec.get("dataset") == "agent_notes":
        return f"Agent note {rec['id']} by {rec.get('written_by')}: {rec.get('note')} (field {rec.get('field')})."
    return f"Prescriber pattern {rec['id']}: {json.dumps({k: v for k, v in rec.items() if k not in ('ts',)}, default=str)}"


class CogneeStore:
    def __init__(self):
        configure_env()
        import cognee  # noqa: F401  (import after env is set)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="cognee-loop").start()
        self.users: Dict[str, Any] = {}
        self.datasets: Dict[str, Any] = {}
        self.pending: Dict[str, asyncio.Task] = {}
        self.running: set = set()
        self.dirty: set = set()
        self.cognify_runs = 0
        self.last_error: Optional[str] = None
        self._run(self._setup(), 180)

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _setup(self):
        from cognee.modules.data.methods import create_authorized_dataset
        from cognee.modules.data.methods.get_authorized_dataset_by_name import get_authorized_dataset_by_name
        from cognee.modules.engine.operations.setup import setup
        from cognee.modules.users.methods import create_user, get_user_by_email
        from cognee.modules.users.permissions.methods import give_permission_on_dataset
        await setup()
        for name in USERS:
            email = f"{name}@agents.pillspark.app"   # internal account ids; never mailed
            user = await get_user_by_email(email)
            if user is None:
                user = await create_user(email=email, password=secrets.token_urlsafe(24), is_superuser=False)
            self.users[name] = user
        for ds, owner in OWNERS.items():
            d = await get_authorized_dataset_by_name(ds, self.users[owner], "write")
            if d is None:
                d = await create_authorized_dataset(ds, self.users[owner])
            self.datasets[ds] = d
            for name, user in self.users.items():
                if name != owner:
                    await give_permission_on_dataset(user, d.id, "read")
            for writer in EXTRA_WRITERS.get(ds, []):
                await give_permission_on_dataset(self.users[writer], d.id, "write")

    async def _add(self, dataset: str, actor: str, text: str, node_set):
        import cognee
        await cognee.add(text, dataset_id=self.datasets[dataset].id, user=self.users[actor], node_set=node_set)

    async def _cognify_later(self, dataset: str, delay: float = 20.0):
        """Debounced graph build: runs once writes to the dataset have been quiet for `delay` seconds,
        never two at once; writes that land during a build trigger one more build afterwards."""
        import cognee
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self.pending.pop(dataset, None)
        self.running.add(dataset)
        try:
            await cognee.cognify(datasets=[self.datasets[dataset].id], user=self.users[OWNERS[dataset]])
            self.cognify_runs += 1
        except Exception as e:  # noqa: BLE001 - graph build is best effort; records are already stored
            self.last_error = f"cognify: {type(e).__name__}: {str(e)[:160]}"
        finally:
            self.running.discard(dataset)
            if dataset in self.dirty:
                self.dirty.discard(dataset)
                self._schedule(dataset)

    def _schedule(self, dataset: str):
        if dataset in self.running:
            self.dirty.add(dataset)
            return
        old = self.pending.pop(dataset, None)
        if old:
            old.cancel()
        self.pending[dataset] = self.loop.create_task(self._cognify_later(dataset))

    def remember(self, dataset: str, actor: str, rec: Dict[str, Any], wait_s: float = 6.0) -> str:
        """Add the record to the cognee dataset as `actor` (cognee enforces the write permission).
        Returns "cognee" when the add finished within wait_s, "cognee (queued)" when it is still running."""
        node_set = [str(rec.get("field") or "general")]
        fut = asyncio.run_coroutine_threadsafe(self._add(dataset, actor, record_text(rec), node_set), self.loop)
        self.loop.call_soon_threadsafe(self._schedule, dataset)
        try:
            fut.result(wait_s)
            return "cognee"
        except TimeoutError:
            fut.add_done_callback(lambda f: f.exception() and setattr(self, "last_error", f"add: {f.exception()!r}"[:200]))
            return "cognee (queued)"

    def stats(self) -> Dict[str, Any]:
        return {"users": list(self.users), "datasets": {k: str(v.id) for k, v in self.datasets.items()},
                "cognify_runs": self.cognify_runs, "last_error": self.last_error}
