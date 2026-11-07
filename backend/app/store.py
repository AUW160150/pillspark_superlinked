"""SQLite persistence: orders, their streamed events, and an append-only audit trail.

The audit table is insert-only in this codebase (no UPDATE or DELETE statement touches it).
Each call opens its own connection so pipeline threads and the API can write safely.
"""
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from . import config
from .schemas import Order

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, sha256 TEXT, idem_key TEXT UNIQUE, filename TEXT,
            created_at REAL, status TEXT, json TEXT);
        CREATE TABLE IF NOT EXISTS events (
            order_id TEXT, seq INTEGER, json TEXT, PRIMARY KEY (order_id, seq));
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, ts REAL, actor TEXT, action TEXT,
            field TEXT, before TEXT, after TEXT, reason TEXT);
        """)


def save_order(o: Order, idem_key: Optional[str] = None) -> None:
    with _lock, _conn() as c:
        c.execute("""INSERT INTO orders(order_id, sha256, idem_key, filename, created_at, status, json)
                     VALUES (?,?,?,?,?,?,?)
                     ON CONFLICT(order_id) DO UPDATE SET status=excluded.status, json=excluded.json""",
                  (o.order_id, o.sha256, idem_key, o.filename, o.created_at, o.status, o.model_dump_json()))


def get_order(order_id: str) -> Optional[Order]:
    with _conn() as c:
        r = c.execute("SELECT json FROM orders WHERE order_id=?", (order_id,)).fetchone()
    return Order.model_validate_json(r["json"]) if r else None


def find_by_idem(idem_key: str) -> Optional[str]:
    with _conn() as c:
        r = c.execute("SELECT order_id FROM orders WHERE idem_key=?", (idem_key,)).fetchone()
    return r["order_id"] if r else None


def list_orders(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT json FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        o = Order.model_validate_json(r["json"])
        out.append({"order_id": o.order_id, "filename": o.filename, "status": o.status, "created_at": o.created_at,
                    "flags": len(o.review_queue), "model_backend": o.model_backend})
    return out


def add_event(order_id: str, ev: Dict[str, Any]) -> int:
    with _lock, _conn() as c:
        seq = (c.execute("SELECT COALESCE(MAX(seq), 0) FROM events WHERE order_id=?", (order_id,)).fetchone()[0]) + 1
        ev["seq"] = seq
        c.execute("INSERT INTO events(order_id, seq, json) VALUES (?,?,?)", (order_id, seq, json.dumps(ev, default=str)))
    return seq


def events_after(order_id: str, seq: int = 0) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT json FROM events WHERE order_id=? AND seq>? ORDER BY seq", (order_id, seq)).fetchall()
    return [json.loads(r["json"]) for r in rows]


def audit(order_id: str, actor: str, action: str, field: Optional[str] = None, before: Any = None,
          after: Any = None, reason: Optional[str] = None) -> Dict[str, Any]:
    row = {"order_id": order_id, "ts": time.time(), "actor": actor, "action": action, "field": field,
           "before": None if before is None else str(before), "after": None if after is None else str(after), "reason": reason}
    with _lock, _conn() as c:
        cur = c.execute("INSERT INTO audit_events(order_id, ts, actor, action, field, before, after, reason) VALUES (?,?,?,?,?,?,?,?)",
                        (row["order_id"], row["ts"], actor, action, field, row["before"], row["after"], reason))
        row["id"] = cur.lastrowid
    return row


def audit_for(order_id: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM audit_events WHERE order_id=? ORDER BY id", (order_id,)).fetchall()
    return [dict(r) for r in rows]
