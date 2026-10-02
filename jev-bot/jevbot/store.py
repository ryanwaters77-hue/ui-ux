"""SQLite log of every signal, order, fill, risk event, and resolved outcome.

The dashboard, daily report, calibration and nightly review all read from here.
Secrets are never written: callers pass only market data and decisions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, price REAL, snapshot TEXT, scores TEXT,
  action TEXT, p_win REAL, confidence REAL, gates TEXT, size_usd REAL, verdict TEXT,
  latency_ms REAL, result_bps REAL);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY, signal_id INTEGER, question TEXT, prob REAL,
  resolve_after TEXT, outcome REAL);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL,
  notional REAL, reason TEXT, pnl_usd REAL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, ts TEXT, level TEXT, kind TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, equity REAL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def log_signal(self, **kw) -> int:
        for k in ("snapshot", "scores", "gates"):
            kw[k] = json.dumps(kw.get(k))
        cols = ",".join(kw)
        return self._exec(f"INSERT INTO signals ({cols}) VALUES ({','.join('?' * len(kw))})",
                          tuple(kw.values()))

    def set_signal_result(self, signal_id: int, result_bps: float) -> None:
        self._exec("UPDATE signals SET result_bps=? WHERE id=?", (result_bps, signal_id))

    def log_prediction(self, signal_id: int, question: str, prob: float, resolve_after: str) -> None:
        self._exec("INSERT INTO predictions (signal_id,question,prob,resolve_after) VALUES (?,?,?,?)",
                   (signal_id, question, prob, resolve_after))

    def resolve_prediction(self, pred_id: int, outcome: float) -> None:
        self._exec("UPDATE predictions SET outcome=? WHERE id=?", (outcome, pred_id))

    def log_fill(self, **kw) -> None:
        kw.setdefault("ts", now())
        self._exec(f"INSERT INTO fills ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})",
                   tuple(kw.values()))

    def event(self, level: str, kind: str, message: str) -> None:
        self._exec("INSERT INTO events (ts,level,kind,message) VALUES (?,?,?,?)",
                   (now(), level, kind, message))

    def mark_equity(self, equity: float) -> None:
        self._exec("INSERT INTO equity VALUES (?,?)", (now(), equity))
