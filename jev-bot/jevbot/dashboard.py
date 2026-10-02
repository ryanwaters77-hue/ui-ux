"""Live dashboard: every signal, its probability, confidence, action taken and result.

Run:  uvicorn jevbot.dashboard:app --host 127.0.0.1 --port 8080
Binds to localhost by default. Put it behind an SSH tunnel or Tailscale, not the open internet.
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .config import ROOT, Settings
from .outcomes import calibration_reports
from .store import Store

settings = Settings.from_env()
store = Store(settings.db_path)
app = FastAPI(title="Jev bot")


@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard" / "index.html")


@app.get("/api/state")
def state():
    sig = store.query("SELECT * FROM signals ORDER BY id DESC LIMIT 200")
    for s in sig:
        for k in ("scores", "gates", "snapshot"):
            s[k] = json.loads(s[k]) if s[k] else None
    traded = [s for s in sig if s["result_bps"] is not None]
    return {
        "symbol": settings.symbol,
        "killed": settings.kill_file.exists(),
        "kill_reason": settings.kill_file.read_text().strip() if settings.kill_file.exists() else None,
        "live": settings.live_trading,
        "signals": sig,
        "equity": store.query("SELECT * FROM (SELECT * FROM equity ORDER BY ts DESC LIMIT 500) ORDER BY ts"),
        "events": store.query("SELECT * FROM events ORDER BY id DESC LIMIT 50"),
        "stats": {
            "trades": len(traded),
            "win_rate": sum(s["result_bps"] > 0 for s in traded) / len(traded) if traded else None,
            "avg_latency_ms": (sum(s["latency_ms"] or 0 for s in sig) / len(sig)) if sig else None,
        },
        "calibration": [{"question": r.question, "n": r.n, "brier": r.brier, "ece": r.ece,
                         "calibrated": r.calibrated, "curve": r.curve}
                        for r in calibration_reports(store)],
    }


@app.post("/api/kill")
def kill(x_dashboard_token: str | None = Header(default=None)):
    token = os.getenv("DASHBOARD_TOKEN")
    if token and x_dashboard_token != token:
        raise HTTPException(403, "bad token")
    settings.kill_file.write_text("killed from dashboard\n")
    store.event("critical", "kill", "killed from dashboard")
    return {"killed": True}
