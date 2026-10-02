"""Daily report: trades, P&L, win rate, largest loss, Jev latency and cost, calibration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .outcomes import calibration_reports
from .store import Store

JEV_USD_PER_MTOK = 0.042  # TypeSafe published input price; output is free


def daily_report(store: Store, hours: int = 24) -> str:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    sig = store.query("SELECT * FROM signals WHERE ts >= ?", (since,))
    closed = [s for s in sig if s["result_bps"] is not None]
    pnl_bps = [s["result_bps"] for s in closed]
    eq = store.query("SELECT equity FROM equity WHERE ts >= ? ORDER BY ts", (since,))
    pnl_usd = (eq[-1]["equity"] - eq[0]["equity"]) if len(eq) > 1 else 0.0
    lat = [s["latency_ms"] for s in sig if s["latency_ms"]]
    tokens_est = len(sig) * 900  # ~900 input tokens per decision call (snapshot + legend + questions)
    cal = calibration_reports(store)
    lines = [
        f"Daily report ({hours}h)",
        f"Decisions: {len(sig)}  Trades closed: {len(closed)}",
        f"P&L: ${pnl_usd:+.2f}",
        f"Win rate: {sum(x > 0 for x in pnl_bps) / len(pnl_bps):.1%}" if pnl_bps else "Win rate: n/a",
        f"Largest loss: {min(pnl_bps):.1f} bps" if pnl_bps else "Largest loss: n/a",
        f"Jev latency avg: {sum(lat) / len(lat):.0f} ms" if lat else "Jev latency: n/a",
        f"Jev cost/decision: ~${900 * JEV_USD_PER_MTOK / 1e6:.6f} (est. ${tokens_est * JEV_USD_PER_MTOK / 1e6:.4f} today)",
        "Calibration (Brier / ECE / n):",
        *[f"  {r.question}: {r.brier:.4f} / {r.ece:.4f} / {r.n}{'  OK' if r.calibrated else ''}" for r in cal],
    ]
    return "\n".join(lines)
