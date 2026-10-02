"""Resolve logged predictions against what the market actually did, for calibration.

Outcome definitions (fixed, so calibration is measured against the same thing every day):
- direction:up / down / flat : close at horizon vs. close at decision, with the flat band
- buying_pressure            : close at horizon > close at decision
- p_win                      : would a long have hit take-profit before the stop within
                               max_hold_bars (judged on closes; ties go to the stop)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import QuestionReport, evaluate
from .store import Store
from .strategy import StrategyParams


def resolve_pending(store: Store, candles: pd.DataFrame, p: StrategyParams) -> int:
    rows = store.query(
        "SELECT pr.id, pr.question, pr.resolve_after, s.ts AS sig_ts, s.price AS px, s.result_bps "
        "FROM predictions pr JOIN signals s ON s.id = pr.signal_id WHERE pr.outcome IS NULL")
    if not rows:
        return 0
    c = candles.set_index("ts")["close"]
    last = c.index.max()
    done = 0
    for r in rows:
        need = pd.Timestamp(r["resolve_after"])
        if r["question"] == "p_win":
            sig_ts = pd.Timestamp(r["sig_ts"])
            need = max(need, sig_ts + (c.index[1] - c.index[0]) * p.max_hold_bars)
        if need > last:
            continue
        px0 = r["px"]
        if r["question"] == "p_win":
            if r["result_bps"] is not None:
                y = float(r["result_bps"] > 0)
            else:
                path = c[(c.index > pd.Timestamp(r["sig_ts"]))].iloc[: p.max_hold_bars] / px0 - 1
                y = _tp_before_stop(path.to_numpy() * 1e4, p.take_profit_bps, p.stop_bps)
        else:
            px1 = float(c[c.index <= need].iloc[-1])
            move = (px1 / px0 - 1) * 1e4
            q = r["question"]
            if q == "buying_pressure":
                y = float(move > 0)
            elif q == "direction:up":
                y = float(move > p.flat_band_bps)
            elif q == "direction:down":
                y = float(move < -p.flat_band_bps)
            else:
                y = float(abs(move) <= p.flat_band_bps)
        store.resolve_prediction(r["id"], y)
        done += 1
    return done


def _tp_before_stop(path_bps: np.ndarray, tp: float, stop: float) -> float:
    for x in path_bps:
        if x <= -stop:
            return 0.0
        if x >= tp:
            return 1.0
    return float(path_bps[-1] > 0) if len(path_bps) else 0.0


def calibration_reports(store: Store) -> list[QuestionReport]:
    rows = store.query("SELECT question, prob, outcome FROM predictions WHERE outcome IS NOT NULL")
    out = []
    for q in sorted({r["question"] for r in rows}):
        sub = [r for r in rows if r["question"] == q]
        out.append(evaluate(q, np.array([r["prob"] for r in sub]),
                            np.array([r["outcome"] for r in sub])))
    return out


def calibration_verified(reports: list[QuestionReport]) -> bool:
    needed = {"p_win", "buying_pressure", "direction:up"}
    have = {r.question: r for r in reports}
    return all(q in have and have[q].calibrated for q in needed)
