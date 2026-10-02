"""Calibration: Brier score, reliability curve, and Platt recalibration, per question.

Jev is calibrated on its training data, not on our venue. Every decision gets logged with
its predicted probability, and once the horizon passes we record the realized outcome. Live
sizing stays blocked until each question has enough samples and a small calibration error.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_SAMPLES = 200
MAX_ECE = 0.05


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "freq": float(y[m].mean())})
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    n = len(p)
    return float(sum(b["n"] / n * abs(b["mean_p"] - b["freq"]) for b in reliability(p, y, bins)))


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def fit_platt(p: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Fit y ~ sigmoid(a * logit(p) + b) by Newton's method. Identity is (1, 0)."""
    x = _logit(p)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([np.sum((z - y) * x), np.sum(z - y)])
        w = z * (1 - z) + 1e-9
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        h += np.eye(2) * 1e-6
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


@dataclass
class QuestionReport:
    question: str
    n: int
    brier: float
    ece: float
    curve: list[dict]
    platt: tuple[float, float]
    ece_after: float

    @property
    def calibrated(self) -> bool:
        return self.n >= MIN_SAMPLES and min(self.ece, self.ece_after) <= MAX_ECE


def evaluate(question: str, p: np.ndarray, y: np.ndarray) -> QuestionReport:
    p, y = np.asarray(p, float), np.asarray(y, float)
    a, b = fit_platt(p, y) if len(p) >= 30 else (1.0, 0.0)
    after = 1 / (1 + np.exp(-(a * _logit(p) + b)))
    return QuestionReport(question, len(p), brier(p, y), ece(p, y), reliability(p, y),
                          (a, b), ece(after, y) if len(p) else 0.0)


class Recalibrator:
    """Per-question Platt maps. Apply only maps that were fit on enough of OUR fills."""

    def __init__(self, maps: dict[str, tuple[float, float]] | None = None):
        self.maps = maps or {}

    def apply(self, question: str, prob: float) -> float:
        if question not in self.maps:
            return prob
        a, b = self.maps[question]
        x = math.log(min(max(prob, 1e-4), 1 - 1e-4) / (1 - min(max(prob, 1e-4), 1 - 1e-4)))
        return 1 / (1 + math.exp(-(a * x + b)))

    @classmethod
    def from_reports(cls, reports: list[QuestionReport]) -> "Recalibrator":
        # Only recalibrate when the curve actually bends and the fix helps.
        return cls({r.question: r.platt for r in reports if r.question != "p_win"
                    and r.n >= MIN_SAMPLES and r.ece > MAX_ECE and r.ece_after < r.ece})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.maps, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Recalibrator":
        if not path.exists():
            return cls()
        return cls({k: tuple(v) for k, v in json.loads(path.read_text()).items()})
