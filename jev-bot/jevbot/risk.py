"""Hard risk rules. Checked before every order. No model output reaches this module.

The kill switch is a file on disk (KILL). Creating it from anywhere (the dashboard, `touch KILL`,
the risk manager itself) flattens all positions and halts trading. Only a human deleting the
file and restarting brings the bot back.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .config import HardLimits


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    needs_approval: bool = False


class RiskManager:
    def __init__(self, limits: HardLimits, kill_file: Path, start_equity: float):
        self.limits = limits
        self.kill_file = kill_file
        self.peak_equity = start_equity
        self.day: date | None = None
        self.day_start_equity = start_equity

    # -- kill switch ---------------------------------------------------------
    @property
    def killed(self) -> bool:
        return self.kill_file.exists()

    def kill(self, reason: str) -> None:
        self.kill_file.write_text(reason + "\n")

    # -- equity tracking -----------------------------------------------------
    def update_equity(self, equity: float, today: date | None = None) -> str | None:
        """Call on every mark. Returns a kill reason if a hard limit tripped."""
        today = today or datetime.now(timezone.utc).date()
        if self.day is None:
            self.day = today
        elif today != self.day:
            self.day, self.day_start_equity = today, equity
        self.peak_equity = max(self.peak_equity, equity)
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        day_loss = self.day_start_equity - equity
        reason = None
        if day_loss >= self.limits.daily_loss_limit_usd:
            reason = f"daily loss ${day_loss:.2f} >= limit ${self.limits.daily_loss_limit_usd}"
        elif dd_pct >= self.limits.max_drawdown_pct:
            reason = f"drawdown {dd_pct:.2f}% >= limit {self.limits.max_drawdown_pct}%"
        if reason and not self.killed:
            self.kill(reason)
        return reason

    # -- per-order check -----------------------------------------------------
    def check_order(self, *, notional_usd: float, current_position_usd: float,
                    reduces_risk: bool) -> Verdict:
        if reduces_risk:
            return Verdict(True, "risk-reducing order always allowed")
        if self.killed:
            return Verdict(False, "kill switch is active")
        if notional_usd <= 0:
            return Verdict(False, "non-positive size")
        if abs(current_position_usd) + notional_usd > self.limits.max_position_usd + 1e-9:
            return Verdict(False, f"would exceed max position ${self.limits.max_position_usd}")
        if notional_usd > self.limits.manual_approval_above_usd:
            return Verdict(False, "needs manual approval", needs_approval=True)
        return Verdict(True, "ok")
