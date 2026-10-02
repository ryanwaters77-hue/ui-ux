# Spec, architecture and plan

This is the document the original brief asked to approve before any code. Code exists now
because the build was requested directly. Review this anyway; anything you'd change here
is cheap to change.

## Decisions taken for the unfilled placeholders

| Placeholder | Default chosen | Where to change |
|---|---|---|
| `[Alpaca paper / exchange]` | Alpaca **paper** | `broker.py` |
| `[asset]` | `BTC/USD`, 15-minute candles (trades 24/7) | `.env` `SYMBOL`, `TIMEFRAME_MIN` |
| `[strategy idea]` | Long-only trend pullback: trade with the trend when flow confirms | `strategy.json` |
| `[$ size]` (manual approval) | $500 | `.env` `MANUAL_APPROVAL_ABOVE_USD` |
| `[Vercel / VPS / Mac Mini]` | VPS (systemd), Mac Mini (launchd). Vercel can't run a 24/7 loop. | `deploy/` |

Alpaca crypto is long-only, so `allow_short` is off and short signals are ignored.

## Architecture: three layers that never overlap

```
            nightly                       every candle (sub-second)
  ┌───────────────────────┐   ┌──────────────────────────────────────────────┐
  │ Opus 5.5 (slow brain) │   │ state_engine ─► Jev (5 questions, 1 call)    │
  │ reads logs, proposes  │   │      │               │ probabilities          │
  │ strategy.json + rules │   │      ▼               ▼                        │
  └──────────┬────────────┘   │  decision.py (gates + explicit weights)       │
             │ proposal       │      ▼                                        │
             ▼                │  sizing.py (≤ ¼ Kelly) ─► risk.py ─► broker   │
  backtest gate (harness) ────┤      ▲                       ▲                │
  accept / reject / rollback  │  config.HardLimits ──────────┘  KILL file     │
                              └──────────────────────────────────────────────┘
```

- **Deterministic code** owns market state, every threshold, every size, every risk veto,
  every order. `config.py`, `risk.py` and `sizing.py` import no model code; `go-live` checks
  this with an AST scan.
- **Jev** gets only the numeric snapshot plus a legend. No headlines and no free text, so
  there's nothing for a prompt injection to ride in on. A Jev error or timeout means no trade.
- **Opus 5.5** can change only the whitelisted strategy params (clamped to bounds), the
  learned-rules list, and the *wording* of Jev questions. Outcomes stay fixed. It can't
  touch hard limits, and its proposals are graded by the backtest, never by itself.

## Jev questions (one call per candle)

| Name | Type | Outcomes |
|---|---|---|
| regime | choice | trending_up, trending_down, ranging, volatile_chop |
| direction | choice | up / down / flat over `horizon_bars` (± `flat_band_bps`) |
| buying_pressure | noul | is buying pressure real |
| setup_quality | score | 0–4 rubric |
| risk_state | choice | normal, elevated, extreme |

Fire only when **every** gate clears: p(regime) ≥ `min_p_regime`, p(direction) ≥
`min_p_direction`, p(pressure) ≥ `min_p_buying_pressure`, setup ≥ `min_setup_quality`,
p(extreme) ≤ `max_p_risk_extreme`, and the weighted `p_win` ≥ `p_cutoff`.

## Sizing

`f* = p − (1−p)/b` with `b = take_profit/stop`. The bot uses `f* × ¼ × (0.5…1.0 by confidence)`,
capped at ¼, at max position size, and at equity (no leverage). Size is zero below the cutoff.
**Live** sizing is zero until `p_win`, `buying_pressure` and `direction:up` are each
calibrated on your own resolved decisions (n ≥ 200, ECE ≤ 0.05).

## Hard risk rules (code only)

max position $, daily loss limit $, max drawdown %, manual approval above $X (the order is
escalated to Telegram and not placed), and the kill switch (`KILL` file). Risk is checked
before every order, and risk-reducing exits are always allowed. Paper trading is the default
until `go-live` passes and you sign it.

## Backtest gate (out of sample, last 30% of data)

≥ 730 days of data, ≥ 30 OOS trades, Sharpe > 1.5, max DD < 15%, hit rate > 55%,
t-stat > 2.0, ≥ 2 regimes covered, and no regime with ≥ 10 trades losing money. Costs:
25 bps fee + 5 bps slippage per side. Entries fill at the next bar's open. If stop and
target are both touched in one bar, the stop counts.

**Limitation:** the backtest scores setups with `ProxyScorer`, a deterministic stand-in with
Jev's output shape, because replaying 2 years of candles through Jev is ~70k calls. It
validates the rules and exits, not Jev's judgment. That's why paper trading has to match
the backtest, and calibration has to be verified on your own fills, before any live money.

## Calibration

Every decision logs per-question probabilities. After the horizon, `outcomes.py` resolves
them with fixed definitions. Brier, ECE and the reliability curve are on the dashboard.
When a curve bends (ECE > 0.05 with n ≥ 200), a Platt map is fitted and applied in code.

## Build plan and test map

| Module | Test file |
|---|---|
| state_engine (no lookahead) | tests/test_state_engine.py |
| sizing, risk, kill switch | tests/test_sizing_and_risk.py |
| decision gates, Platt recalibration | tests/test_decision_calibration.py |
| backtest gate, costs | tests/test_backtest.py |
| engine loop, escalation, kill flatten, outcome resolution | tests/test_engine.py |
| Jev SDK response parsing | tests/test_jev_client.py |
| nightly accept/reject, go-live refusal | tests/test_nightly_golive.py |

Rollback: every accepted nightly revision archives the previous version in `strategies/`.
To restore it, run `python -m jevbot rollback`.
