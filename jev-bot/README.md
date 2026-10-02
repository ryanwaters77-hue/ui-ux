# Jev trading bot

Opus 5.5 is the slow brain (nightly research and review). Jev is the fast, calibrated
decision layer (it scores each candle). Deterministic code owns state, thresholds, sizing,
risk and orders. The models advise and the code decides.

**Status:** this is a paper-trading system. Live trading is refused until `python -m jevbot go-live`
says every final-check answer is clean and you sign it.

Design and gates: [docs/SPEC.md](docs/SPEC.md).

## 1. Install

```sh
cd jev-bot
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 41 tests, no network or keys needed
```

## 2. Keys (in `.env` only)

```sh
cp .env.example .env && chmod 600 .env
```

Fill these in **yourself**, in the file, not in chat:

- `TYPESAFE_API_KEY` from console.typesafe.ai (Jev)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: **paper** keys from app.alpaca.markets. Alpaca API
  keys can trade but can't withdraw.
- `ANTHROPIC_API_KEY`, or run `ant auth login` (nightly Opus review)
- `TELEGRAM_BOT_TOKEN` (from @BotFather) and `TELEGRAM_CHAT_ID`
- `DASHBOARD_TOKEN`: any random string; it protects the dashboard kill button

The bot never asks for or enters passwords or 2FA codes. `python -m jevbot check-env` prints
which keys are set, never their values.

## 3. Prove a strategy first

```sh
python -m jevbot fetch-history   # ~800 days of closed bars
python -m jevbot backtest        # writes strategy.md ONLY if the OOS gate passes
```

If it's rejected, the failures are printed. Edit `strategy.json` (or let the nightly loop
propose changes) and rerun. Don't paper trade a strategy that hasn't passed.

## 4. Run on paper

```sh
python -m jevbot run                                         # per-candle loop
uvicorn jevbot.dashboard:app --host 127.0.0.1 --port 8080    # dashboard
```

The dashboard shows every signal with its probability, confidence, gates, action, size,
verdict and result, plus equity, calibration and events. It refreshes every 2 s.

To stop everything immediately, either `touch KILL` or press the dashboard button. The bot
flattens and halts. To resume, delete `KILL` and restart.

## 5. Nightly self-improvement

`python -m jevbot nightly` sends the day's logs to Opus 5.5. Opus proposes param changes,
new rules and Jev question rewording. The harness clamps them, runs the backtest gate, and
accepts them only if they pass and beat the current OOS Sharpe. `python -m jevbot rollback`
undoes the last accepted change.

## 6. Deploy 24/7

**VPS (recommended):** Ubuntu with Python 3.11+.

```sh
sudo useradd -r -m -d /opt/jev-bot jevbot
sudo -u jevbot git clone <this repo> /tmp/r && sudo -u jevbot cp -r /tmp/r/jev-bot/. /opt/jev-bot
cd /opt/jev-bot && sudo -u jevbot python3 -m venv .venv && sudo -u jevbot .venv/bin/pip install -r requirements.txt
sudo -u jevbot cp .env.example .env && sudo -u jevbot nano .env && sudo chmod 600 .env
sudo cp deploy/jevbot*.service deploy/jevbot-nightly.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jevbot jevbot-dashboard jevbot-nightly.timer
journalctl -u jevbot -f
```

`Restart=always` brings the bot back after crashes and reboots. To reach the dashboard,
use `ssh -L 8080:127.0.0.1:8080 you@vps` and open http://localhost:8080.

**Mac Mini:** use `deploy/com.jevbot.run.plist` with launchd (`KeepAlive` restarts it), and
run `sudo pmset -a sleep 0`.

**Vercel:** not suitable. Serverless functions can't hold a 24/7 loop. At most it could host
a read-only dashboard.

**Alerts:** Telegram gets every fill, error, escalation (an order over your approval
threshold), kill-switch event, the daily report and the nightly verdict.

## 7. Before real money

```sh
python -m jevbot go-live
```

It answers each final-check question with data (paper vs. backtest, kill switch fired,
limits in code, calibration on your fills, losing regimes). It ends with
**WHAT COULD BLOW UP THIS ACCOUNT?** and refuses unless every answer is clean. If they are,
you type a confirmation, set `LIVE_TRADING=1`, swap in live Alpaca keys, and restart.
