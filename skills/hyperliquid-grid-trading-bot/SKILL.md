---
name: hyperliquid-grid-trading-bot
description: Configure and run a grid trading bot on Hyperliquid DEX with layered buy/sell orders, risk management, and rebalancing support
triggers:
  - set up hyperliquid trading bot
  - configure grid trading strategy
  - run hyperliquid bot
  - place grid orders on hyperliquid
  - automate trading on hyperliquid dex
  - hyperliquid grid bot configuration
  - stop loss take profit grid trading
  - deploy crypto trading bot hyperliquid
---

# Hyperliquid Grid Trading Bot

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A configurable grid strategy runner for [Hyperliquid](https://hyperliquid.xyz) DEX. Places layered buy and sell orders around a price range and supports risk rules: stop loss, take profit, drawdown limits, position sizing, and rebalancing. Primary implementation is **TypeScript/Node.js**; a **Python** legacy tree is available for reference and learning examples.

---

## Installation

### Prerequisites

- Node.js 20.19 or newer
- A Hyperliquid wallet private key (use a dedicated testnet key first)
- `git`

### Steps

```bash
git clone https://github.com/PolyPulse-Analytics/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot
npm install
```

### Environment Setup

```bash
cp .env.example .env
```

Edit `.env` — minimum required fields:

```bash
# Testnet (recommended to start)
HYPERLIQUID_TESTNET_PRIVATE_KEY=0xyour_private_key_here
HYPERLIQUID_TESTNET=true

# Mainnet (when ready for live trading)
# HYPERLIQUID_MAINNET_PRIVATE_KEY=0xyour_mainnet_key_here
# HYPERLIQUID_TESTNET=false
```

> **Never commit `.env` or expose your private key.**

---

## Key CLI Commands

| Command | Purpose |
|---|---|
| `npm start` | Run bot using first `active: true` config in `bots/` |
| `npm run validate` | Validate selected YAML config (no key required) |
| `npx tsx ts/src/runBot.ts path/to/config.yaml` | Run with explicit config file |
| `npm test` | Run automated tests (grid math, etc.) |
| `npx tsc --noEmit` | TypeScript type check |

### Python (legacy/examples)

```bash
uv sync
uv run src/run_bot.py --validate
uv run src/run_bot.py

# Learning examples
uv run learning_examples/01_websockets/realtime_prices.py
uv run learning_examples/02_market_data/get_all_prices.py
uv run learning_examples/04_trading/place_limit_order.py
```

---

## Configuration

Configs live in `bots/*.yaml`. Only one should have `active: true` when using auto-discovery.

### Full YAML Configuration Reference

```yaml
# bots/my_strategy.yaml
name: "my_grid"
active: true  # Set true for auto-discovery runner; only one active at a time

exchange:
  type: "hyperliquid"
  testnet: true  # Override with HYPERLIQUID_TESTNET env var

account:
  max_allocation_pct: 10.0  # % of account to allocate to this bot

grid:
  symbol: "BTC"
  levels: 10                 # Number of grid levels (buy + sell)
  price_range:
    mode: "auto"             # "auto" or "manual"
    auto:
      range_pct: 5.0         # ± % around current price
    # manual:
    #   lower: 90000
    #   upper: 100000

risk_management:
  stop_loss_enabled: false
  stop_loss_pct: 10.0        # Trigger stop if price drops this % from entry
  take_profit_enabled: false
  take_profit_pct: 20.0
  max_drawdown_pct: 15.0     # Cancel and halt if drawdown exceeds this
  max_position_size_pct: 40.0  # Max % of allocation in a single position
  rebalance:
    price_move_threshold_pct: 12.0  # Re-center grid if price moves this far

monitoring:
  log_level: "INFO"          # DEBUG | INFO | WARN | ERROR
```

### Conservative BTC Example

```yaml
# bots/btc_conservative.yaml
name: "btc_conservative"
active: true

exchange:
  type: "hyperliquid"
  testnet: true

account:
  max_allocation_pct: 5.0

grid:
  symbol: "BTC"
  levels: 8
  price_range:
    mode: "auto"
    auto:
      range_pct: 3.0

risk_management:
  stop_loss_enabled: true
  stop_loss_pct: 8.0
  take_profit_enabled: false
  max_drawdown_pct: 10.0
  max_position_size_pct: 25.0
  rebalance:
    price_move_threshold_pct: 8.0

monitoring:
  log_level: "INFO"
```

### ETH Aggressive Example

```yaml
# bots/eth_aggressive.yaml
name: "eth_aggressive"
active: false  # Set true when btc_conservative is stopped

exchange:
  type: "hyperliquid"
  testnet: false  # LIVE — ensure HYPERLIQUID_MAINNET_PRIVATE_KEY is set

account:
  max_allocation_pct: 20.0

grid:
  symbol: "ETH"
  levels: 15
  price_range:
    mode: "auto"
    auto:
      range_pct: 8.0

risk_management:
  stop_loss_enabled: true
  stop_loss_pct: 12.0
  take_profit_enabled: true
  take_profit_pct: 25.0
  max_drawdown_pct: 20.0
  max_position_size_pct: 50.0
  rebalance:
    price_move_threshold_pct: 15.0

monitoring:
  log_level: "DEBUG"
```

---

## Running the Bot

### Validate first (always)

```bash
npm run validate
```

### Start with auto-discovery

```bash
npm start
```

The runner finds the first `bots/*.yaml` with `active: true`.

### Start with explicit config

```bash
npx tsx ts/src/runBot.ts bots/btc_conservative.yaml
```

### Graceful shutdown

Press **Ctrl+C** — the engine attempts to cancel all open orders before exiting. Check logs to confirm cancellations succeeded.

---

## Python Learning Examples

Use these to understand Hyperliquid API behavior before running the full bot:

```bash
# Real-time price feed via WebSocket
uv run learning_examples/01_websockets/realtime_prices.py

# Fetch all current market prices
uv run learning_examples/02_market_data/get_all_prices.py

# Place a single limit order (use testnet!)
uv run learning_examples/04_trading/place_limit_order.py
```

### Example: Fetching Market Data (Python)

```python
# learning_examples/02_market_data/get_all_prices.py pattern
import os
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Use testnet endpoint
info = Info(constants.TESTNET_API_URL, skip_ws=True)

# Get all mid prices
all_mids = info.all_mids()
for asset, price in all_mids.items():
    print(f"{asset}: {price}")

# Get specific asset metadata
meta = info.meta()
print(meta)
```

### Example: Placing a Limit Order (Python)

```python
# learning_examples/04_trading/place_limit_order.py pattern
import os
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

private_key = os.environ["HYPERLIQUID_TESTNET_PRIVATE_KEY"]
account = Account.from_key(private_key)

exchange = Exchange(account, constants.TESTNET_API_URL)

# Place a limit buy for 0.001 BTC at $90,000
order_result = exchange.order(
    "BTC",           # coin
    True,            # is_buy
    0.001,           # sz (size)
    90000,           # limit_px
    {"limit": {"tif": "Gtc"}},  # order_type (Good Till Cancel)
)
print(order_result)
```

### Example: WebSocket Real-time Prices (Python)

```python
# learning_examples/01_websockets/realtime_prices.py pattern
import os
from hyperliquid.info import Info
from hyperliquid.utils import constants

def on_message(msg):
    if msg.get("channel") == "allMids":
        mids = msg["data"]["mids"]
        btc_price = mids.get("BTC")
        if btc_price:
            print(f"BTC: ${float(btc_price):,.2f}")

info = Info(constants.TESTNET_API_URL)
subscription = {"type": "allMids"}
info.subscribe(subscription, on_message)
```

---

## Common Patterns

### Switching Between Testnet and Mainnet

Control via `.env` — the TypeScript runner respects `HYPERLIQUID_TESTNET`:

```bash
# .env for testnet
HYPERLIQUID_TESTNET=true
HYPERLIQUID_TESTNET_PRIVATE_KEY=0x...

# .env for mainnet
HYPERLIQUID_TESTNET=false
HYPERLIQUID_MAINNET_PRIVATE_KEY=0x...
```

Also verify `exchange.testnet` in your YAML matches your intent. The env var overrides the YAML value when set.

### Running Multiple Strategies

Only one bot should have `active: true` for auto-discovery. To run a specific config:

```bash
# Terminal 1
npx tsx ts/src/runBot.ts bots/btc_conservative.yaml

# Terminal 2
npx tsx ts/src/runBot.ts bots/eth_aggressive.yaml
```

### Grid Math Verification

```bash
# Run the grid math tests before deploying a new config
npm test
```

### Validating a New Config

```bash
npm run validate
# or validate a specific file:
npx tsx ts/src/runBot.ts bots/new_config.yaml --validate-only
```

---

## Troubleshooting

### Bot doesn't start / no active config found

Check that exactly one `bots/*.yaml` has `active: true`:

```bash
grep -l "active: true" bots/*.yaml
```

### Orders not placing — authentication error

- Confirm `HYPERLIQUID_TESTNET_PRIVATE_KEY` (or mainnet equivalent) is set in `.env`
- Confirm `HYPERLIQUID_TESTNET` matches the exchange you're targeting
- Verify the key has funds (testnet faucet: https://faucet.chainstack.com/hyperliquid-testnet-faucet)

### Orders not cancelled on shutdown

If Ctrl+C is interrupted abruptly, manually cancel via Hyperliquid UI or place a cancel-all script:

```python
import os
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

private_key = os.environ["HYPERLIQUID_TESTNET_PRIVATE_KEY"]
account = Account.from_key(private_key)
exchange = Exchange(account, constants.TESTNET_API_URL)

# Cancel all open orders for BTC
result = exchange.cancel_by_cloid("BTC", cloid)  # if you track cloids
# Or use the UI at https://app.hyperliquid.xyz/trade (testnet)
```

### TypeScript type errors

```bash
npx tsc --noEmit
```

Fix any errors before running. Common cause: mismatched Node.js version — ensure Node 20.19+:

```bash
node --version  # should be v20.19.x or higher
```

### Python `uv` not found

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Grid rebalancing triggering too often

Increase `price_move_threshold_pct` in your YAML:

```yaml
risk_management:
  rebalance:
    price_move_threshold_pct: 15.0  # was 5.0 — increase to reduce frequency
```

### Drawdown limit halting bot unexpectedly

Raise `max_drawdown_pct` or reduce `max_allocation_pct` so individual drawdowns matter less:

```yaml
account:
  max_allocation_pct: 5.0   # Reduce allocation
risk_management:
  max_drawdown_pct: 20.0    # More room before halt
```

---

## Security Checklist

- [ ] Use a **dedicated wallet** for the bot — never your main wallet
- [ ] Start on **testnet** with `HYPERLIQUID_TESTNET=true`
- [ ] `.env` is in `.gitignore` — never commit it
- [ ] Rotate any key that has been exposed in logs, chat, or version control
- [ ] Set `max_allocation_pct` conservatively (5–10%) when starting live
- [ ] Enable `stop_loss_enabled: true` and `max_drawdown_pct` for live trading

---

## Project Structure

```
hyperliquid-trading-bot/
├── bots/                        # YAML strategy configs
│   └── btc_conservative.yaml    # Sample conservative BTC grid
├── ts/src/
│   └── runBot.ts                # Main TypeScript bot entrypoint
├── src/
│   └── run_bot.py               # Legacy Python entrypoint
├── learning_examples/
│   ├── 01_websockets/           # WebSocket price feeds
│   ├── 02_market_data/          # Market data queries
│   └── 04_trading/              # Order placement examples
├── scripts/
│   └── publish-to-polypulse.ps1
├── .env.example                 # Environment variable template
├── package.json
└── pyproject.toml
```
