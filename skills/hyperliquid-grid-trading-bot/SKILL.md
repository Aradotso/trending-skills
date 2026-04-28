---
name: hyperliquid-grid-trading-bot
description: Configure and run a grid trading bot on the Hyperliquid DEX using TypeScript/Node.js with layered buy/sell orders, risk management, and rebalancing.
triggers:
  - set up a hyperliquid trading bot
  - configure grid trading on hyperliquid
  - run a bot on hyperliquid dex
  - hyperliquid grid strategy configuration
  - automate trading on hyperliquid
  - hyperliquid perpetuals trading bot
  - set up crypto grid bot with stop loss
  - deploy hyperliquid spot trading automation
---

# Hyperliquid Grid Trading Bot

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A configurable grid strategy runner for [Hyperliquid](https://hyperliquid.xyz) DEX. Places layered buy and sell orders around a price range and supports stop loss, take profit, drawdown limits, position size limits, and automatic rebalancing. Primary implementation is **TypeScript on Node.js 20.19+**; a legacy Python tree exists for reference and learning examples.

---

## Installation

```bash
git clone https://github.com/PolyPulse-Analytics/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot
npm install
```

**Requirements:**
- Node.js 20.19 or newer
- A Hyperliquid wallet private key (use a dedicated testnet key to start)

---

## Environment Setup

```bash
cp .env.example .env
```

Edit `.env` — minimum required fields:

```env
# Testnet (recommended for development)
HYPERLIQUID_TESTNET_PRIVATE_KEY=0xyour_private_key_here
HYPERLIQUID_TESTNET=true

# Mainnet (real funds — double-check YAML exchange.testnet: false)
# HYPERLIQUID_MAINNET_PRIVATE_KEY=0xyour_private_key_here
# HYPERLIQUID_TESTNET=false
```

**Never commit `.env` or share your private key.**

---

## Key Commands

| Command | Purpose |
|---------|---------|
| `npm start` | Run bot using first `active: true` config in `bots/` |
| `npm run validate` | Validate selected YAML config (no private key needed) |
| `npm test` | Run automated tests (grid math, etc.) |
| `npx tsx ts/src/runBot.ts path/to/config.yaml` | Run with explicit config file |
| `npx tsc --noEmit` | TypeScript type check |

Press **Ctrl+C** to stop — the engine attempts to cancel open orders on shutdown.

---

## Bot Configuration (YAML)

Configs live in `bots/*.yaml`. Only one file should have `active: true` when using auto-discovery.

### Full example: `bots/btc_conservative.yaml`

```yaml
name: "btc_conservative"
active: true

exchange:
  type: "hyperliquid"
  testnet: true          # Set false for mainnet — also check HYPERLIQUID_TESTNET in .env

account:
  max_allocation_pct: 10.0   # Use at most 10% of account balance

grid:
  symbol: "BTC"
  levels: 10                 # Number of buy/sell levels on each side
  price_range:
    mode: "auto"             # "auto" computes range from current price; "manual" uses fixed bounds
    auto:
      range_pct: 5.0         # ±5% from mid price
    # manual:
    #   lower: 90000
    #   upper: 100000

risk_management:
  stop_loss_enabled: false
  stop_loss_pct: 8.0
  take_profit_enabled: false
  take_profit_pct: 20.0
  max_drawdown_pct: 15.0       # Cancel all orders if drawdown exceeds this
  max_position_size_pct: 40.0  # Max position as % of account
  rebalance:
    price_move_threshold_pct: 12.0  # Rebuild grid if price moves outside band by this %

monitoring:
  log_level: "INFO"   # DEBUG | INFO | WARN | ERROR
```

### Minimal config for a new symbol

```yaml
name: "eth_grid"
active: true

exchange:
  type: "hyperliquid"
  testnet: true

account:
  max_allocation_pct: 5.0

grid:
  symbol: "ETH"
  levels: 8
  price_range:
    mode: "auto"
    auto:
      range_pct: 4.0

risk_management:
  max_drawdown_pct: 10.0
  max_position_size_pct: 30.0
  rebalance:
    price_move_threshold_pct: 10.0

monitoring:
  log_level: "INFO"
```

---

## Running with Explicit Config

To run multiple strategies or switch configs without editing `active`:

```bash
# Run a specific config file
npx tsx ts/src/runBot.ts bots/eth_grid.yaml

# Validate before running
npm run validate -- bots/eth_grid.yaml
npx tsx ts/src/runBot.ts bots/eth_grid.yaml
```

---

## Python Legacy Entrypoint

The `src/` tree and `src/run_bot.py` are the older Python implementation. Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync

# Validate config
uv run src/run_bot.py --validate

# Run bot
uv run src/run_bot.py
```

---

## Learning Examples (Python)

Educational scripts under `learning_examples/` — always use testnet keys:

```bash
# Real-time price feed via WebSocket
uv run learning_examples/01_websockets/realtime_prices.py

# Fetch all current prices
uv run learning_examples/02_market_data/get_all_prices.py

# Place a single limit order
uv run learning_examples/04_trading/place_limit_order.py
```

---

## Common Patterns

### Pattern 1: Conservative testnet setup (start here)

```yaml
# bots/testnet_safe.yaml
name: "testnet_safe"
active: true

exchange:
  type: "hyperliquid"
  testnet: true        # MUST be true for testnet

account:
  max_allocation_pct: 5.0   # Small allocation

grid:
  symbol: "BTC"
  levels: 5            # Fewer levels = fewer orders = easier to monitor
  price_range:
    mode: "auto"
    auto:
      range_pct: 3.0   # Tight range for low volatility testing

risk_management:
  stop_loss_enabled: true
  stop_loss_pct: 5.0
  max_drawdown_pct: 8.0
  max_position_size_pct: 20.0
  rebalance:
    price_move_threshold_pct: 8.0

monitoring:
  log_level: "DEBUG"   # Verbose for learning
```

### Pattern 2: Manual price range (range-bound markets)

```yaml
grid:
  symbol: "ETH"
  levels: 12
  price_range:
    mode: "manual"
    manual:
      lower: 3000
      upper: 3600
```

### Pattern 3: Run two bots on different symbols

Create two YAML files. Use explicit paths, not `active: true` on both:

```bash
# Terminal 1
npx tsx ts/src/runBot.ts bots/btc_grid.yaml

# Terminal 2
npx tsx ts/src/runBot.ts bots/eth_grid.yaml
```

---

## TypeScript Integration Examples

### Importing bot types (if extending the bot)

```typescript
// ts/src/runBot.ts is the main entrypoint
// To run programmatically from your own script:

import { runBot } from './ts/src/runBot';

// Pass config path as argument
// npx tsx your_script.ts bots/btc_conservative.yaml
```

### Checking grid math in tests

```bash
npm test
# Runs grid calculation unit tests — useful when tuning levels/range
```

---

## Troubleshooting

### Bot doesn't start / "no active config found"
- Ensure exactly one `bots/*.yaml` has `active: true`
- Or pass an explicit path: `npx tsx ts/src/runBot.ts bots/myconfig.yaml`

### Auth / private key errors
- Confirm `HYPERLIQUID_TESTNET_PRIVATE_KEY` is set in `.env` (not committed, not quoted with extra spaces)
- For mainnet, use the mainnet key variable and set `HYPERLIQUID_TESTNET=false`
- Ensure `exchange.testnet` in YAML matches `HYPERLIQUID_TESTNET` in `.env`

### Orders not cancelling on shutdown
- Check logs after Ctrl+C for cancellation errors
- Manually review open orders on Hyperliquid UI or testnet explorer
- On hard crashes, cancel orders manually via the exchange interface

### `HYPERLIQUID_TESTNET` mismatch
- The TypeScript runner can override YAML `exchange.testnet` with the env var
- To be explicit, set both: `HYPERLIQUID_TESTNET=true` in `.env` **and** `testnet: true` in YAML

### Node.js version errors
```bash
node --version   # Must be 20.19+
nvm install 20
nvm use 20
```

### Python uv not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Validate config without connecting
```bash
npm run validate
# Exits with error code and prints schema violations if config is malformed
```

---

## Risk Reminders

- **Always start on testnet** with a dedicated key and small allocation
- `max_allocation_pct` limits exposure as a percentage of account balance
- Enable `stop_loss_enabled: true` and set `max_drawdown_pct` before going live
- Grid bots lose money in strongly trending markets — tune `rebalance.price_move_threshold_pct` accordingly
- This project is for education and research; you are solely responsible for trading decisions
