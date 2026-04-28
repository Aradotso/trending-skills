---
name: hyperliquid-grid-trading-bot
description: Configurable grid trading bot for Hyperliquid DEX supporting perpetuals and spot, with layered buy/sell orders, risk management, and TypeScript/Python implementations.
triggers:
  - "set up hyperliquid trading bot"
  - "configure grid trading on hyperliquid"
  - "hyperliquid bot grid strategy"
  - "automate trading on hyperliquid"
  - "hyperliquid perpetuals bot"
  - "grid bot crypto trading setup"
  - "hyperliquid dex algorithmic trading"
  - "configure stop loss take profit grid bot"
---

# Hyperliquid Grid Trading Bot

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A configurable grid strategy runner for [Hyperliquid](https://hyperliquid.xyz) DEX. Places layered buy/sell limit orders around a price range with risk controls (stop loss, take profit, drawdown limits, rebalancing). Primary implementation is **TypeScript/Node.js**; a legacy **Python** tree exists for reference and learning examples.

---

## Installation

### Requirements

- Node.js **20.19+** (primary bot)
- A Hyperliquid wallet private key (use a dedicated testnet key to start)
- `git`
- Optional: [uv](https://github.com/astral-sh/uv) for Python examples

### Setup

```bash
git clone https://github.com/PolyPulse-Analytics/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot
npm install
cp .env.example .env
```

Edit `.env` — minimum required fields:

```bash
# Testnet (recommended to start)
HYPERLIQUID_TESTNET=true
HYPERLIQUID_TESTNET_PRIVATE_KEY=$YOUR_TESTNET_PRIVATE_KEY

# Mainnet (real funds — use with caution)
# HYPERLIQUID_TESTNET=false
# HYPERLIQUID_MAINNET_PRIVATE_KEY=$YOUR_MAINNET_PRIVATE_KEY
```

Never commit `.env` or expose private keys.

---

## Key CLI Commands

| Command | Purpose |
|---|---|
| `npm start` | Run bot using first `active: true` config in `bots/` |
| `npm run validate` | Validate selected YAML config (no private key needed) |
| `npm test` | Run automated tests (grid math, etc.) |
| `npx tsx ts/src/runBot.ts path/to/config.yaml` | Run with an explicit config file |
| `npx tsc --noEmit` | TypeScript type check |

### Python (legacy/learning)

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

### Bot YAML (`bots/<name>.yaml`)

Each bot config is a YAML file. Set `active: true` on exactly one file for auto-discovery, or pass an explicit path to the runner.

**Full annotated example:**

```yaml
name: "btc_conservative"
active: true

exchange:
  type: "hyperliquid"
  testnet: true          # Set false for mainnet (also set HYPERLIQUID_TESTNET=false in .env)

account:
  max_allocation_pct: 10.0   # % of account balance bot may use

grid:
  symbol: "BTC"              # Trading pair (e.g. BTC, ETH, SOL for perps)
  levels: 10                 # Number of grid levels (buy + sell orders)
  order_size_usd: 50.0       # Size per grid level in USD
  price_range:
    mode: "auto"             # "auto" or "manual"
    auto:
      range_pct: 5.0         # Grid spans ±5% from current price
    # manual:
    #   lower: 90000
    #   upper: 100000

risk_management:
  stop_loss_enabled: false
  stop_loss_pct: 8.0          # Stop all trading if loss exceeds this %
  take_profit_enabled: false
  take_profit_pct: 20.0       # Close and stop if profit exceeds this %
  max_drawdown_pct: 15.0      # Maximum portfolio drawdown allowed
  max_position_size_pct: 40.0 # Max % of allocation in any single position
  rebalance:
    enabled: true
    price_move_threshold_pct: 12.0  # Rebalance grid if price moves >12% from center

monitoring:
  log_level: "INFO"           # DEBUG | INFO | WARN | ERROR
```

### Environment Variables (`.env`)

```bash
# Network
HYPERLIQUID_TESTNET=true

# Keys
HYPERLIQUID_TESTNET_PRIVATE_KEY=$YOUR_TESTNET_KEY
HYPERLIQUID_MAINNET_PRIVATE_KEY=$YOUR_MAINNET_KEY

# API URLs (optional — defaults in .env.example)
HYPERLIQUID_TESTNET_API_URL=https://api.hyperliquid-testnet.xyz
HYPERLIQUID_MAINNET_API_URL=https://api.hyperliquid.xyz
```

---

## How Grid Trading Works Here

1. Bot fetches current market price for the configured `symbol`.
2. In `auto` mode, it calculates `levels` equally-spaced price points spanning `±range_pct` around current price.
3. Buy limit orders are placed below current price; sell limit orders above.
4. As price moves through levels, filled orders trigger new orders on the opposite side, capturing the spread.
5. Risk rules are evaluated continuously — stop loss, drawdown, position size, and rebalance triggers.

---

## Real Code Examples

### TypeScript — Running the bot programmatically

```typescript
// ts/src/runBot.ts (simplified invocation pattern)
import { runBot } from './bot/engine';
import { loadConfig } from './config/loader';

async function main() {
  const configPath = process.argv[2] ?? './bots/btc_conservative.yaml';
  const config = await loadConfig(configPath);
  await runBot(config);
}

main().catch((err) => {
  console.error('Bot crashed:', err);
  process.exit(1);
});
```

```bash
npx tsx ts/src/runBot.ts bots/btc_conservative.yaml
```

### TypeScript — Custom config override (testnet flag from env)

```typescript
import { loadConfig } from './config/loader';

const config = await loadConfig('./bots/btc_conservative.yaml');

// HYPERLIQUID_TESTNET env var overrides YAML exchange.testnet
// This is handled automatically by the loader — no manual override needed.
console.log('Testnet mode:', config.exchange.testnet);
```

### Python — Place a limit order (learning example pattern)

```python
# learning_examples/04_trading/place_limit_order.py
import os
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

def get_exchange(testnet: bool = True) -> Exchange:
    private_key = os.environ.get(
        "HYPERLIQUID_TESTNET_PRIVATE_KEY" if testnet
        else "HYPERLIQUID_MAINNET_PRIVATE_KEY"
    )
    account = eth_account.Account.from_key(private_key)
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    return Exchange(account, base_url)

def place_limit_order(symbol: str, is_buy: bool, size: float, price: float):
    exchange = get_exchange(testnet=True)
    result = exchange.order(
        symbol,
        is_buy=is_buy,
        sz=size,
        limit_px=price,
        order_type={"limit": {"tif": "Gtc"}},  # Good-till-cancelled
    )
    print("Order result:", result)
    return result

if __name__ == "__main__":
    # BTC buy limit: 0.001 BTC at $90,000
    place_limit_order("BTC", is_buy=True, size=0.001, price=90000.0)
```

### Python — Fetch real-time prices via WebSocket

```python
# learning_examples/01_websockets/realtime_prices.py
import os
from hyperliquid.utils import constants
from hyperliquid.websocket_manager import WebsocketManager

def on_price_update(data):
    print("Price update:", data)

def main():
    ws = WebsocketManager(constants.TESTNET_API_URL)
    ws.subscribe({"type": "allMids"}, on_price_update)
    ws.daemon = True
    ws.start()
    input("Press Enter to stop...\n")

if __name__ == "__main__":
    main()
```

### Python — Validate bot config before starting

```bash
uv run src/run_bot.py --validate
```

```python
# Pattern inside run_bot.py
import argparse
from src.config import load_config, validate_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--config", default="bots/btc_conservative.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    issues = validate_config(config)

    if issues:
        for issue in issues:
            print(f"[CONFIG ERROR] {issue}")
        raise SystemExit(1)

    if args.validate:
        print("Config is valid.")
        return

    # start bot...
```

---

## Common Patterns

### Pattern: Multiple bots with explicit config paths

Only use `active: true` for auto-discovery. For multiple strategies, pass paths explicitly:

```bash
# Terminal 1
npx tsx ts/src/runBot.ts bots/btc_conservative.yaml

# Terminal 2
npx tsx ts/src/runBot.ts bots/eth_aggressive.yaml
```

### Pattern: Adding a new grid config

```bash
cp bots/btc_conservative.yaml bots/eth_grid.yaml
# Edit bots/eth_grid.yaml:
#   name: "eth_grid"
#   active: false          # keep false; pass path explicitly
#   grid.symbol: "ETH"
#   grid.levels: 8
#   grid.price_range.auto.range_pct: 6.0
npm run validate -- bots/eth_grid.yaml
npx tsx ts/src/runBot.ts bots/eth_grid.yaml
```

### Pattern: Safe testnet-first workflow

1. Set `HYPERLIQUID_TESTNET=true` in `.env`
2. Set `exchange.testnet: true` in YAML
3. Run `npm run validate`
4. Run `npm start` and monitor logs
5. Only after validating behavior, switch both flags to `false` for mainnet

### Pattern: Graceful shutdown

The bot handles `Ctrl+C` (SIGINT) and attempts to cancel all open orders before exit. Always check logs after shutdown to confirm order cancellation succeeded:

```bash
npm start 2>&1 | tee bot.log
# On Ctrl+C, review bot.log for "cancelled" confirmations
```

---

## Troubleshooting

### Bot won't start — "No active config found"

Ensure exactly one `bots/*.yaml` has `active: true`, or pass an explicit path:

```bash
npx tsx ts/src/runBot.ts bots/btc_conservative.yaml
```

### Validation errors

```bash
npm run validate
# or for a specific file:
npx tsx ts/src/validateConfig.ts bots/my_config.yaml
```

Common causes: missing required fields (`grid.symbol`, `grid.levels`), invalid `price_range.mode`, `max_allocation_pct` out of `0–100` range.

### Private key / authentication errors

- Confirm the correct env var is set for testnet vs. mainnet (`HYPERLIQUID_TESTNET_PRIVATE_KEY` vs. `HYPERLIQUID_MAINNET_PRIVATE_KEY`).
- Confirm `HYPERLIQUID_TESTNET` in `.env` matches `exchange.testnet` in YAML.
- Fund testnet wallet via [Hyperliquid testnet faucet](https://faucet.chainstack.com/hyperliquid-testnet-faucet).

### Orders not filling / grid not moving

- `range_pct` too narrow (price already outside band) — increase `range_pct` or use `manual` mode with wider bounds.
- Insufficient account balance for `max_allocation_pct` × `levels` × `order_size_usd`.
- Check `monitoring.log_level: "DEBUG"` for detailed order placement logs.

### Python `uv` errors

```bash
uv sync          # Re-sync dependencies
uv python pin 3.11   # Pin Python version if needed
```

### TypeScript compilation errors

```bash
npx tsc --noEmit   # Check for type errors without building
npm test           # Run tests to catch logic regressions
```

---

## Project Structure

```
hyperliquid-trading-bot/
├── bots/                      # YAML strategy configs
│   └── btc_conservative.yaml  # Sample conservative BTC grid
├── ts/
│   └── src/
│       ├── runBot.ts          # TypeScript entrypoint
│       ├── bot/engine.ts      # Core grid engine
│       └── config/loader.ts   # YAML + env config loader
├── src/                       # Legacy Python bot
│   └── run_bot.py             # Python entrypoint
├── learning_examples/         # Educational scripts (Python)
│   ├── 01_websockets/
│   ├── 02_market_data/
│   └── 04_trading/
├── scripts/
│   └── publish-to-polypulse.ps1
├── .env.example               # Environment variable template
├── package.json
└── AGENTS.md / CLAUDE.md      # Contributor conventions
```

---

## Risk Reminder

- Always test on **testnet** with a dedicated key before any mainnet use.
- Grid trading can accumulate large directional positions in trending markets.
- `max_drawdown_pct` and `stop_loss_enabled: true` are your primary safeguards — review them before going live.
- Never share or commit private keys. Use environment variables only.
