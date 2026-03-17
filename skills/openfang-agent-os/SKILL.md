```markdown
---
name: openfang-agent-os
description: Expertise in OpenFang, the open-source Agent Operating System built in Rust for running autonomous AI agents on schedules with Hands, tools, and MCP support.
triggers:
  - set up OpenFang agent operating system
  - create an autonomous agent with OpenFang
  - configure OpenFang Hands
  - build a custom Hand for OpenFang
  - OpenFang CLI commands and usage
  - integrate LLM providers with OpenFang
  - OpenFang HAND.toml configuration
  - deploy OpenFang agents on a schedule
---

# OpenFang Agent OS

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

OpenFang is an open-source **Agent Operating System** written in Rust that runs autonomous AI agents 24/7 without requiring user prompts. Unlike chatbot frameworks, OpenFang agents work *for* you — on schedules, building knowledge graphs, monitoring targets, generating leads, and delivering results to a dashboard. The entire system compiles to a single ~32MB binary.

---

## Installation

### Linux / macOS
```bash
curl -fsSL https://openfang.sh/install | sh
```

### Windows (PowerShell)
```powershell
irm https://openfang.sh/install.ps1 | iex
```

### From source (requires Rust toolchain)
```bash
git clone https://github.com/RightNow-AI/openfang.git
cd openfang
cargo build --release
# Binary at ./target/release/openfang
```

### Initialize and start
```bash
openfang init          # Scaffolds config in ~/.openfang/
openfang start         # Starts daemon + dashboard at http://localhost:4200
```

---

## Core Concepts

| Concept | Description |
|---|---|
| **Hand** | Autonomous capability package (researcher, lead gen, OSINT, etc.) |
| **HAND.toml** | Manifest defining tools, schedule, settings, and metrics for a Hand |
| **SKILL.md** | Domain expertise injected into agent context at runtime |
| **FangHub** | Registry for sharing/installing community Hands |
| **MCP** | Model Context Protocol — tool/resource interface for agents |
| **Channel Adapter** | Integration layer (Telegram, Slack, WhatsApp, webhooks, etc.) |

---

## CLI Reference

### Global Commands
```bash
openfang init                    # Initialize config and workspace
openfang start                   # Start the OpenFang daemon
openfang stop                    # Stop the daemon
openfang status                  # Show daemon and agent status
openfang logs                    # Tail all logs
openfang logs --hand researcher  # Tail logs for a specific Hand
openfang upgrade                 # Upgrade to latest version
```

### Hand Management
```bash
openfang hand list               # List all available Hands
openfang hand activate <name>    # Activate a Hand (starts running immediately)
openfang hand pause <name>       # Pause without losing state
openfang hand resume <name>      # Resume from paused state
openfang hand stop <name>        # Stop and clear state
openfang hand status <name>      # Show Hand run status and metrics
openfang hand inspect <name>     # Show full Hand manifest and config
openfang hand run <name>         # Trigger a one-off manual run
openfang hand logs <name>        # Stream Hand-specific logs
```

### FangHub (Community Registry)
```bash
openfang hub search "crypto monitor"   # Search available Hands
openfang hub install <author>/<hand>   # Install a community Hand
openfang hub publish                   # Publish your Hand to FangHub
openfang hub update <hand>             # Update an installed Hand
```

### Config Management
```bash
openfang config show                   # Print current config
openfang config set llm.provider openai
openfang config set llm.model gpt-4o
openfang config get llm.provider
```

### Provider / Key Management
```bash
openfang provider list                 # List configured LLM providers
openfang provider add openai           # Interactively add a provider
openfang provider set-default openai   # Set default provider
```

---

## Configuration

OpenFang reads from `~/.openfang/config.toml` (created by `openfang init`).

```toml
# ~/.openfang/config.toml

[core]
workspace = "~/.openfang/workspace"
log_level  = "info"          # trace | debug | info | warn | error
dashboard_port = 4200
daemon_port    = 7700

[llm]
provider = "openai"          # openai | anthropic | gemini | groq | ollama | ...
model    = "gpt-4o"
temperature = 0.7
max_tokens  = 8192
timeout_secs = 120

[llm.fallback]               # Automatic fallback chain
providers = ["anthropic", "groq"]

[security]
approval_required = ["browser.purchase", "twitter.post", "email.send"]
sandbox_mode = false         # true = no outbound network in tools
audit_log = true

[scheduler]
timezone = "America/New_York"
max_concurrent_hands = 4

[channels.telegram]
bot_token = "${TELEGRAM_BOT_TOKEN}"   # Always use env var references
chat_id   = "${TELEGRAM_CHAT_ID}"

[channels.slack]
webhook_url = "${SLACK_WEBHOOK_URL}"

[storage]
backend = "sqlite"           # sqlite | postgres
sqlite_path = "~/.openfang/db/openfang.db"
# postgres_url = "${DATABASE_URL}"
```

### Environment Variables
```bash
# LLM Providers
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY}"
export GROQ_API_KEY="${GROQ_API_KEY}"

# Channels
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}"
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}"

# Storage (if using Postgres)
export DATABASE_URL="${DATABASE_URL}"
```

---

## Bundled Hands — Activation & Config

### Researcher Hand
```bash
openfang hand activate researcher
```

Configure in `~/.openfang/hands/researcher/config.toml`:
```toml
[researcher]
schedule     = "0 7 * * *"          # Cron: run at 7 AM daily
topic        = "Rust async runtimes 2026"
depth        = "deep"               # quick | standard | deep
sources      = ["web", "arxiv", "news"]
output_format = "markdown"          # markdown | pdf | json
language     = "en"
deliver_to   = ["telegram", "dashboard"]
cite_sources = true
credibility_check = true            # Uses CRAAP criteria
```

### Lead Hand
```bash
openfang hand activate lead
```

```toml
[lead]
schedule = "0 6 * * 1-5"           # Weekdays at 6 AM
icp = """
  B2B SaaS companies, 10-200 employees,
  Series A or B, hiring engineers,
  US or Canada
"""
score_threshold = 65                # Only deliver leads scored >= 65
deduplicate = true
output_format = ["csv", "json"]
deliver_to = ["telegram", "dashboard"]
max_leads_per_run = 50
```

### Collector Hand (OSINT)
```bash
openfang hand activate collector
```

```toml
[collector]
schedule  = "*/30 * * * *"         # Every 30 minutes
targets   = [
  { type = "company", value = "Acme Corp" },
  { type = "topic",   value = "AI regulation EU" },
]
alert_on_change = true
sentiment_tracking = true
knowledge_graph   = true
deliver_to = ["telegram"]
alert_threshold = "significant"    # minor | significant | critical
```

### Twitter Hand
```bash
openfang hand activate twitter
```

```toml
[twitter]
schedule        = "0 9,13,18 * * *"  # Post 3x daily
account_handle  = "@yourhandle"
topics          = ["Rust", "AI agents", "open source"]
content_formats = ["thread", "single", "poll", "question"]
approval_queue  = true              # Posts queue for your approval
respond_to_mentions = true
track_metrics   = true
```

### Browser Hand
```bash
openfang hand activate browser
```

```toml
[browser]
# NOTE: purchase_approval is always enforced regardless of config
headless        = true
session_persist = true
playwright_port = 9222
allowed_domains = ["*"]            # Restrict with ["example.com"]
```

---

## Building a Custom Hand

### Directory structure
```
my-hand/
├── HAND.toml       # Required: manifest
├── SKILL.md        # Required: domain expertise injected at runtime
├── prompts/
│   └── system.md   # Required: multi-phase operational playbook
└── tools/
    └── custom.toml # Optional: additional tool definitions
```

### HAND.toml
```toml
[hand]
name        = "price-monitor"
version     = "0.1.0"
description = "Monitors product prices and alerts on drops"
author      = "yourname"
license     = "MIT"

[hand.schedule]
default  = "0 */4 * * *"    # Every 4 hours
timezone = "UTC"

[hand.tools]
required = ["web_search", "http_get", "notify"]
optional = ["browser_navigate"]

[hand.settings]
targets = { type = "array", description = "Product URLs to monitor", required = true }
threshold_pct = { type = "float", default = 5.0, description = "Alert when price drops by this %" }
currency = { type = "string", default = "USD" }

[hand.metrics]
# Metrics surfaced in dashboard
prices_tracked  = { type = "counter", label = "Prices Tracked" }
alerts_sent     = { type = "counter", label = "Alerts Sent" }
last_drop_pct   = { type = "gauge",   label = "Last Drop %" }

[hand.deliver]
channels = ["telegram", "dashboard"]

[hand.guardrails]
# These actions require user approval
require_approval = []
```

### prompts/system.md
```markdown
# Price Monitor Agent — Operational Playbook

## Phase 1: Target Validation
For each URL in settings.targets:
- Fetch the page using http_get
- Identify the price element using known patterns (schema.org, og:price, common CSS selectors)
- Record current price and timestamp

## Phase 2: Price Comparison
- Load previous prices from agent memory
- Calculate percentage change for each target
- Flag items where drop >= settings.threshold_pct

## Phase 3: Alert Generation
For each flagged item:
- Format alert with: product name, old price, new price, % drop, URL
- Send via configured delivery channels

## Phase 4: State Persistence
- Write updated prices to agent memory for next run
- Update dashboard metrics

Always be conservative: if price parsing is ambiguous, skip and log rather than alert falsely.
```

### SKILL.md (injected into context)
```markdown
# Price Monitoring Domain Knowledge

## Common Price Selectors
- schema.org: `[itemprop="price"]`, `[itemprop="priceCurrency"]`
- Open Graph: `<meta property="og:price:amount">`
- Amazon: `#priceblock_ourprice`, `.a-price-whole`
- Shopify: `.price__current`, `[data-product-price]`

## Currency Normalization
Always strip symbols and commas before parsing: `$1,299.99` → `1299.99`

## Anti-bot Considerations
Some retailers block automated requests. Use browser_navigate tool as fallback.
```

### Activate your custom Hand
```bash
# From the directory containing HAND.toml
openfang hand install ./my-hand/
openfang hand activate price-monitor
```

---

## Rust SDK / Embedding OpenFang

Add to `Cargo.toml`:
```toml
[dependencies]
openfang-sdk = "0.3"
tokio = { version = "1", features = ["full"] }
```

### Basic agent execution
```rust
use openfang_sdk::{Agent, AgentConfig, LlmProvider, Tool};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build provider from environment
    let provider = LlmProvider::openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()?;

    // Define tools the agent can use
    let tools = vec![
        Tool::web_search(),
        Tool::http_get(),
    ];

    let config = AgentConfig::builder()
        .provider(provider)
        .tools(tools)
        .system_prompt("You are a competitive intelligence researcher.")
        .max_iterations(10)
        .build()?;

    let agent = Agent::new(config);

    let result = agent
        .run("Research the top 5 Rust web frameworks in 2026 and compare their performance benchmarks.")
        .await?;

    println!("{}", result.output);
    Ok(())
}
```

### Running a Hand programmatically
```rust
use openfang_sdk::{HandRegistry, HandRunOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let registry = HandRegistry::load_default().await?;

    let hand = registry.get("researcher")?;

    let options = HandRunOptions::builder()
        .setting("topic", "Rust memory safety vs C++ 2026")
        .setting("depth", "deep")
        .setting("output_format", "markdown")
        .deliver_to(vec!["dashboard"])
        .build();

    let run = hand.run_once(options).await?;

    println!("Run ID: {}", run.id);
    println!("Status: {:?}", run.status);

    // Stream logs
    let mut log_stream = run.logs().await?;
    while let Some(line) = log_stream.next().await {
        println!("[LOG] {}", line?);
    }

    let output = run.wait_for_completion().await?;
    println!("Output:\n{}", output.text);
    Ok(())
}
```

### Custom Tool definition
```rust
use openfang_sdk::tool::{Tool, ToolDef, ToolInput, ToolOutput};
use serde_json::{json, Value};

struct HackerNewsSearchTool;

#[async_trait::async_trait]
impl ToolDef for HackerNewsSearchTool {
    fn name(&self) -> &str { "hn_search" }

    fn description(&self) -> &str {
        "Search Hacker News for stories matching a query"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query":    { "type": "string",  "description": "Search query" },
                "max_hits": { "type": "integer", "description": "Max results", "default": 10 }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, input: ToolInput) -> anyhow::Result<ToolOutput> {
        let query     = input.get_str("query")?;
        let max_hits  = input.get_int("max_hits").unwrap_or(10);

        let url = format!(
            "https://hn.algolia.com/api/v1/search?query={}&hitsPerPage={}",
            urlencoding::encode(query), max_hits
        );

        let resp: Value = reqwest::get(&url).await?.json().await?;

        let results: Vec<String> = resp["hits"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|h| format!(
                "- {} ({}pts) {}", 
                h["title"].as_str().unwrap_or(""),
                h["points"].as_i64().unwrap_or(0),
                h["url"].as_str().unwrap_or("")
            ))
            .collect();

        Ok(ToolOutput::text(results.join("\n")))
    }
}

// Register with an agent
let agent = Agent::new(
    AgentConfig::builder()
        .provider(provider)
        .tool(Tool::custom(HackerNewsSearchTool))
        .build()?
);
```

### MCP Server integration
```rust
use openfang_sdk::mcp::{McpClient, McpTransport};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to an MCP server (e.g. filesystem MCP)
    let mcp = McpClient::connect(
        McpTransport::stdio("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/workspace"])
    ).await?;

    let tools = mcp.list_tools().await?;
    println!("MCP tools: {:?}", tools.iter().map(|t| &t.name).collect::<Vec<_>>());

    // Use MCP tools in an agent
    let agent = Agent::new(
        AgentConfig::builder()
            .provider(provider)
            .mcp_client(mcp)
            .build()?
    );

    agent.run("List all Rust source files in /workspace and summarize what each does.").await?;
    Ok(())
}
```

---

## Scheduling Reference (Cron Syntax)

```toml
# Every 15 minutes
schedule = "*/15 * * * *"

# Daily at 7:30 AM
schedule = "30 7 * * *"

# Weekdays at 9 AM
schedule = "0 9 * * 1-5"

# Every 4 hours
schedule = "0 */4 * * *"

# First of every month at midnight
schedule = "0 0 1 * *"

# Multiple times per day (9 AM, 1 PM, 6 PM)
schedule = "0 9,13,18 * * *"
```

---

## Common Patterns

### Multi-Hand pipeline (research → lead → report)
```bash
# Activate in sequence; Collector feeds context to Lead
openfang hand activate collector
openfang hand activate lead
openfang hand activate researcher

# Wire outputs together via shared workspace
openfang config set hands.lead.use_collector_context true
```

### Approval queue workflow
```bash
# See pending approvals (Twitter posts, browser actions, etc.)
openfang approvals list

# Approve a specific item
openfang approvals approve <id>

# Reject
openfang approvals reject <id>

# Approve all pending for a Hand
openfang approvals approve --hand twitter --all
```

### Export and backup
```bash
openfang export --format json > backup.json
openfang export --hand researcher --format markdown > research-runs.md
```

### Run in CI/headless mode
```bash
# Disable dashboard, run single Hand and exit
OPENFANG_NO_DASHBOARD=1 openfang hand run researcher --once --wait

# JSON output for scripting
openfang hand run researcher --once --wait --output json | jq '.output'
```

---

## Troubleshooting

### Daemon won't start
```bash
openfang status              # Check if already running
openfang stop && openfang start   # Restart cleanly
cat ~/.openfang/logs/daemon.log   # Read full logs
# Check port conflicts
lsof -i :4200 -i :7700
```

### Hand not triggering on schedule
```bash
openfang hand inspect <name>   # Verify cron expression
openfang hand status <name>    # Check last/next run times
# Manually trigger to test
openfang hand run <name>
# Verify timezone config
openfang config get scheduler.timezone
```

### LLM provider errors
```bash
openfang provider list         # Confirm provider is configured
# Test provider connectivity
openfang provider test openai
# Check env vars are exported
echo $OPENAI_API_KEY
# Switch to fallback provider temporarily
openfang config set llm.provider anthropic
```

### Tool execution failures
```bash
# Enable trace logging for a Hand run
OPENFANG_LOG=trace openfang hand run researcher
# Check tool-specific logs
openfang logs --hand researcher --level debug
```

### Database / state corruption
```bash
# Backup first
cp ~/.openfang/db/openfang.db ~/.openfang/db/openfang.db.bak
# Run integrity check
openfang db check
# Repair
openfang db repair
```

### Hand memory / state reset
```bash
# Clear state for a Hand (useful for fresh runs)
openfang hand reset <name>
# Reset only memory, keep config
openfang hand reset <name> --memory-only
```

### Upgrade issues
```bash
# Pin to a specific version
curl -fsSL "https://openfang.sh/install?version=0.3.30" | sh
# Verify binary integrity
openfang --version
openfang self-check
```

---

## Key Files and Directories

```
~/.openfang/
├── config.toml              # Main configuration
├── db/
│   └── openfang.db          # SQLite state store
├── hands/
│   ├── researcher/
│   │   ├── config.toml      # Hand-specific overrides
│   │   └── memory/          # Persistent agent memory
│   └── <hand-name>/
├── logs/
│   ├── daemon.log
│   └── hands/
│       └── <hand-name>.log
├── workspace/               # Shared file workspace between Hands
└── approvals/               # Pending approval queue
```

---

## Resources

- **Documentation**: https://openfang.sh/docs
- **Quick Start**: https://openfang.sh/docs/getting-started
- **FangHub Registry**: https://openfang.sh/hub
- **GitHub**: https://github.com/RightNow-AI/openfang
- **Issues**: https://github.com/RightNow-AI/openfang/issues
- **Twitter/X**: https://x.com/openfangg
```
