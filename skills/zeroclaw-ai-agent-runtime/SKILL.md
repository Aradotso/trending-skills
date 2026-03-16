```markdown
---
name: zeroclaw-ai-agent-runtime
description: Expertise in ZeroClaw — the fast, lean, fully autonomous AI agent infrastructure written in Rust with swappable providers, channels, and tools
triggers:
  - set up zeroclaw agent infrastructure
  - deploy zeroclaw on edge hardware
  - configure zeroclaw provider or model
  - zeroclaw tool and channel integration
  - build agentic workflow with zeroclaw
  - zeroclaw cli commands and configuration
  - swap zeroclaw provider or memory backend
  - troubleshoot zeroclaw agent runtime
---

# ZeroClaw AI Agent Runtime

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

ZeroClaw is a zero-overhead, fully autonomous AI agent runtime written in Rust. It runs on <5MB RAM, starts in <10ms, and ships as a single static binary for ARM, x86, and RISC-V. Its trait-driven architecture makes every core system (LLM providers, communication channels, tools, memory backends, tunnels) hot-swappable without recompiling.

---

## Installation

### One-Click Setup (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/zeroclaw-labs/zeroclaw/master/install.sh | bash
```

### Build from Source

```bash
# Prerequisites: Rust stable toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

git clone https://github.com/zeroclaw-labs/zeroclaw.git
cd zeroclaw

# Release build (optimised, small binary)
cargo build --release

# Verify binary size and startup performance
ls -lh target/release/zeroclaw
/usr/bin/time -l target/release/zeroclaw --help   # macOS
/usr/bin/time    target/release/zeroclaw --help   # Linux
```

### Verify Installation

```bash
zeroclaw --version
zeroclaw status
```

---

## Core Concepts

| Concept | Description |
|---|---|
| **Provider** | LLM backend (OpenAI, Anthropic, local, custom OpenAI-compatible) |
| **Channel** | Input/output surface (CLI, HTTP, WebSocket, Slack, etc.) |
| **Tool** | Capability exposed to the agent (shell, browser, file I/O, custom) |
| **Memory** | Conversation + long-term storage backend (in-memory, vector DB, etc.) |
| **Tunnel** | Secure inbound connectivity (ngrok-style) for remote channels |
| **Workspace** | Scoped working directory with allowlisted paths and tool access |

---

## Configuration

ZeroClaw uses a TOML config file. Default location: `~/.zeroclaw/config.toml` (override with `--config`).

### Minimal Config

```toml
# ~/.zeroclaw/config.toml

[provider]
kind = "openai"                         # or "anthropic", "ollama", "custom"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"         # env var name, never hardcode

[channel]
kind = "cli"

[workspace]
root = "/home/user/projects"
allow = ["/home/user/projects"]        # explicit allowlist — nothing outside is accessible
```

### Full Config with All Sections

```toml
[provider]
kind       = "openai"
model      = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
base_url   = "https://api.openai.com/v1"   # override for OpenAI-compatible endpoints
timeout_s  = 60
max_tokens = 4096

[channel]
kind = "http"
host = "127.0.0.1"
port = 8080

[memory]
kind     = "sqlite"
path     = "~/.zeroclaw/memory.db"
max_msgs = 512

[tools]
allow = ["shell", "read_file", "write_file", "http_fetch"]
deny  = []                                  # explicit deny list

[tunnel]
enabled  = false
provider = "cloudflare"                     # or "ngrok"

[workspace]
root  = "/home/user/work"
allow = ["/home/user/work", "/tmp/zeroclaw"]

[logging]
level = "info"      # trace | debug | info | warn | error
format = "pretty"   # pretty | json
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."  # Anthropic
export ZEROCLAW_CONFIG="/etc/zeroclaw/config.toml"
export ZEROCLAW_LOG="debug"
export ZEROCLAW_WORKSPACE="/opt/agent-work"
```

---

## CLI Reference

### Global Flags

```bash
zeroclaw [--config <path>] [--log <level>] <subcommand>
```

### Key Commands

```bash
# Show runtime status and loaded config
zeroclaw status

# Start interactive CLI agent session
zeroclaw run

# Start agent as a background daemon
zeroclaw daemon start
zeroclaw daemon stop
zeroclaw daemon status
zeroclaw daemon logs --follow

# Run a single task and exit (headless / scriptable)
zeroclaw exec "summarise the file ./report.md and save the summary to ./summary.md"

# Provider management
zeroclaw provider list
zeroclaw provider test              # fires a test ping to the configured provider
zeroclaw provider set openai        # switch active provider

# Tool management
zeroclaw tool list
zeroclaw tool enable  http_fetch
zeroclaw tool disable shell

# Memory operations
zeroclaw memory list
zeroclaw memory clear               # wipe current session
zeroclaw memory export ./backup.json
zeroclaw memory import ./backup.json

# Workspace operations
zeroclaw workspace init /path/to/project
zeroclaw workspace allow /path/to/extra-dir
zeroclaw workspace show

# Pairing (for remote channel auth)
zeroclaw pair --channel http        # prints a one-time pairing code
```

---

## Rust Library Usage

ZeroClaw exposes its internals as a Rust library for embedding or extension.

### Add to `Cargo.toml`

```toml
[dependencies]
zeroclaw = { git = "https://github.com/zeroclaw-labs/zeroclaw.git", features = ["full"] }
tokio    = { version = "1", features = ["full"] }
```

### Minimal Embedded Agent

```rust
use zeroclaw::{
    agent::Agent,
    config::{Config, ProviderConfig, ChannelConfig, WorkspaceConfig},
    provider::openai::OpenAiProvider,
    channel::cli::CliChannel,
    tools::ToolRegistry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load from env — never hardcode keys
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");

    let config = Config {
        provider: ProviderConfig::openai("gpt-4o-mini", api_key),
        channel:  ChannelConfig::cli(),
        workspace: WorkspaceConfig::new("/tmp/agent-workspace")
            .with_allow("/tmp/agent-workspace"),
        ..Default::default()
    };

    let provider = OpenAiProvider::from_config(&config.provider)?;
    let channel  = CliChannel::new();
    let tools    = ToolRegistry::from_config(&config.tools);

    let mut agent = Agent::builder()
        .provider(provider)
        .channel(channel)
        .tools(tools)
        .config(config)
        .build()?;

    agent.run().await
}
```

### Custom Provider (Trait Implementation)

```rust
use async_trait::async_trait;
use zeroclaw::provider::{Provider, ProviderRequest, ProviderResponse};

pub struct MyLocalProvider {
    endpoint: String,
}

#[async_trait]
impl Provider for MyLocalProvider {
    async fn complete(
        &self,
        request: ProviderRequest,
    ) -> anyhow::Result<ProviderResponse> {
        // Call your local inference server
        let client = reqwest::Client::new();
        let res = client
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await?
            .json::<ProviderResponse>()
            .await?;
        Ok(res)
    }

    fn model_name(&self) -> &str {
        "local-llama3"
    }
}
```

### Custom Tool (Trait Implementation)

```rust
use async_trait::async_trait;
use zeroclaw::tools::{Tool, ToolCall, ToolResult};

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }

    fn description(&self) -> &str {
        "Echoes the input string back to the agent"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to echo" }
            },
            "required": ["text"]
        })
    }

    async fn invoke(&self, call: ToolCall) -> anyhow::Result<ToolResult> {
        let text = call.args["text"].as_str().unwrap_or("");
        Ok(ToolResult::text(text))
    }
}

// Register and use
let mut registry = ToolRegistry::default();
registry.register(Box::new(EchoTool));
```

### Using OpenAI-Compatible Endpoints (Ollama, local LLMs)

```toml
[provider]
kind        = "openai"
model       = "llama3.2"
api_key_env = "OLLAMA_API_KEY"       # set to any non-empty string if not required
base_url    = "http://localhost:11434/v1"
```

```rust
use zeroclaw::config::ProviderConfig;

let provider_cfg = ProviderConfig {
    kind:        "openai".into(),
    model:       "llama3.2".into(),
    api_key_env: Some("OLLAMA_API_KEY".into()),
    base_url:    Some("http://localhost:11434/v1".into()),
    ..Default::default()
};
```

---

## Common Patterns

### Pattern 1 — Headless Automation Script

```bash
#!/usr/bin/env bash
# Run a zeroclaw task in CI/CD with no interactive prompt
export OPENAI_API_KEY="${OPENAI_API_KEY}"

zeroclaw exec \
  --config ./zeroclaw.ci.toml \
  "Read ./CHANGELOG.md, extract all breaking changes since v1.0.0, and write them to ./BREAKING.md"
```

### Pattern 2 — HTTP Channel + Tunnel for Remote Access

```toml
[channel]
kind = "http"
host = "127.0.0.1"
port = 8080

[tunnel]
enabled  = true
provider = "cloudflare"
```

```bash
zeroclaw daemon start
# ZeroClaw prints a public tunnel URL — share it with your remote client
```

### Pattern 3 — Multi-Agent via Multiple Configs

```bash
# Agent A: code review specialist
zeroclaw daemon start --config ~/.zeroclaw/reviewer.toml --pid-file /tmp/reviewer.pid

# Agent B: documentation writer
zeroclaw daemon start --config ~/.zeroclaw/docs-writer.toml --pid-file /tmp/docs.pid
```

### Pattern 4 — Restricting Tool Access for Untrusted Workloads

```toml
[tools]
allow = ["read_file", "http_fetch"]  # read-only + outbound HTTP only
deny  = ["shell", "write_file"]

[workspace]
root  = "/sandbox"
allow = ["/sandbox"]                 # nothing outside /sandbox is reachable
```

### Pattern 5 — Memory Export / Backup Cron

```bash
#!/usr/bin/env bash
# Save agent memory snapshot daily
DATE=$(date +%Y%m%d)
zeroclaw memory export "/backups/memory-${DATE}.json"
```

---

## Deployment

### Docker (Minimal Image)

```dockerfile
FROM rust:1.77-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/zeroclaw /usr/local/bin/zeroclaw
COPY config.toml /etc/zeroclaw/config.toml
ENV ZEROCLAW_CONFIG=/etc/zeroclaw/config.toml
ENTRYPOINT ["zeroclaw", "daemon", "start"]
```

```bash
docker build -t zeroclaw-agent .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY zeroclaw-agent
```

### Systemd Unit

```ini
# /etc/systemd/system/zeroclaw.service
[Unit]
Description=ZeroClaw AI Agent Daemon
After=network.target

[Service]
Type=simple
User=zeroclaw
EnvironmentFile=/etc/zeroclaw/env
ExecStart=/usr/local/bin/zeroclaw daemon start --config /etc/zeroclaw/config.toml
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/zeroclaw/env  (mode 600, owned by zeroclaw user)
OPENAI_API_KEY=sk-...

sudo systemctl enable --now zeroclaw
sudo journalctl -u zeroclaw -f
```

---

## Troubleshooting

### `zeroclaw status` shows provider unreachable

```bash
# Test provider connectivity explicitly
zeroclaw provider test

# Check env var is exported in the current shell
echo $OPENAI_API_KEY

# Increase log verbosity for detailed error
ZEROCLAW_LOG=debug zeroclaw status
```

### Agent refuses to access a file path

ZeroClaw enforces explicit workspace allowlists. Add the path:

```bash
zeroclaw workspace allow /path/to/needed/dir

# Or in config.toml
[workspace]
allow = ["/existing/path", "/path/to/needed/dir"]
```

### High memory / unexpected resource usage

Verify you are using a **release** build, not a debug build:

```bash
cargo build --release                     # not: cargo build
ls -lh target/release/zeroclaw           # should be ~8–9 MB
/usr/bin/time -l target/release/zeroclaw status   # should be ~4 MB peak
```

### Daemon fails to start (port already in use)

```bash
# Check what is on the configured port
lsof -i :8080

# Change the port in config.toml
[channel]
port = 9090
```

### Memory database locked (SQLite)

```bash
# Stop all running zeroclaw processes first
zeroclaw daemon stop

# Then run memory operations
zeroclaw memory export ./backup.json
zeroclaw memory clear
```

### Build errors on older Rust toolchains

```bash
rustup update stable
rustup override set stable
cargo clean && cargo build --release
```

---

## Security Notes

- **Never commit API keys.** Always reference keys via `api_key_env` pointing to an environment variable name.
- Workspace `allow` lists are enforced at the runtime level — agents cannot read or write outside listed paths.
- Pairing codes for remote channels are one-time-use. Re-pair after any suspected compromise (`zeroclaw pair`).
- Avoid Claude Code OAuth tokens in ZeroClaw per Anthropic's 2026-02-19 ToS update; use direct Anthropic API keys via `ANTHROPIC_API_KEY` instead.
- Only install from the official repo: `github.com/zeroclaw-labs/zeroclaw`. The domains `zeroclaw.org` and `zeroclaw.net` are impersonators.

---

## Key Files & Directories

| Path | Purpose |
|---|---|
| `~/.zeroclaw/config.toml` | Default user config |
| `~/.zeroclaw/memory.db` | Default SQLite memory store |
| `~/.zeroclaw/logs/` | Daemon log output |
| `target/release/zeroclaw` | Release binary after `cargo build --release` |
| `docs/README.md` | Documentation hub |
| `docs/reference/README.md` | Full API/config reference |
| `docs/ops/troubleshooting.md` | Ops troubleshooting guide |
| `docs/hardware/README.md` | Edge/embedded hardware guide |

---

## Resources

- **Repo:** https://github.com/zeroclaw-labs/zeroclaw
- **Website:** https://www.zeroclawlabs.ai/
- **Docs Hub:** https://github.com/zeroclaw-labs/zeroclaw/blob/master/docs/README.md
- **X / Twitter:** https://x.com/zeroclawlabs
- **Reddit:** https://www.reddit.com/r/zeroclawlabs/
- **License:** Apache-2.0
```
