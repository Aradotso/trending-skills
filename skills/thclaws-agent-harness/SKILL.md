```markdown
---
name: thclaws-agent-harness
description: Expert skill for using thClaws, the native Rust AI agent workspace platform with multi-provider support, skills, MCP servers, and agent orchestration.
triggers:
  - "set up thClaws agent harness"
  - "configure thClaws with a provider"
  - "install a skill in thClaws"
  - "add an MCP server to thClaws"
  - "create an AGENTS.md for thClaws"
  - "run thClaws in CLI mode"
  - "orchestrate agents with thClaws"
  - "build a thClaws plugin"
---

# thClaws Agent Harness Platform

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

thClaws is a native-Rust AI agent workspace that runs entirely on your local machine. It edits code, automates workflows, searches knowledge bases, and coordinates teams of agents through a single binary. Three interfaces — Desktop GUI, interactive CLI REPL, and non-interactive one-shot mode — share one config, one session store, and one provider layer.

---

## Installation

### Pre-built binary (fastest)

Download from [thclaws.ai/downloads](https://thclaws.ai/downloads) or the GitHub Releases page for macOS (Apple Silicon / Intel), Windows (x86_64 / ARM64), or Linux (x86_64 / ARM64).

### Build from source

Prerequisites: Rust 1.85+, Node.js 20+, pnpm 9+.

```sh
git clone https://github.com/thClaws/thClaws.git
cd thClaws

# 1. Build the React frontend (bundled as a single HTML file)
cd frontend && pnpm install && pnpm build && cd ..

# 2. Build the Rust binary with GUI support
cargo build --release --features gui --bin thclaws

# 3. Verify
./target/release/thclaws --version
```

---

## Running thClaws

```sh
# Desktop GUI (default)
thclaws

# Interactive CLI REPL — no window, ideal for SSH / headless
thclaws --cli

# Non-interactive one-shot — runs one turn and exits
thclaws -p "summarise src/main.rs in three bullet points"

# One-shot with a specific working directory
thclaws -p "list all TODO comments" --cwd /path/to/project

# Shell escape inside the REPL — prefix with !
❯ ! git status
❯ ! ls -la
```

---

## First-Run Setup

On first launch, thClaws prompts you to choose a secrets backend:

- **OS Keychain** (recommended) — macOS Keychain, Windows Credential Manager, Linux Secret Service
- **`.env` file** — for CI or environments without a keychain

API keys are **never written to config JSON files**.

### Configure a provider inside the REPL

```
❯ /provider anthropic
❯ /model claude-sonnet-4-6

# Switch to OpenRouter (300+ models, one API key)
❯ /provider openrouter
❯ /model openrouter/anthropic/claude-sonnet-4-6

# Use a local Ollama model — no cloud, no API key
❯ /provider ollama
❯ /model llama3

# List models available for the active provider
❯ /models
```

### Supported providers (auto-detected by model name prefix)

| Provider | Model prefix example |
|---|---|
| Anthropic | `claude-*` |
| OpenAI | `gpt-*`, `o1-*`, `o3-*` |
| Google Gemini | `gemini-*` |
| Alibaba DashScope | `qwen-*` |
| OpenRouter | `openrouter/*` |
| Ollama (local) | `ollama/*` or bare model name |
| Agentic Press | configured via `/provider agenticpress` |

---

## Configuration Files

Settings are merged in this precedence order (higher index wins):

```
compiled defaults
~/.config/thclaws/settings.json   (user-global)
~/.claude/settings.json           (fallback location)
.thclaws/settings.json            (project-level)
CLI flags
```

### Minimal project settings — `.thclaws/settings.json`

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "permissionMode": "default",
  "thinkingBudget": 8000,
  "allowedTools": ["Read", "Write", "Bash", "Task"],
  "disallowedTools": [],
  "autoApprove": false
}
```

### Permission modes

| Value | Behaviour |
|---|---|
| `"default"` | Approve every mutating tool call interactively |
| `"auto"` | Auto-approve all tool calls (use in CI / trusted scripts) |
| `"restricted"` | Read-only — no writes, no shell |

---

## AGENTS.md — Project Instructions

Drop an `AGENTS.md` (or `CLAUDE.md`) at any directory level. thClaws walks up from `cwd` and injects every match it finds into the system prompt automatically.

```markdown
# AGENTS.md

## Project: Payment Service

### Stack
- Rust 1.85, Axum 0.7, SQLx 0.7, PostgreSQL 15
- Tests: `cargo nextest run`
- Linting: `cargo clippy -- -D warnings`

### Conventions
- All DB queries must use prepared statements via SQLx macros.
- Never commit secrets; use `.env` and the `dotenvy` crate.
- PR titles follow Conventional Commits: `feat:`, `fix:`, `chore:`.

### Commands the agent may run without asking
- `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt`
- `psql $DATABASE_URL -c "..."` for schema inspection

### Off-limits
- Do not modify `migrations/` directly; create new migration files.
- Do not alter `.github/workflows/` without human review.
```

---

## Skills

Skills are reusable expert workflows. The agent picks the right skill automatically when a request matches `whenToUse`, or you invoke one explicitly as `/<skill-name>`.

### Install a skill

```sh
# From a git URL
❯ /skill install https://github.com/anthropics/skills.git

# From a zip archive
❯ /skill install ./my-skill.zip

# List installed skills
❯ /skills
```

### Write a custom skill — `.thclaws/skills/code-review/SKILL.md`

```markdown
---
name: code-review
description: Performs a structured Rust code review checking safety, performance, and idiomatic style.
whenToUse: "review my code, check this PR, audit this file for Rust idioms"
---

# Code Review Skill

## Steps

1. Read every changed file with `Read`.
2. Run `cargo clippy -- -D warnings` and capture output.
3. Run `cargo test` and note failures.
4. Produce a structured report:
   - **Safety** — any `unsafe` blocks, unwrap calls, or panic paths
   - **Performance** — unnecessary allocations, cloning, or blocking calls in async contexts
   - **Idioms** — suggest `?` over `unwrap`, iterators over manual loops, etc.
5. Offer to apply suggested fixes with `Write`.
```

---

## MCP Servers (Tool Plugins)

MCP servers extend the agent's tool set with third-party integrations.

### Add a server inside the REPL

```sh
# stdio transport
❯ /mcp add github https://mcp.github.com

# HTTP Streamable transport
❯ /mcp add mydb http://localhost:3100/mcp

# List active servers
❯ /mcp list
```

### Declare servers in `.mcp.json` (committed to the repo)

```json
{
  "mcpServers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    },
    "postgres": {
      "transport": "http",
      "url": "http://localhost:3200/mcp",
      "oauth": true
    }
  }
}
```

---

## Knowledge Bases (KMS)

Per-project wikis the agent can search and read on demand. No embeddings — plain grep + read.

### Structure

```
.thclaws/kms/
└── architecture/
    ├── index.md          ← one-line entry per page (table of contents)
    └── pages/
        ├── overview.md
        ├── database-schema.md
        └── api-contracts.md
```

### `index.md` format

```markdown
# Architecture Knowledge Base

- overview.md — High-level system diagram and service boundaries
- database-schema.md — PostgreSQL table definitions and relationships
- api-contracts.md — REST and WebSocket API contracts with examples
```

The agent receives this index every turn and uses `KmsRead` / `KmsSearch` tools to pull specific pages on demand.

### Attach a KMS in settings

```json
{
  "kms": [
    { "name": "architecture", "path": ".thclaws/kms/architecture" },
    { "name": "runbooks", "path": "/shared/runbooks" }
  ]
}
```

```sh
# List attached knowledge bases
❯ /kms
```

---

## Agent Orchestration

### Sub-agents via the `Task` tool

The agent can delegate subtasks to isolated child agents (up to 3 levels deep), each with its own tool registry. This happens automatically when the model decides to parallelize — or you can request it explicitly:

```
❯ Write the Axum handler for POST /payments, then in parallel write the
  SQLx query and the unit tests. Use sub-agents for the parallel parts.
```

### Agent Teams (multi-process)

Agent Teams run multiple thClaws processes coordinating through a shared mailbox and task queue, each in its own tmux pane and optional git worktree.

```sh
# Start a team with a lead and two workers
thclaws team start --lead --workers 2 --worktrees

# Each worker gets its own git worktree so branches don't collide
# The lead merges when workers signal completion via the shared queue
```

**Typical team workflow prompt to the lead:**

```
Build a REST API for a task manager.
- Worker 1: implement POST /tasks and GET /tasks with Axum + SQLx
- Worker 2: write integration tests using axum-test and a test database
- You: review both branches, resolve conflicts, produce the final PR description
```

---

## Plugins

Plugins bundle skills + commands + agent definitions + MCP servers under one manifest.

### Install a plugin

```sh
❯ /plugin install https://github.com/my-org/thclaws-deploy-plugin.git
❯ /plugin install ./my-plugin.zip
❯ /plugin list
❯ /plugin uninstall my-plugin
```

### Plugin manifest — `.thclaws-plugin/plugin.json`

```json
{
  "name": "deploy-aws",
  "version": "1.2.0",
  "description": "Deploy thClaws-built apps to AWS via CDK",
  "skills": ["skills/cdk-deploy"],
  "mcpServers": {
    "aws-mcp": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@my-org/aws-mcp-server"],
      "env": { "AWS_PROFILE": "${AWS_PROFILE}" }
    }
  },
  "agentDefinitions": ["agents/deploy-coordinator.json"]
}
```

---

## Memory

thClaws maintains a persistent memory store for facts learned across sessions.

```sh
# View what the agent remembers
❯ /memory list

# Memory is stored as markdown you can read, edit, or commit
cat ~/.config/thclaws/memory.md

# Categories: user | feedback | project | reference
```

Memory entries are injected into the system prompt on each turn alongside `AGENTS.md` content.

---

## Key Slash Commands Reference

| Command | Description |
|---|---|
| `/help` | List all commands |
| `/provider <name>` | Switch provider |
| `/model <name>` | Switch model |
| `/models` | List models for current provider |
| `/skill install <url>` | Install a skill |
| `/skills` | List installed skills |
| `/<skill-name>` | Invoke a skill explicitly |
| `/mcp add <name> <url>` | Add an MCP server |
| `/mcp list` | List MCP servers |
| `/kms` | List knowledge bases |
| `/memory list` | List memory entries |
| `/plugin install <url>` | Install a plugin |
| `/plugin list` | List plugins |
| `! <shell cmd>` | Run a shell command directly (no agent round-trip) |

---

## Real Code Examples

### Rust — reading thClaws settings programmatically

```rust
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ThClawsSettings {
    provider: Option<String>,
    model: Option<String>,
    permission_mode: Option<String>,
    thinking_budget: Option<u32>,
    allowed_tools: Option<Vec<String>>,
    auto_approve: Option<bool>,
}

fn load_project_settings() -> anyhow::Result<ThClawsSettings> {
    let path = PathBuf::from(".thclaws/settings.json");
    let raw = std::fs::read_to_string(&path)?;
    let settings: ThClawsSettings = serde_json::from_str(&raw)?;
    Ok(settings)
}
```

### Non-interactive one-shot from a Rust build script

```rust
// build.rs — generate docs after every build
use std::process::Command;

fn main() {
    let status = Command::new("thclaws")
        .args([
            "-p",
            "Update CHANGELOG.md with a summary of changes in src/ since the last git tag. \
             Be concise. Use conventional-commit style headers.",
            "--cwd", ".",
        ])
        .status()
        .expect("thclaws not found in PATH");

    if !status.success() {
        eprintln!("thclaws exited with {:?}", status.code());
    }
}
```

### Shell script — CI one-shot with auto-approve

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run thClaws in non-interactive mode inside CI
# API key is read from the OS keychain or ANTHROPIC_API_KEY env var
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

thclaws \
  -p "Run cargo clippy -- -D warnings and cargo test. \
      If any check fails, print the error and exit 1." \
  --cwd "$GITHUB_WORKSPACE" \
  --setting permissionMode=auto
```

### `.mcp.json` for a full-stack project

```json
{
  "mcpServers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "postgres": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": { "DATABASE_URL": "${DATABASE_URL}" }
    },
    "brave-search": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" }
    }
  }
}
```

---

## Common Patterns

### Pattern 1 — Iterative feature development

```
❯ Read src/lib.rs and AGENTS.md, then implement the `retry_with_backoff`
  function described in the TODO on line 42. Write tests in tests/retry.rs.
  Run `cargo test retry` and fix any failures before stopping.
```

### Pattern 2 — Parallel branch development with a team

```sh
# Start team
thclaws team start --lead --workers 2 --worktrees

# Prompt to lead agent
❯ We need feature/auth and feature/payments developed in parallel.
  Assign feature/auth to worker-1 and feature/payments to worker-2.
  Each worker should implement, test, and open a draft PR.
  When both signal done, review both PRs and create a merge plan.
```

### Pattern 3 — Knowledge base lookup in a prompt

```
❯ Check the architecture KMS for the database schema, then write a
  SQLx migration that adds an `idempotency_key` column to the
  `payments` table with a unique index.
```

### Pattern 4 — Invoking a skill explicitly

```
❯ /code-review
  Focus on the changes in src/handlers/payment.rs introduced in the
  last two commits.
```

### Pattern 5 — One-shot from a Makefile

```makefile
.PHONY: ai-review
ai-review:
	thclaws -p "Review the diff of the last commit for security issues. \
	            Output findings as GitHub-flavoured markdown." \
	        --setting permissionMode=restricted
```

---

## Troubleshooting

### `thclaws: command not found`

Add the binary to your PATH:

```sh
export PATH="$HOME/.local/bin:$PATH"
# Or symlink it
ln -s /path/to/target/release/thclaws /usr/local/bin/thclaws
```

### API key not found

```sh
# Check which backend is configured
cat ~/.config/thclaws/settings.json | grep secretsBackend

# Set via environment variable as fallback
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."

# Re-run setup to store in keychain
thclaws --setup-secrets
```

### MCP server fails to start

```sh
# Test the server command directly
npx -y @modelcontextprotocol/server-github

# Check thClaws MCP logs
thclaws --cli
❯ /mcp list        # shows status and last error per server
❯ /mcp restart github
```

### GUI won't open on Linux

Ensure a compositor and display server are running. For headless environments, always use `--cli`:

```sh
thclaws --cli
```

### Build fails — missing pnpm or Node

```sh
# Install pnpm
npm install -g pnpm@9

# Verify versions
node --version   # need 20+
pnpm --version   # need 9+
rustc --version  # need 1.85+
```

### Agent loops or hits context limit

- Reduce `thinkingBudget` in settings.
- Break the task into smaller one-shot calls.
- Use sub-agents (`Task` tool) to isolate subtasks.
- Check `AGENTS.md` for instructions that might conflict.

---

## Environment Variables Reference

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (`.env` fallback) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DASHSCOPE_API_KEY` | Alibaba DashScope key |
| `THCLAWS_CONFIG_DIR` | Override `~/.config/thclaws` location |
| `THCLAWS_LOG` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `THCLAWS_NO_KEYCHAIN` | Set to `1` to force `.env` secrets backend |

---

## Project Layout (source)

```
thClaws/
├── src/
│   ├── main.rs          — binary entry point, CLI arg parsing
│   ├── agent/           — core agent loop, tool dispatch, orchestration
│   ├── providers/       — Anthropic, OpenAI, Gemini, Ollama, … adapters
│   ├── tools/           — built-in tools: Read, Write, Bash, Task, KmsRead, …
│   ├── skills/          — skill loading, trigger matching, invocation
│   ├── mcp/             — MCP client (stdio + HTTP), OAuth 2.1 + PKCE
│   ├── memory/          — persistent memory store, classification
│   ├── kms/             — knowledge base index + grep search
│   ├── team/            — multi-process coordination, mailbox, task queue
│   ├── config/          — settings merge, keychain integration
│   └── gui/             — Tauri window, IPC bridge to frontend
├── frontend/            — React + Vite UI (Chat, Terminal, Files, Team tabs)
├── .thclaws/
│   ├── settings.json
│   ├── skills/
│   ├── agents/
│   └── kms/
├── AGENTS.md
├── Cargo.toml
└── README.md
```
```
