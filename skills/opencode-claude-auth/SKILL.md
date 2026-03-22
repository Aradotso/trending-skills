```markdown
---
name: opencode-claude-auth
description: OpenCode plugin that reuses Claude Code OAuth credentials for Anthropic API access — no separate API key needed
triggers:
  - use Claude Code credentials with OpenCode
  - authenticate OpenCode with Claude subscription
  - opencode-claude-auth plugin setup
  - reuse Claude Pro credentials in OpenCode
  - OpenCode no API key auth
  - set up opencode claude auth plugin
  - Claude Code OAuth token for OpenCode
  - opencode anthropic auth without API key
---

# opencode-claude-auth

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A self-contained Anthropic auth provider for OpenCode that reads your existing Claude Code OAuth credentials — no separate API key or login needed. Works on macOS (Keychain), Linux, and Windows (`~/.claude/.credentials.json`).

## How It Works

1. Registers a custom `auth.loader` with OpenCode that intercepts all Anthropic API calls
2. Reads OAuth tokens from macOS Keychain or `~/.claude/.credentials.json`
3. Caches tokens in memory (30s TTL), refreshes via `claude` CLI when near expiry
4. Syncs credentials to OpenCode's `auth.json` as fallback every 5 minutes
5. Handles tool name translation, SSE stream buffering, retry logic (429/529), and required headers

## Prerequisites

- Claude Code installed and authenticated (`claude` run at least once)
- OpenCode installed

## Installation

### Global npm install

```bash
npm install -g opencode-claude-auth
```

### Register the plugin in `opencode.json`

```json
{
  "plugin": ["opencode-claude-auth"]
}
```

That's it — run OpenCode normally. Auth is automatic.

### AI-assisted install (paste into any agent)

```
Fetch https://raw.githubusercontent.com/griffinmartin/opencode-claude-auth/main/installation.md and follow every step exactly as written.
```

## Configuration

No configuration required by default. All parameters have sensible defaults and can be overridden via environment variables:

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_CLI_VERSION` | Claude CLI version for user-agent/billing headers | `2.1.80` |
| `ANTHROPIC_USER_AGENT` | Full User-Agent string (overrides version) | `claude-cli/{version} (external, cli)` |
| `ANTHROPIC_BETA_FLAGS` | Comma-separated beta feature flags | `claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,prompt-caching-scope-2026-01-05` |

```bash
# Override CLI version if Anthropic updates their API before a new plugin release
export ANTHROPIC_CLI_VERSION=2.2.0

# Override full user agent
export ANTHROPIC_USER_AGENT="claude-cli/2.2.0 (external, cli)"

# Override beta flags
export ANTHROPIC_BETA_FLAGS="claude-code-20250219,oauth-2025-04-20"
```

## Supported Models

```
claude-3-haiku-20240307
claude-haiku-4-5
claude-haiku-4-5-20251001
claude-opus-4-0
claude-opus-4-1
claude-opus-4-1-20250805
claude-opus-4-20250514
claude-opus-4-5
claude-opus-4-5-20251101
claude-opus-4-6
claude-sonnet-4-0
claude-sonnet-4-20250514
claude-sonnet-4-5
claude-sonnet-4-5-20250929
claude-sonnet-4-6
```

Verify models against your account:

```bash
npm run test:models
```

## Credential Sources (checked in order)

1. **macOS Keychain** — entry named `"Claude Code-credentials"` (preferred)
2. **`~/.claude/.credentials.json`** — fallback, works on all platforms

If credentials aren't OAuth-based, the plugin falls through to standard API key auth. If credentials are unavailable, the plugin disables itself gracefully.

## opencode.json Full Example

```json
{
  "model": "claude-sonnet-4-5",
  "plugin": ["opencode-claude-auth"],
  "experimental": {
    "chat": {
      "system": {
        "transform": true
      }
    }
  }
}
```

## What the Plugin Does Internally

```typescript
// The plugin registers an auth loader that intercepts Anthropic requests:
// - Reads OAuth token from Keychain or credentials file
// - Caches in memory with 30s TTL
// - Sets Authorization: Bearer <token>
// - Injects required headers:
//   x-api-key, anthropic-version, anthropic-beta, user-agent, x-stainless-*
// - Translates tool names: OpenCode uses mcp_ prefix, API does not
// - Buffers SSE streams at event boundaries for safe translation
// - Retries on 429/529 with exponential backoff + retry-after header
// - Runs `claude` CLI to refresh tokens expiring within 60s
// - Syncs to auth.json every 5 minutes as fallback
// - On Windows: writes to both %USERPROFILE%\.local\share\opencode\auth.json
//   and %LOCALAPPDATA%\opencode\auth.json
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `"Credentials not found"` | Run `claude` to authenticate Claude Code first |
| `"Keychain is locked"` | Run `security unlock-keychain ~/Library/Keychains/login.keychain-db` |
| `"Token expired and refresh failed"` | Run `claude` manually to re-authenticate |
| Not working on Linux/Windows | Ensure `~/.claude/.credentials.json` exists; run `claude` to create it |
| Keychain access denied | Grant access when macOS prompts |
| Keychain read timed out | Restart Keychain Access (macOS Tahoe bug) |
| `"Credentials are unavailable or expired"` | Run `claude` to refresh credentials |

### Verify credentials file exists (Linux/Windows)

```bash
cat ~/.claude/.credentials.json
```

### Unlock macOS Keychain manually

```bash
security unlock-keychain ~/Library/Keychains/login.keychain-db
```

### Force credential refresh

```bash
claude  # triggers OAuth refresh, plugin picks up new token within 30s
```

## Common Patterns

### Minimal setup for a new machine

```bash
# 1. Authenticate Claude Code
claude

# 2. Install OpenCode (if not done)
# See OpenCode docs for platform-specific install

# 3. Install plugin
npm install -g opencode-claude-auth

# 4. Add to opencode.json
cat > opencode.json << 'EOF'
{
  "plugin": ["opencode-claude-auth"]
}
EOF

# 5. Run OpenCode — auth is automatic
opencode
```

### Using a specific model

```json
{
  "model": "claude-opus-4-5",
  "plugin": ["opencode-claude-auth"]
}
```

### Handling API version changes without waiting for a plugin update

```bash
# If Anthropic bumps their CLI version requirement:
export ANTHROPIC_CLI_VERSION=2.3.0
# Plugin will use this in user-agent and billing headers immediately
```

### CI / non-interactive environments

The plugin requires `~/.claude/.credentials.json` to exist (created by running `claude` interactively). For CI, copy the credentials file from a pre-authenticated environment or use a standard `ANTHROPIC_API_KEY` instead (the plugin falls through to API key auth when no OAuth credentials are present).

## Disclaimer

This plugin uses Claude Code's OAuth credentials. Anthropic's Terms of Service restrict Claude Pro/Max subscription tokens to official clients. This is a community workaround that may break if Anthropic changes their OAuth infrastructure. Use at your own discretion.
```
