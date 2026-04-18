```markdown
---
name: wx-cli-wechat-local-data
description: Query local WeChat data from the command line using a Rust daemon with SQLCipher decryption
triggers:
  - query my WeChat messages
  - search WeChat chat history
  - get WeChat contacts from CLI
  - read local WeChat data
  - wx-cli setup and usage
  - decrypt WeChat database
  - WeChat unread messages command line
  - export WeChat chat history
---

# wx-cli — WeChat Local Data CLI

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

`wx-cli` is a Rust-based CLI tool that queries your local WeChat data (messages, contacts, groups, favorites, stats) by scanning the WeChat process memory for SQLCipher keys and decrypting the local databases via a persistent background daemon.

---

## Architecture

```
wx (CLI) ──Unix socket──▶ wx-daemon (background process)
                              │
                    ┌─────────┴──────────┐
               DBCache               Contact Cache
           (mtime-aware reuse)
```

- Daemon decrypts WeChat's SQLCipher 4 databases on first use, caches them in `~/.wx-cli/cache/`
- Subsequent calls reuse cached DBs if mtime is unchanged — millisecond responses
- All data stays local; no network calls

---

## Installation

### npm (recommended, all platforms)
```bash
npm install -g @jackwener/wx-cli
```

### macOS / Linux (curl)
```bash
curl -fsSL https://raw.githubusercontent.com/jackwener/wx-cli/main/install.sh | bash
```

### Windows (PowerShell as Administrator)
```powershell
irm https://raw.githubusercontent.com/jackwener/wx-cli/main/install.ps1 | iex
```

### Build from source
```bash
git clone git@github.com:jackwener/wx-cli.git && cd wx-cli
cargo build --release
# Binary: target/release/wx  (Windows: wx.exe)
sudo mv target/release/wx /usr/local/bin/
```

---

## Initial Setup (one-time)

WeChat must be running and logged in before initializing.

### macOS
```bash
# 1. Re-sign WeChat so wx-cli can read its memory (redo after WeChat updates)
codesign --force --deep --sign - /Applications/WeChat.app

# If you get "signature in use" error:
codesign --remove-signature "/Applications/WeChat.app/Contents/Frameworks/vlc_plugins/librtp_mpeg4_plugin.dylib"
codesign --force --deep --sign - /Applications/WeChat.app

# 2. Restart WeChat and wait for full login
killall WeChat && open /Applications/WeChat.app

# 3. Initialize (extracts encryption keys from memory)
sudo wx init
```

### Linux
```bash
sudo wx init
```

### Windows (Admin PowerShell)
```powershell
wx init
```

### Verify
```bash
wx sessions   # Should show recent conversations
```

---

## Key Commands

### Sessions & Messages
```bash
# Recent 20 sessions
wx sessions

# Sessions with unread messages
wx unread

# Filter by type: private, group, official_account, folded
wx unread --filter private,group

# New messages since last check (incremental)
wx new-messages

# Chat history (last 50 messages)
wx history "张三"

# Chat history with date range
wx history "AI群" --since 2026-04-01 --until 2026-04-15

# Full-text search across all chats
wx search "关键词"

# Search within a specific chat with date filter
wx search "会议" --in "工作群" --since 2026-01-01
```

### Contacts & Groups
```bash
# All contacts
wx contacts

# Search contacts by name
wx contacts --query "李"

# List group members
wx members "AI交流群"
```

### Favorites & Stats
```bash
# All favorites
wx favorites

# Filter favorites by type: text, image, article, card, video
wx favorites --type image

# Search favorites
wx favorites --query "关键词"

# Chat statistics
wx stats "AI群"

# Stats with date range
wx stats "AI群" --since 2026-01-01
```

### Export
```bash
# Export to Markdown
wx export "张三" --format markdown -o chat.md

# Export group chat as JSON with date range
wx export "AI群" --since 2026-01-01 --format json
```

### Output Formats
```bash
# Default: YAML (token-efficient, human-readable)
wx sessions

# JSON output (for jq piping)
wx sessions --json
wx search "关键词" --json | jq '.[0].content'
wx new-messages --json
wx history "张三" --json | jq '[.[] | select(.sender == "张三")]'
```

### Daemon Management
```bash
wx daemon status
wx daemon stop
wx daemon logs --follow
```

---

## chat_type Values

All session/message output includes a `chat_type` field:

| Value | Meaning |
|-------|---------|
| `private` | Direct/private messages |
| `group` | Group chats |
| `official_account` | Public accounts, subscription accounts, service accounts, system notifications (`mphelper`, `qqsafe`) |
| `folded` | Folded subscriptions / folded group chats aggregation entries |

---

## File Structure

```
~/.wx-cli/
├── config.json       # Configuration
├── all_keys.json     # Database encryption keys
├── daemon.sock       # Unix socket (Linux/macOS)
├── daemon.pid        # Daemon PID file
├── daemon.log        # Daemon log file
└── cache/
    ├── _mtimes.json  # mtime index for cache invalidation
    └── *.db          # Decrypted SQLite databases
```

---

## Common Patterns for AI Agents

### Monitor new messages in a loop
```bash
while true; do
  wx new-messages --json | jq '.[] | "\(.sender): \(.content)"'
  sleep 30
done
```

### Get unread count by type
```bash
wx unread --json | jq 'group_by(.chat_type) | map({type: .[0].chat_type, count: length})'
```

### Export all private chats since a date
```bash
wx contacts --json | jq -r '.[].name' | while read name; do
  wx export "$name" --since 2026-01-01 --format markdown -o "exports/${name}.md"
done
```

### Search and format results
```bash
wx search "项目" --in "工作群" --json | jq '[.[] | {time: .timestamp, sender: .sender, msg: .content}]'
```

### Check daemon health and restart if needed
```bash
if ! wx daemon status | grep -q "running"; then
  sudo wx init
fi
```

---

## How It Works (Technical)

WeChat 4.x encrypts local databases with **SQLCipher 4** (AES-256-CBC + HMAC-SHA512, PBKDF2 256,000 iterations). WCDB caches the derived raw key in process memory as `x'<64hex_key><32hex_salt>'`.

wx-cli extracts keys by:
- **macOS**: Mach VM API (`mach_vm_region` + `mach_vm_read`) — requires `sudo` and ad-hoc code signature
- **Linux**: `/proc/<pid>/mem` scanning — requires `sudo`
- **Windows**: Windows memory API — requires Administrator

The daemon then decrypts databases on demand and caches them with mtime tracking.

---

## Troubleshooting

### "signature in use" on macOS codesign
```bash
codesign --remove-signature "/Applications/WeChat.app/Contents/Frameworks/vlc_plugins/librtp_mpeg4_plugin.dylib"
codesign --force --deep --sign - /Applications/WeChat.app
```

### Daemon not starting / no sessions returned
```bash
# Check daemon logs
wx daemon logs

# Ensure WeChat is running and logged in, then re-init
wx daemon stop
sudo wx init
wx sessions
```

### Stale cache after WeChat update
```bash
# Stop daemon to clear in-memory state
wx daemon stop
# Re-sign WeChat if updated
codesign --force --deep --sign - /Applications/WeChat.app
# Re-initialize
sudo wx init
```

### Permission denied on Linux
```bash
# wx init requires root to read /proc/<pid>/mem
sudo wx init
```

### Cache not updating (messages seem stale)
```bash
# Cache invalidates automatically on DB mtime change
# Force refresh by stopping daemon
wx daemon stop
wx sessions   # Daemon auto-restarts and re-reads
```

---

## AI Agent Integration (skills CLI)

```bash
# Install skill for Claude Code / Cursor / Codex
npx skills add jackwener/wx-cli

# Install globally
npx skills add jackwener/wx-cli -g
```

After installation, the agent reads `SKILL.md` automatically to understand wx-cli usage.
```
