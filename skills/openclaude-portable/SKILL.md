```markdown
---
name: openclaude-portable
description: Run Claude Code and other AI coding agents from a USB drive or any folder with zero installation required
triggers:
  - "set up openclaude portable"
  - "run claude code from USB"
  - "portable AI coding agent"
  - "openclaude no install"
  - "configure openclaude provider"
  - "openclaude ollama offline mode"
  - "openclaude dashboard setup"
  - "use openclaude on any PC"
---

# OpenClaude Portable

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

OpenClaude Portable bundles a self-contained Node.js runtime, the OpenClaude AI coding engine, a system-prompt proxy for local models, and a web dashboard — all inside a single folder. Plug a USB drive into any Windows, Linux, or macOS machine and launch a full AI coding agent with no installation.

---

## Installation

### Option 1 — Clone to USB or any folder
```bash
git clone https://github.com/techjarves/OpenClaude-Portable
cd OpenClaude-Portable
```

### Option 2 — Download ZIP
Download and extract from GitHub, then navigate into the folder.

### First Launch (downloads Node.js ~25 MB and engine ~5 MB automatically)

**Windows:**
```bat
.\START.bat
```

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

First-time setup requires internet. All downloads go into `engine/` and `data/` — nothing touches the host machine.

---

## Project Structure

```
OpenClaude-Portable/
├── START.bat               # Windows entry point
├── start.sh                # Linux/macOS entry point
├── RESUME.bat              # Resume a session by ID (Windows)
│
├── data/                   # ALL persistent data lives here
│   ├── ai_settings.env     # Active provider, model, API key
│   ├── openclaude/         # Session history and agent memory
│   ├── ollama/             # Local Ollama binary and models
│   └── proxy.log           # Speed proxy log (silent)
│
├── engine/                 # Bundled Node.js + OpenClaude package
│   ├── node-win-x64/
│   └── node_modules/@gitlawb/openclaude/
│
├── tools/
│   ├── local-proxy.js      # System-prompt trimming proxy
│   ├── Change_Provider.bat / change_provider.sh
│   ├── Open_Dashboard.bat  / open_dashboard.sh
│   └── Setup_Local_Models.bat / setup_local_models.sh
│
└── dashboard/
    ├── server.mjs
    └── index.html
```

---

## Main Menu

Running `START.bat` or `start.sh` presents:

```
1) Launch AI       — Normal Mode      (asks before writing files or running commands)
2) Limitless Mode  — Auto-executes    (fully autonomous, no approval prompts)
3) Open Dashboard  — Web UI at http://localhost:3000
4) Change Provider — Switch model or API key
5) Setup Offline   — Download local Ollama models
```

Auto-selects **Normal Mode** after 10 seconds if no key is pressed.

---

## Configuring an AI Provider

Provider settings are stored in `data/ai_settings.env`. You can edit this file directly or run the interactive switcher:

**Windows:**
```bat
.\tools\Change_Provider.bat
```

**Linux / macOS:**
```bash
./tools/change_provider.sh
```

### `data/ai_settings.env` format

```env
PROVIDER=openrouter
MODEL=openai/gpt-4o
API_KEY=$OPENROUTER_API_KEY
```

Supported `PROVIDER` values and where to get keys:

| Provider | PROVIDER value | Key source |
|---|---|---|
| NVIDIA NIM | `nvidia` | build.nvidia.com |
| DeepSeek | `deepseek` | platform.deepseek.com |
| OpenRouter | `openrouter` | openrouter.ai |
| Google Gemini | `gemini` | aistudio.google.com |
| Anthropic Claude | `anthropic` | console.anthropic.com |
| OpenAI | `openai` | platform.openai.com |
| Ollama (offline) | `ollama` | No key needed |

**Never hard-code API keys.** Reference environment variables or let the interactive setup write `data/ai_settings.env` for you. The file lives only on your drive and is not tracked by git if you add it to `.gitignore`.

---

## Session Management

### Resume an interrupted session (Windows)
```bat
.\RESUME.bat <session-id>
```

Session IDs are stored in `data/openclaude/`. List available sessions:
```bat
dir data\openclaude
```

**Linux / macOS equivalent:**
```bash
ls data/openclaude/
openclaude --resume <session-id>
```

---

## Web Dashboard

The dashboard provides a ChatGPT-style UI with agent mode, tool cards, and thinking visualisation.

**Launch:**
```bat
.\tools\Open_Dashboard.bat        # Windows
./tools/open_dashboard.sh         # Linux/macOS
```

Then open **http://localhost:3000** in any browser.

The dashboard server (`dashboard/server.mjs`) uses the same `data/ai_settings.env` as the CLI — no separate configuration needed.

---

## Offline Mode with Ollama

### Download a local model

**Windows:**
```bat
.\tools\Setup_Local_Models.bat
```

**Linux / macOS:**
```bash
./tools/setup_local_models.sh
```

### Recommended models for CPU inference

| Model | Size | Speed |
|---|---|---|
| `gemma3:1b` | ~800 MB | Fastest |
| `qwen2.5:1.5b` | ~1 GB | Fast |
| `phi3:mini` | ~2.3 GB | Moderate |

Models download into `data/ollama/` — everything stays on the drive.

### Speed proxy for local models

`tools/local-proxy.js` intercepts every Ollama request and trims the OpenClaude system prompt from ~10 000 tokens to ~300 tokens before forwarding. This drops first-token latency from 60–120 s to 5–20 s on CPU-only hardware.

The proxy runs silently in the background and logs to `data/proxy.log`. It binds to port **11435** (not the default Ollama port 11434).

To inspect proxy activity:
```bash
# Linux/macOS
tail -f data/proxy.log

# Windows PowerShell
Get-Content data\proxy.log -Wait
```

If you need to run the proxy manually:
```bash
node tools/local-proxy.js
```

---

## Environment Variables Set at Runtime

OpenClaude Portable redirects all config paths into `data/` so nothing touches the host:

```
XDG_CONFIG_HOME   → <drive>/data
XDG_DATA_HOME     → <drive>/data
CLAUDE_CONFIG_DIR → <drive>/data
```

These are set inside `START.bat` / `start.sh` — you do not need to set them manually.

---

## Real Code Examples

### Custom wrapper script (Linux/macOS) — launch with a specific provider

```bash
#!/usr/bin/env bash
# launch-with-gemini.sh — override provider for a single session

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export XDG_CONFIG_HOME="$SCRIPT_DIR/data"
export XDG_DATA_HOME="$SCRIPT_DIR/data"
export CLAUDE_CONFIG_DIR="$SCRIPT_DIR/data"

# Temporarily override provider
cat > "$SCRIPT_DIR/data/ai_settings.env" <<EOF
PROVIDER=gemini
MODEL=gemini-2.0-flash
API_KEY=$GEMINI_API_KEY
EOF

NODE="$SCRIPT_DIR/engine/node-linux-x64/bin/node"
OPENCLAUDE="$SCRIPT_DIR/engine/node_modules/@gitlawb/openclaude/dist/cli.mjs"

"$NODE" "$OPENCLAUDE" "$@"
```

### Read current provider settings in a script

```bash
#!/usr/bin/env bash
# print-provider.sh

source ./data/ai_settings.env
echo "Provider : $PROVIDER"
echo "Model    : $MODEL"
# Do NOT echo API_KEY
```

### PowerShell — list and resume sessions

```powershell
# list-sessions.ps1
$sessions = Get-ChildItem -Path ".\data\openclaude" -Directory
foreach ($s in $sessions) {
    Write-Host $s.Name
}

# Resume the most recent session
$latest = $sessions | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) {
    Write-Host "Resuming $($latest.Name)"
    .\RESUME.bat $latest.Name
}
```

### Check proxy is running (Node.js)

```javascript
// check-proxy.js — run with bundled Node
import http from 'http';

const req = http.request(
  { hostname: 'localhost', port: 11435, path: '/health', method: 'GET' },
  (res) => {
    console.log(`Proxy status: ${res.statusCode}`);
  }
);
req.on('error', () => console.log('Proxy not running'));
req.end();
```

Run it:
```bash
./engine/node-linux-x64/bin/node check-proxy.js
```

### Dashboard server — add a custom API route

```javascript
// dashboard/server.mjs (extend with your own endpoint)
import express from 'express';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

app.use(express.static(path.join(__dirname)));
app.use(express.json());

// Custom endpoint — return current provider info
app.get('/api/provider', (_req, res) => {
  try {
    const env = readFileSync(
      path.join(__dirname, '../data/ai_settings.env'),
      'utf8'
    );
    const provider = env.match(/^PROVIDER=(.+)$/m)?.[1] ?? 'unknown';
    const model    = env.match(/^MODEL=(.+)$/m)?.[1]    ?? 'unknown';
    res.json({ provider, model });
  } catch {
    res.status(500).json({ error: 'Could not read settings' });
  }
});

app.listen(3000, () => console.log('Dashboard → http://localhost:3000'));
```

---

## Common Patterns

### Pattern 1 — Daily portable workflow

```
1. Plug in USB drive
2. .\START.bat  (or ./start.sh)
3. Choose option 1 (Normal Mode) or 2 (Limitless)
4. Work on your project
5. Unplug — all history in data/, host is clean
```

### Pattern 2 — Switch provider mid-project

```bat
REM Windows
.\tools\Change_Provider.bat
REM Select new provider, enter key, done
.\START.bat
```

### Pattern 3 — Offline day (no internet)

```bat
REM Ensure models already downloaded
.\tools\Setup_Local_Models.bat

REM Set provider to ollama in data\ai_settings.env
REM PROVIDER=ollama
REM MODEL=gemma3:1b

.\START.bat
REM Speed proxy starts automatically, trims prompts for local model
```

### Pattern 4 — Shared USB across OSes

The `data/` folder is platform-agnostic. Work on Windows, switch to Linux:
```bash
./start.sh          # picks up same data/ai_settings.env and session history
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Node.js not found` | Run `START.bat` — it downloads Node automatically |
| `EADDRINUSE: port 11435` | Old proxy still running; `START.bat` kills it on next launch, or kill manually: `taskkill /F /IM node.exe` (Win) / `pkill -f local-proxy` (Unix) |
| `openclaude: dist/cli.mjs not found` | Interrupted install — pull latest and run `START.bat` again to repair |
| `npm error could not determine executable to run` | Pull latest launcher; it now calls the verified bundled binary directly |
| `Claude Code on Windows requires git-bash` | Pull latest `START.bat`; it installs bundled GitPortable automatically |
| `'D_ARGS' is not recognized` | Outdated `START.bat` — pull latest version |
| Ollama very slow | Use `gemma3:1b`; copy `data/ollama/` to local SSD if on USB 2.0 |
| API key rejected | Re-run option 4 to update; verify key at provider's website |
| Port 3000 in use | Dashboard already running — browse to http://localhost:3000 directly |
| `openclaude` not found in PowerShell | Use `.\RESUME.bat <session-id>` instead of calling `openclaude` directly |

---

## Security Notes

- `data/ai_settings.env` contains your API key — keep the drive physically secure.
- Add `data/ai_settings.env` to `.gitignore` before pushing any fork to a public repo.
- Normal Mode (option 1) asks for approval before every file write or shell command — use it on untrusted codebases.
- Limitless Mode (option 2) is fully autonomous — only use it when you trust the codebase and the task.
- All telemetry is disabled; traffic goes only to your chosen AI provider.
```
