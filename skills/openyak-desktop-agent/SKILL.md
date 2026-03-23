```markdown
---
name: openyak-desktop-agent
description: OpenYak local-first AI agent desktop app with 100+ models, 16+ built-in tools, and MCP support
triggers:
  - set up OpenYak desktop agent
  - configure OpenYak with my API key
  - use OpenYak for file automation
  - build a skill or tool for OpenYak
  - connect OpenYak to local models
  - OpenYak agent modes and tools
  - integrate MCP server with OpenYak
  - OpenYak backend API development
---

# OpenYak Desktop Agent

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

OpenYak is a 100% local-first, open-source desktop AI assistant that runs entirely on your machine. It supports 100+ AI models via OpenRouter, 16+ built-in tools (file I/O, bash, web fetch, glob/grep), 7 specialized agent modes, MCP server integration, and a secure remote tunnel for mobile access — all without uploading data to the cloud.

---

## Installation

### End-User (Desktop App)

1. Download the installer from [https://open-yak.com/download/](https://open-yak.com/download/) (Windows or macOS).
2. Launch the app and connect a model (free tier: 1M tokens/week, or bring your OpenRouter API key).

### Developer Setup

The project has two parts: `frontend/` (Electron/UI) and `backend/` (Python).

#### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env
python main.py
```

#### Frontend

```bash
cd frontend
npm install
npm run dev          # development mode with hot reload
npm run build        # production build
npm run package      # create distributable installer
```

See [frontend/README.md](frontend/README.md) and [backend/README.md](backend/README.md) for full setup details.

---

## Configuration

### Environment Variables (backend `.env`)

```env
# Model provider — use OpenRouter or your own key
OPENROUTER_API_KEY=$OPENROUTER_API_KEY

# Optional: direct provider keys
OPENAI_API_KEY=$OPENAI_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# Local model endpoint (e.g. Ollama)
LOCAL_MODEL_BASE_URL=http://localhost:11434/v1

# App settings
DATA_DIR=~/.openyak/data
LOG_LEVEL=INFO
```

### Connecting a Model in the UI

1. Open **Settings → Models**.
2. Choose a provider: OpenRouter, OpenAI, Anthropic, or Local (Ollama/LM Studio).
3. Paste your API key or leave blank for free-tier OpenRouter models.
4. Select a default model (e.g. `anthropic/claude-opus-4.6` or `deepseek/deepseek-v3.2`).

---

## Agent Modes

OpenYak ships 7 specialized agents:

| Mode | Purpose |
|------|---------|
| **Build** | Code generation, project scaffolding |
| **Plan** | Multi-step task decomposition |
| **Explore** | File system and data investigation |
| **Write** | Document drafting and editing |
| **Analyze** | Spreadsheet / CSV data analysis |
| **Automate** | Office workflow automation |
| **Chat** | General conversation |

Switch modes from the mode selector in the chat UI or via the API:

```python
import httpx

response = httpx.post("http://localhost:8765/api/chat", json={
    "mode": "analyze",
    "message": "Summarize trends in sales.csv",
    "attachments": ["/home/user/data/sales.csv"]
})
print(response.json()["reply"])
```

---

## Built-in Tools (16+)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write / overwrite a file |
| `edit_file` | Patch specific lines |
| `list_dir` | Directory listing |
| `glob_search` | Pattern-based file search |
| `grep_search` | Text search across files |
| `bash` | Execute shell commands |
| `web_fetch` | HTTP GET a URL |
| `create_dir` | Create directories |
| `delete_file` | Delete files with audit log |
| `move_file` | Move / rename files |
| `copy_file` | Copy files |
| `read_csv` | Parse CSV into structured data |
| `read_docx` | Extract text from Word docs |
| `read_pdf` | Extract text from PDFs |
| `summarize` | Summarize long content |

### Using Tools via the Python Backend API

```python
import httpx

BASE = "http://localhost:8765"

# Ask the agent to use tools automatically
resp = httpx.post(f"{BASE}/api/chat", json={
    "mode": "explore",
    "message": "Find all Python files larger than 10KB in ~/projects and list their sizes",
})
print(resp.json())

# Call a tool directly
resp = httpx.post(f"{BASE}/api/tools/glob_search", json={
    "pattern": "**/*.py",
    "root": "/home/user/projects",
    "min_size_kb": 10
})
for match in resp.json()["matches"]:
    print(match["path"], match["size_kb"])
```

---

## MCP (Model Context Protocol) Integration

OpenYak supports MCP servers, letting you extend it with custom tool providers.

### Registering an MCP Server

```python
# backend/mcp_servers/my_server.py
from openyak.mcp import MCPServer, tool

server = MCPServer(name="my-tools")

@tool(server, description="Fetch weather for a city")
def get_weather(city: str) -> dict:
    import httpx
    r = httpx.get(f"https://wttr.in/{city}?format=j1")
    return r.json()

if __name__ == "__main__":
    server.run()   # starts stdio MCP transport
```

Add it to `~/.openyak/mcp_config.json`:

```json
{
  "servers": [
    {
      "name": "my-tools",
      "command": "python",
      "args": ["/path/to/backend/mcp_servers/my_server.py"],
      "transport": "stdio"
    }
  ]
}
```

Restart OpenYak; the new tool appears in the agent's toolbox automatically.

---

## Building a Custom Skill

Skills are Python modules dropped into `~/.openyak/skills/`:

```python
# ~/.openyak/skills/summarize_inbox.py
"""Skill: summarize emails from a local mbox file."""

from openyak.skills import skill, SkillContext
from mailbox import mbox

@skill(
    name="summarize_inbox",
    description="Parse a local .mbox file and return a structured summary of unread emails",
    triggers=["summarize my inbox", "what emails do I have"]
)
def summarize_inbox(ctx: SkillContext, mbox_path: str) -> str:
    box = mbox(mbox_path)
    summaries = []
    for msg in list(box)[:20]:   # last 20 messages
        subject = msg.get("subject", "(no subject)")
        sender = msg.get("from", "unknown")
        summaries.append(f"- From: {sender} | Subject: {subject}")
    return "\n".join(summaries)
```

Reload skills without restarting:

```bash
curl -X POST http://localhost:8765/api/skills/reload
```

---

## File Automation Patterns

### Batch Rename Files

```python
import httpx

resp = httpx.post("http://localhost:8765/api/chat", json={
    "mode": "automate",
    "message": (
        "Rename all .jpeg files in ~/Downloads/photos to use ISO date format "
        "YYYY-MM-DD_original-name.jpg. Show me an audit log of changes."
    )
})
print(resp.json()["reply"])
# The agent uses move_file + list_dir tools; audit log saved to ~/.openyak/logs/
```

### Analyze a CSV and Export Report

```python
import httpx, pathlib

csv_path = str(pathlib.Path.home() / "data" / "q1_sales.csv")

resp = httpx.post("http://localhost:8765/api/chat", json={
    "mode": "analyze",
    "message": "Find the top 5 products by revenue, identify any anomalies, and write a summary report to ~/reports/q1_summary.md",
    "attachments": [csv_path]
})
print(resp.json()["reply"])
```

### Draft a Document from Notes

```python
import httpx

resp = httpx.post("http://localhost:8765/api/chat", json={
    "mode": "write",
    "message": "Read ~/notes/meeting_notes.txt and draft a formal memo to the team. Save it as ~/docs/team_memo.docx",
})
print(resp.json()["reply"])
```

---

## Remote Access (Secure Tunnel)

Enable one-click tunnel from Settings → Remote Access, or via API:

```python
import httpx

# Start tunnel
resp = httpx.post("http://localhost:8765/api/tunnel/start")
data = resp.json()
print("QR URL:", data["qr_url"])      # scan with phone
print("Tunnel URL:", data["tunnel_url"])

# Stop tunnel
httpx.post("http://localhost:8765/api/tunnel/stop")
```

The tunnel is end-to-end encrypted and bound to your session token. No data passes through external servers.

---

## Backend REST API Reference

All endpoints are on `http://localhost:8765` by default.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send a message to the agent |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/tools/{tool_name}` | Call a built-in tool directly |
| `GET` | `/api/skills` | List installed skills |
| `POST` | `/api/skills/reload` | Hot-reload skills directory |
| `GET` | `/api/history` | Retrieve conversation history |
| `DELETE` | `/api/history` | Clear conversation history |
| `POST` | `/api/tunnel/start` | Start remote access tunnel |
| `POST` | `/api/tunnel/stop` | Stop remote access tunnel |
| `GET` | `/api/health` | Health check |

### Chat Request Schema

```python
{
    "message": str,                  # required
    "mode": str,                     # optional: build|plan|explore|write|analyze|automate|chat
    "model": str,                    # optional: overrides default, e.g. "openai/gpt-4.1"
    "attachments": list[str],        # optional: list of absolute file paths
    "stream": bool,                  # optional: stream SSE response (default False)
    "session_id": str                # optional: continue a previous session
}
```

### Streaming Responses

```python
import httpx

with httpx.stream("POST", "http://localhost:8765/api/chat", json={
    "message": "Analyze ~/data/sales.csv and describe the trend",
    "mode": "analyze",
    "stream": True
}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line[6:], end="", flush=True)
```

---

## Troubleshooting

### Backend won't start

```bash
# Check Python version (requires 3.10+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check port conflict
lsof -i :8765   # macOS/Linux
netstat -ano | findstr 8765   # Windows
```

### Model returns errors

- Verify your API key is set: `echo $OPENROUTER_API_KEY`
- Free-tier rate limit: 1M tokens/week. Check usage at [openrouter.ai/activity](https://openrouter.ai/activity).
- For local models, confirm Ollama is running: `ollama list`

### Tool execution fails

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py

# Check audit log for file operations
cat ~/.openyak/logs/audit.log | tail -50
```

### MCP server not detected

- Confirm `mcp_config.json` is valid JSON: `python -m json.tool ~/.openyak/mcp_config.json`
- Test the server manually: `python /path/to/my_server.py`
- Reload via API: `curl -X POST http://localhost:8765/api/skills/reload`

### Frontend can't connect to backend

```bash
# Confirm backend is healthy
curl http://localhost:8765/api/health

# Check CORS settings in backend/.env
ALLOWED_ORIGINS=http://localhost:3000,app://openyak
```

---

## Key Paths

| Path | Purpose |
|------|---------|
| `~/.openyak/data/` | Local conversation and file data |
| `~/.openyak/skills/` | Custom skill modules |
| `~/.openyak/mcp_config.json` | MCP server registry |
| `~/.openyak/logs/audit.log` | File operation audit trail |
| `~/.openyak/config.json` | App configuration (model, theme, etc.) |
| `backend/` | Python backend source |
| `frontend/` | Electron frontend source |

---

## Resources

- **Homepage:** [https://open-yak.com/](https://open-yak.com/)
- **Download:** [https://open-yak.com/download/](https://open-yak.com/download/)
- **GitHub:** [https://github.com/openyak/desktop](https://github.com/openyak/desktop)
- **License:** AGPL-3.0
```
