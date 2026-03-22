```markdown
---
name: codex-console-openai-registration
description: Skill for using codex-console, an integrated console for automated OpenAI account registration, login, token retrieval, batch processing, and data export with Web UI management.
triggers:
  - "set up codex-console"
  - "automate OpenAI account registration"
  - "batch register OpenAI accounts"
  - "run codex-console web UI"
  - "configure codex console database"
  - "package codex-console executable"
  - "fix OpenAI registration flow"
  - "deploy codex-console with docker"
---

# codex-console

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

codex-console is an enhanced, actively-maintained fork of [cnlimiter/codex-manager](https://github.com/cnlimiter/codex-manager). It provides a Web UI and CLI for automated OpenAI account registration, login, token retrieval, batch task management, log viewing, and data export. Key fixes include Sentinel POW solving, split register/login flows, deduplication of verification code sending, and improved page-state detection.

---

## Installation

### Requirements

- Python 3.10+
- `uv` (recommended) or `pip`

### Clone and Install

```bash
git clone https://github.com/dou-jiang/codex-console.git
cd codex-console

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env as needed
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `APP_HOST` | Listen host | `0.0.0.0` |
| `APP_PORT` | Listen port | `8000` |
| `APP_ACCESS_PASSWORD` | Web UI access password | `admin123` |
| `APP_DATABASE_URL` | Database connection string | `data/database.db` |

Priority order: `CLI args > .env variables > DB settings > defaults`

---

## Starting the Web UI

```bash
# Default (127.0.0.1:8000)
python webui.py

# Custom host and port
python webui.py --host 0.0.0.0 --port 8080

# With access password
python webui.py --access-password mypassword

# Debug mode (hot reload)
python webui.py --debug

# Combined
python webui.py --host 0.0.0.0 --port 8080 --access-password mypassword
```

Access the UI at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Docker Deployment

### docker-compose (recommended)

```bash
docker-compose up -d
```

Customize environment variables in `docker-compose.yml`:

```yaml
version: "3.8"
services:
  codex-console:
    image: ghcr.io/<yourname>/codex-console:latest
    ports:
      - "1455:1455"
    environment:
      WEBUI_HOST: "0.0.0.0"
      WEBUI_PORT: "1455"
      WEBUI_ACCESS_PASSWORD: "${WEBUI_ACCESS_PASSWORD}"
      LOG_LEVEL: "info"
    volumes:
      - ./data:/app/data
```

### docker run

```bash
docker run -d \
  -p 1455:1455 \
  -e WEBUI_HOST=0.0.0.0 \
  -e WEBUI_PORT=1455 \
  -e WEBUI_ACCESS_PASSWORD="${WEBUI_ACCESS_PASSWORD}" \
  -v $(pwd)/data:/app/data \
  --name codex-console \
  ghcr.io/<yourname>/codex-console:latest
```

> ⚠️ Always mount `-v $(pwd)/data:/app/data` to persist database and account data across container restarts.

Docker environment variables:

| Variable | Description |
|---|---|
| `WEBUI_HOST` | Listen host (default `0.0.0.0`) |
| `WEBUI_PORT` | Listen port (default `1455`) |
| `WEBUI_ACCESS_PASSWORD` | Web UI password |
| `DEBUG` | Set `1` or `true` for debug mode |
| `LOG_LEVEL` | Log level: `info`, `debug`, etc. |

---

## Database Configuration

### SQLite (default)

```bash
# Uses data/database.db by default
python webui.py
```

### Remote PostgreSQL

```bash
export APP_DATABASE_URL="postgresql://user:password@host:5432/dbname"
python webui.py
```

Also supports `DATABASE_URL` env var (lower priority than `APP_DATABASE_URL`).

---

## Building Executables

### Windows

```bat
build.bat
```

Output: `dist/codex-console-windows-X64.exe`

The built executable supports CLI arguments:

```bash
codex-console.exe --access-password mypassword
codex-console.exe --host 0.0.0.0 --port 8080
```

### Linux / macOS

```bash
bash build.sh
```

Build troubleshooting checklist:
- Python is added to PATH
- All dependencies installed (`uv sync` or `pip install -r requirements.txt`)
- Antivirus is not blocking PyInstaller output
- Check terminal output for specific error messages

---

## Key Architectural Changes (vs upstream)

Understanding these helps when extending or debugging:

### 1. Sentinel POW Solving

OpenAI now enforces Sentinel Proof-of-Work validation. The original codebase passed empty values; this fork implements actual POW solving:

```python
# The registration flow now calls POW solver before submitting
pow_token = solve_sentinel_pow(challenge)
headers["openai-sentinel-token"] = pow_token
```

### 2. Split Register + Login Flow

Registration no longer returns a usable token directly. The flow is now:

```python
# Step 1: Register account
register_result = await register_account(email, password)

# Step 2: Separately log in to retrieve token
token = await login_and_get_token(email, password)
```

### 3. Verification Code Handling

The server sends the verification code email automatically. The old logic sent a duplicate manual request, causing conflicts:

```python
# Old (broken): manually trigger code send
# await send_verification_code(email)  # REMOVED

# New: wait for the system-sent code
code = await wait_for_verification_email(email_client)
```

### 4. Login Page State Detection

Login re-entry now correctly detects page transitions and submits credentials at the right stage:

```python
# Detect current page state before acting
page_state = detect_login_page_state(response)
if page_state == "password_required":
    await submit_password(password)
elif page_state == "verification_required":
    await submit_verification_code(code)
```

---

## Common Patterns

### Running a Batch Registration Task

1. Open Web UI at `http://127.0.0.1:8000`
2. Navigate to **Tasks** → **New Task**
3. Configure email service (IMAP/接码 provider), proxy settings, and batch count
4. Start the task and monitor logs in real-time via the **Logs** panel

### Exporting Account Data

Use the Web UI **Export** feature to download account credentials and tokens as CSV or JSON.

### Viewing Logs

Real-time logs are available in the Web UI. For CLI log output:

```bash
python webui.py --debug 2>&1 | tee run.log
```

### Using a Custom Email Service

Set up your email接码 provider credentials in the Web UI settings panel or via environment:

```bash
export EMAIL_IMAP_HOST="imap.yourprovider.com"
export EMAIL_IMAP_USER="${EMAIL_USER}"
export EMAIL_IMAP_PASS="${EMAIL_PASS}"
```

---

## Troubleshooting

### Web UI not accessible

```bash
# Check the host binding — 127.0.0.1 only allows local access
python webui.py --host 0.0.0.0 --port 8000
```

### Registration stuck or failing

- Ensure your proxy is configured and working
- Check logs for Sentinel POW errors — update to latest commit
- Verify email接码 service credentials are correct

### Database errors on startup

```bash
# Ensure data directory exists
mkdir -p data

# For PostgreSQL, verify connection string
psql "postgresql://user:password@host:5432/dbname" -c "SELECT 1"
```

### PyInstaller build fails on Windows

```bat
# Run in a clean environment
pip install pyinstaller
pyinstaller --clean codex-console.spec
```

Check for antivirus interference — whitelist the `dist/` directory during build.

### Token retrieval returns empty

This typically means the login flow is hitting a new page state. Check:
- That you're on the latest commit (login page detection is actively maintained)
- Proxy is not flagged by OpenAI
- Verification code email was received and parsed correctly

---

## Project Structure (key files)

```
codex-console/
├── webui.py              # Main entry point — starts Web UI
├── build.bat             # Windows packaging script
├── build.sh              # Linux/macOS packaging script
├── docker-compose.yml    # Docker Compose configuration
├── .env.example          # Environment variable template
├── data/                 # SQLite DB and account data (persisted)
└── requirements.txt      # Python dependencies
```

---

## Attribution

This project is a fix-and-enhancement fork of [cnlimiter/codex-manager](https://github.com/cnlimiter/codex-manager). When publishing or redistributing, include:

```
Forked and fixed from cnlimiter/codex-manager
```

**Disclaimer:** For learning, research, and technical exchange only. Comply with all platform terms of service. Users assume all responsibility for use.
```
