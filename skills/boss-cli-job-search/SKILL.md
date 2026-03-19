```markdown
---
name: boss-cli-job-search
description: CLI tool for BOSS 直聘 job search, recommendations, applications, and recruiter messaging via reverse-engineered API
triggers:
  - search jobs on boss zhipin
  - find jobs using boss cli
  - send greetings to recruiters on boss
  - export job listings to csv
  - view job recommendations boss zhipin
  - manage job applications boss cli
  - batch greet recruiters boss
  - authenticate boss zhipin cli
---

# boss-cli Job Search & Recruiter Tool

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Python CLI for BOSS 直聘 (boss.zhipin.com) that enables job searching, personalized recommendations, application management, and recruiter messaging via a reverse-engineered API. Supports 10+ browsers for cookie extraction, QR login, structured JSON/YAML output for agent pipelines, and anti-detection rate limiting.

## Installation

```bash
# Recommended (fast, isolated)
uv tool install kabi-boss-cli

# Or pipx
pipx install kabi-boss-cli

# With YAML output support
pip install kabi-boss-cli[yaml]

# Upgrade
uv tool upgrade kabi-boss-cli

# From source
git clone git@github.com:jackwener/boss-cli.git
cd boss-cli
uv sync
```

## Authentication

Credentials are stored at `~/.config/boss-cli/credential.json`. Cookies auto-refresh from browser after 7 days.

```bash
# Auto-detect browser cookies, fallback to QR code
boss login

# Extract from a specific browser
boss login --cookie-source chrome
boss login --cookie-source firefox
boss login --cookie-source edge
boss login --cookie-source brave
boss login --cookie-source arc
boss login --cookie-source safari

# QR code login only (scan with BOSS 直聘 APP)
boss login --qrcode

# Verify session against real API
boss status
boss status --json

# Clear saved credentials
boss logout
```

`boss status --json` returns per-flow health:
```json
{
  "ok": true,
  "schema_version": "1",
  "data": {
    "authenticated": true,
    "search_authenticated": true,
    "recommend_authenticated": true,
    "cookie_names": ["__zp_stoken__", "bst", "buid"]
  }
}
```

## Job Search

```bash
# Basic search
boss search "golang"
boss search "Python 工程师"

# Filter by city (use `boss cities` for full list)
boss search "Python" --city 杭州
boss search "Java" --city 北京

# Filter by salary range
boss search "Java" --salary 20-30K

# Filter by experience
boss search "前端" --exp 3-5年
boss search "后端" --exp 1-3年

# Filter by degree
boss search "AI" --degree 硕士
boss search "运维" --degree 本科

# Filter by industry
boss search "后端" --industry 互联网

# Filter by company scale
boss search "产品" --scale 1000-9999人
boss search "数据" --scale 500-999人

# Filter by funding stage
boss search "数据" --stage 已上市
boss search "AI" --stage B轮

# Filter by job type
boss search "运维" --job-type 全职
boss search "设计" --job-type 兼职

# Combine multiple filters
boss search "Python" --city 深圳 --salary 15-25K --exp 1-3年 --degree 本科

# Pagination
boss search "后端" --city 深圳 -p 2
boss search "golang" -p 3

# JSON output (structured envelope)
boss search "Python" --json
boss search "golang" --json | jq .data.jobs
```

## Job Details & Export

```bash
# Show job #3 from last search (short-index navigation)
boss show 3
boss show 1 --json

# View full job details by securityId
boss detail <securityId>
boss detail <securityId> --json

# Export search results to CSV
boss export "Python" -n 50 -o jobs.csv

# Export as JSON
boss export "golang" --format json -o jobs.json

# Export with filters
boss export "Java" --city 上海 --salary 20-30K -n 100 -o shanghai_java.csv
```

## Recommendations & Personal Data

```bash
# Personalized job recommendations
boss recommend
boss recommend -p 2
boss recommend --json

# View profile / resume status
boss me
boss me --json

# View applied jobs
boss applied
boss applied --json

# View interview invitations
boss interviews
boss interviews --json

# View browsing history
boss history

# View communicated bosses list
boss chat
```

## Recruiter Greeting (Apply / Message)

```bash
# Send greeting to a single recruiter (by job securityId)
boss greet <securityId>
boss greet <securityId> --json

# Batch greet top N results from a search
boss batch-greet "golang" --city 杭州 -n 5
boss batch-greet "Python" --salary 20-30K -n 10

# Dry run — preview only, no messages sent
boss batch-greet "Python" --salary 20-30K --dry-run

# Batch greet with multiple filters
boss batch-greet "Java" --city 北京 --exp 3-5年 -n 20 --dry-run
```

> **Note:** Batch greet uses a 1.5s delay between requests to avoid rate limiting.

## Supported Cities

```bash
# List all 40+ supported cities
boss cities
```

Common cities: 北京, 上海, 广州, 深圳, 杭州, 成都, 武汉, 西安, 南京, 苏州, 重庆, 天津, 长沙, 郑州, 青岛

## Structured Output (Agent Pipelines)

All `--json` / `--yaml` commands use a unified envelope:

```json
{
  "ok": true,
  "schema_version": "1",
  "data": { ... }
}
```

- **Rich display** → stderr (never pollutes pipes)
- **Non-TTY stdout** → auto YAML
- **`--json`** → explicit JSON
- **`--yaml`** → explicit YAML (requires `kabi-boss-cli[yaml]`)

```bash
# Pipe-safe patterns
boss search "Python" --json | jq '.data.jobs[] | {title, company, salary}'
boss recommend --json | jq '.data.jobs[0]'
boss me --json | jq '.data.profile.name'

# Check auth in scripts
boss status --json | jq '.data.search_authenticated'

# Export pipeline
boss export "golang" --format json -o - | jq '.[] | select(.salary > 30)'
```

## Verbose / Debug Logging

```bash
# Show request URLs, status codes, and timing
boss -v search "Python"
boss -v recommend
boss -v greet <securityId>
```

## Common Patterns

### Script: Search and Export

```python
import subprocess
import json

result = subprocess.run(
    ["boss", "search", "Python", "--city", "上海", "--json"],
    capture_output=True, text=True
)
envelope = json.loads(result.stdout)
if envelope["ok"]:
    jobs = envelope["data"]["jobs"]
    for job in jobs:
        print(f"{job['title']} @ {job['company']} — {job['salary']}")
```

### Script: Batch Apply with Logging

```python
import subprocess
import json
import time

def batch_greet(keyword: str, city: str, count: int = 10, dry_run: bool = True):
    cmd = [
        "boss", "batch-greet", keyword,
        "--city", city,
        "-n", str(count),
        "--json"
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    envelope = json.loads(result.stdout)
    
    if envelope["ok"]:
        print(f"Greeted {len(envelope['data']['results'])} recruiters")
        return envelope["data"]["results"]
    else:
        print(f"Error: {envelope.get('error')}")
        return []

# Preview first
batch_greet("golang", "深圳", count=5, dry_run=True)

# Then execute
batch_greet("golang", "深圳", count=5, dry_run=False)
```

### Script: Check Auth and Re-login if Needed

```python
import subprocess
import json
import sys

def ensure_authenticated() -> bool:
    result = subprocess.run(
        ["boss", "status", "--json"],
        capture_output=True, text=True
    )
    try:
        envelope = json.loads(result.stdout)
        return envelope.get("ok") and envelope["data"].get("search_authenticated")
    except json.JSONDecodeError:
        return False

if not ensure_authenticated():
    print("Not authenticated. Running boss login...")
    subprocess.run(["boss", "login", "--cookie-source", "chrome"])
    if not ensure_authenticated():
        print("Login failed. Try: boss login --qrcode")
        sys.exit(1)

print("Authenticated. Proceeding...")
```

### Script: Export Multiple Keywords

```python
import subprocess

keywords = ["Python", "golang", "Java", "前端", "数据分析"]
city = "上海"

for kw in keywords:
    output_file = f"jobs_{kw}_{city}.csv"
    subprocess.run([
        "boss", "export", kw,
        "--city", city,
        "-n", "50",
        "-o", output_file
    ])
    print(f"Exported {kw} → {output_file}")
```

## Rate Limiting & Anti-Detection (Built-in)

boss-cli handles this automatically:

| Mechanism | Details |
|-----------|---------|
| Gaussian jitter | `random.gauss(0.3, 0.15)` delay between requests |
| Random long pauses | 5% chance of 2–5s pause (mimics reading) |
| Rate-limit cooldown | Code 9 → exponential backoff 10s→20s→40s→60s |
| HTTP 429/5xx retry | Auto-retry up to 3 times |
| Browser fingerprint | macOS Chrome 145 UA, `sec-ch-ua`, `DNT`, `Priority` headers |
| Cookie merge | `Set-Cookie` headers merged back into session |

## Troubleshooting

**`boss status` reports `authenticated=false` but credential file exists**
```bash
# Session expired server-side; re-authenticate
boss logout && boss login
```

**`环境异常 (__zp_stoken__ 已过期)`**
```bash
# Cookies expired — log into boss.zhipin.com in browser first, then:
boss logout && boss login
# If browser cookies work but QR gives partial set, prefer browser method:
boss login --cookie-source chrome
```

**`暂无投递记录` despite having applied**
```bash
# Requires fresh __zp_stoken__ — re-login from browser, then:
boss login --cookie-source chrome
boss applied
```

**Search returns no results**
```bash
# Verify city spelling
boss cities
boss search "Python" --city 上海   # exact city name from cities list
```

**Rate limited / code=9 errors**
```bash
# Use verbose mode to see timing
boss -v search "Python"
# The CLI will auto-backoff; wait for cooldown to complete
# Avoid running multiple boss commands in parallel
```

**QR code not rendering in terminal**
```bash
# Ensure terminal supports Unicode half-blocks (most modern terminals do)
# Try iTerm2, Warp, or Windows Terminal
boss login --qrcode
```

## Project Structure (for Contributors)

```text
boss_cli/
├── cli.py          # Click entry point
├── client.py       # API client (rate-limit, retry, anti-detection)
├── auth.py         # Browser cookie extraction, QR login, TTL refresh
├── constants.py    # URLs, headers, city codes, filter enums
├── exceptions.py   # BossApiError hierarchy
├── index_cache.py  # Short-index cache for `boss show`
└── commands/
    ├── _common.py  # Schema envelope, handle_command, stderr console
    ├── auth.py     # login, logout, status, me
    ├── search.py   # search, recommend, detail, show, export, history, cities
    ├── personal.py # applied, interviews
    └── social.py   # chat, greet, batch-greet
```

```bash
# Development setup
uv sync
uv run pytest tests/ -v          # Unit tests
uv run pytest tests/ -v -m smoke # Smoke tests (requires valid cookies)
uv run ruff check .               # Lint
```
```
