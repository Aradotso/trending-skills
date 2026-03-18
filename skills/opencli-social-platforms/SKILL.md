---
name: opencli-social-platforms
description: Use opencli CLI to control 16 social/content platforms (Bilibili, Twitter/X, YouTube, Zhihu, Reddit, HackerNews, Weibo, etc.) via Claude Code by reusing your Chrome login sessions — no API keys needed.
triggers:
  - search YouTube for videos
  - get trending on Twitter
  - check Bilibili hot list
  - search Reddit posts
  - get HackerNews top stories
  - check stock price on Yahoo Finance
  - post a tweet from Claude
  - browse Zhihu hot topics
---

# opencli-skill — Social Platform CLI for Claude Code

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Control Bilibili, Twitter/X, YouTube, Zhihu, Reddit, HackerNews, Weibo, Xueqiu, and 8 more platforms using natural language — by reusing your existing Chrome login sessions via the opencli CLI tool. No API keys, no re-authentication.

---

## How It Works

opencli is a CLI tool that drives your real Chrome browser (via Playwright MCP Bridge) to interact with platforms using your existing logged-in sessions. This skill wraps opencli commands and exposes them through Claude Code.

**Architecture:**
```
User (natural language) → Claude Code → opencli CLI → Playwright MCP Bridge Chrome Extension → Chrome (logged-in session) → Platform
```

---

## Prerequisites Checklist

Before using any commands, verify ALL of these:

- [ ] **Node.js v16+** installed (`node --version`)
- [ ] **Chrome** open and logged in to target platforms
- [ ] **Playwright MCP Bridge** Chrome extension installed
- [ ] **Playwright MCP** configured in Claude Code
- [ ] **opencli** installed globally

---

## Installation (4 Steps)

### Step 1 — Install opencli

```bash
npm install -g @jackwener/opencli

# Verify
opencli --version
```

### Step 2 — Install Playwright MCP Bridge Chrome Extension

Install from Chrome Web Store:
```
https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn
```

Confirm the extension icon appears in Chrome's toolbar.

### Step 3 — Configure Playwright MCP in Claude Code

```bash
# Add playwright MCP server (run once)
claude mcp add playwright --scope user -- npx @playwright/mcp@latest

# Verify it was added
claude mcp list
# You should see "playwright" in the list
```

### Step 4 — Install This Skill

```bash
npx skills add joeseesun/opencli-skill

# Then restart Claude Code
```

---

## Supported Platforms & Capabilities

| Platform | Read | Search | Write |
|---|---|---|---|
| Bilibili (B站) | Hot/Ranking/Feed/History | Videos/Users | — |
| Zhihu (知乎) | Hot list | ✅ | Question details |
| Weibo (微博) | Trending | — | Post (Playwright) |
| Twitter/X | Timeline/Trending/Bookmarks | ✅ | Post/Reply/Like |
| YouTube | — | ✅ | — |
| Xiaohongshu (小红书) | Recommended feed | ✅ | — |
| Reddit | Home/Hot | ✅ | — |
| HackerNews | Top stories | — | — |
| V2EX | Hot/Latest | — | Daily check-in |
| Xueqiu (雪球) | Hot/Stocks/Watchlist | ✅ | — |
| BOSS直聘 | — | Jobs | — |
| BBC | News | — | — |
| Reuters | — | ✅ | — |
| 什么值得买 | — | Deals | — |
| Yahoo Finance | Stock quotes | — | — |
| Ctrip (携程) | — | Attractions/Cities | — |

---

## Command Reference by Platform

### Bilibili (B站)

```bash
# Get hot/trending videos (top 10)
opencli bilibili hot --limit 10 -f json

# Get ranking list
opencli bilibili ranking -f json

# Get your feed/recommendations
opencli bilibili feed -f json

# Get watch history
opencli bilibili history -f json

# Search videos
opencli bilibili search --keyword "AI大模型" -f json

# Search users
opencli bilibili search-user --keyword "科技" -f json
```

### Twitter / X

```bash
# Get your timeline
opencli twitter timeline -f json

# Get trending topics
opencli twitter trending -f json

# Get your bookmarks
opencli twitter bookmarks -f json

# Search tweets
opencli twitter search --query "claude AI" -f json

# Post a tweet (requires confirmation)
opencli twitter post --text "Hello from Claude Code!"

# Reply to a tweet
opencli twitter reply --tweet-id 1234567890 --text "Great point!"

# Like a tweet
opencli twitter like --tweet-id 1234567890
```

### YouTube

```bash
# Search videos
opencli youtube search --query "LLM tutorial" -f json

# Search with limit
opencli youtube search --query "machine learning 2024" --limit 20 -f json
```

### Zhihu (知乎)

```bash
# Get hot list
opencli zhihu hot -f json

# Search questions/answers
opencli zhihu search --keyword "大模型" -f json

# Get question details
opencli zhihu question --id 12345678 -f json
```

### Weibo (微博)

```bash
# Get trending/hot search
opencli weibo hot -f json

# Post a Weibo update (requires confirmation — uses Playwright)
opencli weibo post --text "今天天气真好"
```

### HackerNews

```bash
# Get top stories
opencli hackernews top --limit 20 -f json

# Get new stories
opencli hackernews new --limit 20 -f json

# Get best stories
opencli hackernews best --limit 20 -f json
```

### Reddit

```bash
# Get home feed
opencli reddit home -f json

# Get hot posts from a subreddit
opencli reddit hot --subreddit MachineLearning -f json

# Get hot posts (default front page)
opencli reddit hot -f json

# Search Reddit
opencli reddit search --query "transformer papers" -f json
```

### Xueqiu (雪球) — Stock Market

```bash
# Get hot stocks/discussions
opencli xueqiu hot -f json

# Get stock quote
opencli xueqiu stock --symbol SH600519   # Moutai (茅台)
opencli xueqiu stock --symbol AAPL       # Apple

# Get your watchlist
opencli xueqiu watchlist -f json

# Search stocks
opencli xueqiu search --keyword "新能源" -f json
```

### Yahoo Finance

```bash
# Get stock quote
opencli yahoo-finance quote --symbol AAPL -f json
opencli yahoo-finance quote --symbol TSLA -f json
opencli yahoo-finance quote --symbol BTC-USD -f json
```

### Xiaohongshu (小红书)

```bash
# Get recommended feed
opencli xiaohongshu feed -f json

# Search notes
opencli xiaohongshu search --keyword "旅行攻略" -f json
```

### V2EX

```bash
# Get hot topics
opencli v2ex hot -f json

# Get latest topics
opencli v2ex latest -f json

# Daily check-in
opencli v2ex checkin
```

### BOSS直聘

```bash
# Search jobs
opencli boss search --keyword "AI工程师" -f json
opencli boss search --keyword "frontend developer" --city "北京" -f json
```

### BBC News

```bash
# Get latest news
opencli bbc news -f json
```

### Reuters

```bash
# Search Reuters articles
opencli reuters search --query "artificial intelligence" -f json
```

### 什么值得买

```bash
# Search deals
opencli smzdm search --keyword "显卡" -f json
```

### Ctrip (携程)

```bash
# Search attractions
opencli ctrip attractions --city "北京" -f json

# Search cities
opencli ctrip cities --keyword "云南" -f json
```

---

## Output Format

All commands support `-f json` for structured JSON output. Example output from `opencli hackernews top --limit 5 -f json`:

```json
{
  "stories": [
    {
      "rank": 1,
      "title": "Show HN: I built a tool that...",
      "url": "https://example.com",
      "score": 342,
      "comments": 87,
      "author": "username",
      "time": "2026-03-18T10:00:00Z"
    }
  ]
}
```

---

## Common Usage Patterns for AI Agents

### Pattern 1: Get trending content and summarize

```bash
# Fetch trending, pipe into analysis
opencli bilibili hot --limit 20 -f json
opencli twitter trending -f json
opencli hackernews top --limit 20 -f json
```

Claude then summarizes, translates titles (English→Chinese or vice versa), and presents as a clean table.

### Pattern 2: Cross-platform search

```bash
# Search the same topic across platforms
opencli youtube search --query "rust programming" -f json
opencli reddit search --query "rust programming" -f json
opencli hackernews top -f json  # then filter client-side
```

### Pattern 3: Stock research workflow

```bash
# Check multiple stocks
opencli yahoo-finance quote --symbol NVDA -f json
opencli xueqiu stock --symbol SH600519 -f json
opencli xueqiu hot -f json  # see what's trending in Chinese markets
```

### Pattern 4: Write operations (ALWAYS confirm first)

```bash
# Claude MUST show content and ask for user confirmation before running:
opencli twitter post --text "Just discovered opencli — control 16 platforms from your terminal!"
opencli weibo post --text "用 Claude Code 发微博太方便了！"
opencli v2ex checkin
```

---

## Scripting & Automation Examples

### Bash: Daily digest script

```bash
#!/bin/bash
# daily-digest.sh — Run from Claude Code to get morning briefing

echo "=== HackerNews Top 10 ==="
opencli hackernews top --limit 10 -f json | jq '.stories[] | "\(.rank). \(.title) (\(.score) pts)"' -r

echo ""
echo "=== Twitter Trending ==="
opencli twitter trending -f json | jq '.trends[] | "\(.rank). \(.name)"' -r

echo ""
echo "=== Bilibili Hot ==="
opencli bilibili hot --limit 10 -f json | jq '.videos[] | "\(.rank). \(.title)"' -r
```

### Node.js: Parse opencli JSON output

```javascript
const { execSync } = require('child_process');

function getHackerNewsTop(limit = 10) {
  const raw = execSync(`opencli hackernews top --limit ${limit} -f json`, {
    encoding: 'utf-8'
  });
  return JSON.parse(raw);
}

function searchYouTube(query) {
  const raw = execSync(
    `opencli youtube search --query "${query}" -f json`,
    { encoding: 'utf-8' }
  );
  return JSON.parse(raw);
}

function getStockQuote(symbol) {
  const raw = execSync(
    `opencli yahoo-finance quote --symbol ${symbol} -f json`,
    { encoding: 'utf-8' }
  );
  return JSON.parse(raw);
}

// Usage
const hn = getHackerNewsTop(5);
console.log('Top HN stories:', hn.stories.map(s => s.title));

const stocks = getStockQuote('AAPL');
console.log('AAPL price:', stocks.price);
```

### Python: Multi-platform aggregator

```python
import subprocess
import json

def opencli(args: list[str]) -> dict:
    """Run an opencli command and return parsed JSON."""
    result = subprocess.run(
        ['opencli'] + args + ['-f', 'json'],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)

def get_tech_digest():
    """Aggregate tech news from multiple platforms."""
    sources = {}
    
    try:
        sources['hackernews'] = opencli(['hackernews', 'top', '--limit', '10'])
    except subprocess.CalledProcessError as e:
        sources['hackernews'] = {'error': str(e)}
    
    try:
        sources['reddit_ml'] = opencli(['reddit', 'hot', '--subreddit', 'MachineLearning'])
    except subprocess.CalledProcessError as e:
        sources['reddit_ml'] = {'error': str(e)}
    
    try:
        sources['twitter'] = opencli(['twitter', 'trending'])
    except subprocess.CalledProcessError as e:
        sources['twitter'] = {'error': str(e)}
    
    return sources

if __name__ == '__main__':
    digest = get_tech_digest()
    print(json.dumps(digest, ensure_ascii=False, indent=2))
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `opencli: command not found` | Not installed or PATH issue | `npm install -g @jackwener/opencli`; check `echo $PATH` includes npm global bin |
| `Error: Chrome not responding` | Chrome not open | Open Chrome before running any command |
| `Login state not found` | Not logged in to that platform | Manually log in to the site in Chrome, then retry |
| `Playwright MCP not configured` | Step 3 was skipped | Run: `claude mcp add playwright --scope user -- npx @playwright/mcp@latest` |
| Extension not working | Playwright MCP Bridge not enabled | Go to `chrome://extensions/` and enable it |
| JSON parse error | Command failed silently | Try without `-f json` first to see raw error output |
| Rate limit / CAPTCHA triggered | Too many requests | Wait 5–10 minutes; avoid rapid repeated calls to same platform |
| `npx skills add` fails | Node.js too old | Upgrade to Node.js v16+: `node --version` |
| Write op posted without confirmation | Shouldn't happen | Claude must always show content and ask before posting |

### Debug: Test opencli directly

```bash
# Test without Claude — run in terminal
opencli hackernews top --limit 3 -f json

# If this works, opencli is fine; issue is in Claude Code integration
# If this fails, fix opencli install first
```

### Check MCP configuration

```bash
claude mcp list
# Expected output includes:
# playwright    npx @playwright/mcp@latest
```

---

## Write Operations Safety Rules

When asked to post content, Claude Code MUST:

1. **Show the exact content** that will be posted
2. **Explicitly ask for confirmation** before executing
3. **Never post automatically** without user approval
4. **Warn about irreversibility** — posted content is immediately public

Example safe interaction pattern:
```
User: "Post a tweet saying I'm learning Rust"

Claude: I'll post the following tweet:
"I'm learning Rust! 🦀"

⚠️ This will be posted publicly and cannot be automatically recalled.
Confirm? (yes/no)

[User: yes]

Claude: *runs* opencli twitter post --text "I'm learning Rust! 🦀"
```

---

## Natural Language → Command Mapping

| What a user says | Command Claude should run |
|---|---|
| "What's trending on Twitter?" | `opencli twitter trending -f json` |
| "Search YouTube for React tutorials" | `opencli youtube search --query "React tutorials" -f json` |
| "Get top 20 HackerNews stories" | `opencli hackernews top --limit 20 -f json` |
| "Check r/MachineLearning hot posts" | `opencli reddit hot --subreddit MachineLearning -f json` |
| "What's hot on Bilibili today?" | `opencli bilibili hot --limit 20 -f json` |
| "Get Zhihu hot list" | `opencli zhihu hot -f json` |
| "Check TSLA stock price" | `opencli yahoo-finance quote --symbol TSLA -f json` |
| "Get my Twitter timeline" | `opencli twitter timeline -f json` |
| "Search Zhihu for AI topics" | `opencli zhihu search --keyword "AI" -f json` |
| "Get Weibo trending topics" | `opencli weibo hot -f json` |
| "Check 茅台 stock" | `opencli xueqiu stock --symbol SH600519 -f json` |
| "Get BBC news" | `opencli bbc news -f json` |
| "Search BOSS for AI jobs in Beijing" | `opencli boss search --keyword "AI" --city "北京" -f json` |

---

## Links

- **opencli GitHub**: https://github.com/jackwener/opencli
- **Playwright MCP Bridge Extension**: https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn
- **Full command reference**: `references/commands.md` in the skill repo
- **Claude Code docs**: https://claude.ai/code
