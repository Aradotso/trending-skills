---
name: opencli-social-platforms
description: Use opencli CLI to control Bilibili, Twitter/X, YouTube, Reddit, HackerNews, Zhihu, Weibo, and 9 more platforms via natural language through Claude Code
triggers:
  - search YouTube for videos
  - get trending on Twitter
  - check Bilibili hot videos
  - browse HackerNews top stories
  - search Reddit posts
  - check stock price on Yahoo Finance
  - post a tweet
  - get Zhihu hot list
---

# opencli-skill: Control 16 Social Platforms via CLI

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

opencli is a CLI tool that turns 16 major platforms (Bilibili, Twitter/X, YouTube, Reddit, HackerNews, Zhihu, Weibo, Xiaohongshu, V2EX, Xueqiu, BOSS直聘, BBC, Reuters, 什么值得买, Yahoo Finance, Ctrip) into command-line interfaces by reusing your existing Chrome login sessions. No API keys required — just log in to Chrome normally and opencli piggybacks on your session.

## Prerequisites

All four must be in place before using opencli commands:

1. **Node.js v16+** — [nodejs.org](https://nodejs.org/)
2. **Chrome browser** open and logged in to target platforms
3. **Playwright MCP Bridge** Chrome extension — [Install from Chrome Web Store](https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn)
4. **Playwright MCP** configured in Claude Code

## Installation

```bash
# Step 1: Install opencli globally
npm install -g @jackwener/opencli

# Verify installation
opencli --version

# Step 2: Install Playwright MCP Bridge extension in Chrome
# (manual step — visit Chrome Web Store link above)

# Step 3: Configure Playwright MCP in Claude Code
claude mcp add playwright --scope user -- npx @playwright/mcp@latest

# Verify Playwright MCP is registered
claude mcp list

# Step 4: Install this skill
npx skills add joeseesun/opencli-skill
```

Restart Claude Code after installation.

## Platform Support Matrix

| Platform | Read | Search | Write |
|----------|------|--------|-------|
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

## Key Commands

### Bilibili (B站)

```bash
# Get hot videos (JSON output for parsing)
opencli bilibili hot --limit 10 -f json

# Get ranking list
opencli bilibili ranking -f json

# Search videos
opencli bilibili search --keyword "AI大模型" -f json

# Search users
opencli bilibili search-user --keyword "技术博主" -f json

# Get your feed/动态
opencli bilibili feed -f json

# Get watch history
opencli bilibili history --limit 20 -f json
```

### Twitter/X

```bash
# Get your timeline
opencli twitter timeline -f json

# Get trending topics
opencli twitter trending -f json

# Get bookmarks
opencli twitter bookmarks -f json

# Search tweets
opencli twitter search --query "claude AI" -f json

# Post a tweet (CONFIRM BEFORE RUNNING)
opencli twitter post --text "Hello from Claude Code!"

# Reply to a tweet
opencli twitter reply --id "TWEET_ID" --text "Great point!"

# Like a tweet
opencli twitter like --id "TWEET_ID"
```

### YouTube

```bash
# Search videos
opencli youtube search --query "LLM tutorial" -f json

# Search with limit
opencli youtube search --query "rust programming" --limit 10 -f json
```

### HackerNews

```bash
# Get top stories
opencli hackernews top --limit 20 -f json

# Get top 5 stories
opencli hackernews top --limit 5 -f json
```

### Reddit

```bash
# Get home feed
opencli reddit home -f json

# Get hot posts from a subreddit
opencli reddit hot --subreddit MachineLearning -f json

# Search Reddit
opencli reddit search --query "rust async" -f json
```

### Zhihu (知乎)

```bash
# Get hot list
opencli zhihu hot -f json

# Search questions
opencli zhihu search --keyword "大模型" -f json

# Get question details
opencli zhihu question --id "QUESTION_ID" -f json
```

### Weibo (微博)

```bash
# Get trending/hot search
opencli weibo hot -f json

# Post to Weibo (uses Playwright, CONFIRM BEFORE RUNNING)
opencli weibo post --text "今天天气真好！"
```

### Xueqiu (雪球) — Stocks

```bash
# Get hot stocks
opencli xueqiu hot -f json

# Get stock quote (A-shares use SH/SZ prefix)
opencli xueqiu stock --symbol SH600519   # Moutai/茅台
opencli xueqiu stock --symbol SZ000858   # Wuliangye/五粮液

# US stocks
opencli xueqiu stock --symbol AAPL
opencli xueqiu stock --symbol TSLA

# Get your watchlist (must be logged in to Xueqiu)
opencli xueqiu watchlist -f json
```

### Yahoo Finance

```bash
# Get stock quote
opencli yahoo-finance quote --symbol AAPL -f json
opencli yahoo-finance quote --symbol GOOGL -f json
opencli yahoo-finance quote --symbol BTC-USD -f json
```

### Xiaohongshu (小红书)

```bash
# Get recommended feed
opencli xiaohongshu feed -f json

# Search notes
opencli xiaohongshu search --keyword "穿搭" -f json
```

### V2EX

```bash
# Get hot topics
opencli v2ex hot -f json

# Get latest posts
opencli v2ex latest -f json

# Daily check-in (signs you in for points)
opencli v2ex checkin
```

### BOSS直聘

```bash
# Search jobs
opencli boss search --keyword "前端工程师" --city "北京" -f json
opencli boss search --keyword "product manager" --city "上海" -f json
```

### BBC News

```bash
# Get news headlines
opencli bbc news -f json
```

### Reuters

```bash
# Search Reuters
opencli reuters search --query "AI regulation" -f json
```

### 什么值得买

```bash
# Search deals
opencli smzdm search --keyword "机械键盘" -f json
```

### Ctrip (携程)

```bash
# Search attractions
opencli ctrip search --keyword "北京故宫" -f json

# Search by city
opencli ctrip city --name "成都" -f json
```

## Output Format

All commands support `-f json` for structured JSON output. Always use this flag when you need to parse or display results:

```bash
# JSON output example (Bilibili hot)
opencli bilibili hot --limit 5 -f json
# Returns array of: { title, author, views, url, cover }

# Default output is human-readable table format
opencli hackernews top --limit 5
```

## Common Patterns for AI Agents

### Pattern 1: Fetch and display in a table

```bash
# Get HackerNews top stories, display as table
opencli hackernews top --limit 20 -f json
# Agent formats JSON into markdown table with rank, title, score, url
```

### Pattern 2: Cross-platform topic search

```bash
# Search the same topic across multiple platforms
opencli zhihu search --keyword "Rust编程" -f json
opencli reddit search --query "Rust programming" -f json
opencli hackernews top --limit 30 -f json  # then filter client-side
```

### Pattern 3: Stock research workflow

```bash
# Get market overview then specific stocks
opencli xueqiu hot -f json
opencli xueqiu stock --symbol SH600519 -f json
opencli yahoo-finance quote --symbol BABA -f json
```

### Pattern 4: Social media monitoring

```bash
# Check what's trending across platforms
opencli twitter trending -f json
opencli weibo hot -f json
opencli bilibili hot --limit 10 -f json
opencli zhihu hot -f json
```

### Pattern 5: Write operations (always confirm first)

```bash
# ALWAYS show the user what will be posted and wait for explicit confirmation

# Twitter
opencli twitter post --text "Content to post"

# Weibo
opencli weibo post --text "内容"

# V2EX check-in (safe, non-public)
opencli v2ex checkin
```

## Output Parsing Tips

```bash
# Pipe JSON to jq for filtering (if jq is available)
opencli bilibili hot --limit 20 -f json | jq '.[].title'

# Get just the top 5 titles from HackerNews
opencli hackernews top --limit 5 -f json | jq '[.[] | {title, url, score}]'

# Filter Reddit posts by score
opencli reddit hot --subreddit programming -f json | jq '[.[] | select(.score > 1000)]'
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `opencli: command not found` | Not installed or PATH issue | `npm install -g @jackwener/opencli`; check `echo $PATH` |
| Chrome not being controlled | Extension missing or Chrome closed | Open Chrome; verify Playwright MCP Bridge extension is enabled |
| Login state not recognized | Not logged in on Chrome | Manually log in on Chrome first, then retry |
| `Playwright MCP not found` error | MCP not configured | Run `claude mcp add playwright --scope user -- npx @playwright/mcp@latest` |
| Empty results returned | Rate limited or session expired | Wait a moment; refresh the page in Chrome manually |
| `npx skills add` fails | Old Node.js | Upgrade to Node.js v16+: `node --version` |
| CAPTCHA triggered | Bot detection | Complete CAPTCHA manually in Chrome, then retry |
| Weibo post fails | Playwright not connecting | Ensure Chrome is open with Playwright MCP Bridge active |

## Important Warnings

⚠️ **Write operations** (posting tweets, Weibo updates, V2EX check-ins):
- Always show the user the exact content and request explicit confirmation before executing
- Platform bot-detection may trigger rate limits or temporary bans
- Published content is immediately public — the AI cannot recall it
- Never post rapidly in succession

⚠️ **Login sessions**: opencli reuses your Chrome session. If Chrome is closed or you log out of a platform, commands for that platform will fail until you log back in via Chrome.

⚠️ **Rate limits**: Avoid running many commands in rapid succession against the same platform. Space requests out by a few seconds minimum.
