---
name: opencli-social-platforms
description: Use opencli CLI to control 16 social/content platforms (Bilibili, Twitter/X, YouTube, Reddit, HackerNews, Zhihu, Weibo, etc.) via natural language by reusing existing Chrome login sessions — no API keys needed.
triggers:
  - search YouTube for videos
  - get trending on Twitter or Bilibili
  - check HackerNews top stories
  - search Reddit posts
  - post a tweet using opencli
  - check stock price on Yahoo Finance
  - get Bilibili hot videos
  - browse social media platforms with Claude
---

# opencli-skill: Control 16 Social Platforms via CLI

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

opencli wraps 16 major platforms (Bilibili, Twitter/X, YouTube, Reddit, HackerNews, Zhihu, Weibo, Xiaohongshu, V2EX, Xueqiu, BOSS直聘, BBC, Reuters, 什么值得买, Yahoo Finance, Ctrip) into CLI commands that **reuse your existing Chrome login sessions** — no API keys, no re-authentication.

## How It Works

1. User is logged into platforms in Chrome
2. Playwright MCP Bridge Chrome extension lets opencli control Chrome
3. Claude runs `opencli <platform> <command>` to fetch/post data
4. Results are returned as JSON and rendered as tables

## Prerequisites Checklist

- [ ] Node.js v16+
- [ ] Chrome open and logged into target platforms
- [ ] [Playwright MCP Bridge](https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn) Chrome extension installed
- [ ] Playwright MCP configured in Claude Code
- [ ] Claude Code installed

## Installation

### Step 1 — Install opencli globally

```bash
npm install -g @jackwener/opencli
opencli --version   # verify
```

### Step 2 — Install Playwright MCP Bridge in Chrome

Install from the [Chrome Web Store](https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn). Confirm the extension icon appears in Chrome's toolbar.

### Step 3 — Configure Playwright MCP in Claude Code

```bash
claude mcp add playwright --scope user -- npx @playwright/mcp@latest
claude mcp list   # verify "playwright" appears
```

### Step 4 — Install the skill

```bash
npx skills add joeseesun/opencli-skill
```

Restart Claude Code to activate.

## Platform Commands Reference

### Bilibili (B站)

```bash
# Hot/trending videos
opencli bilibili hot --limit 10 -f json

# Video rankings
opencli bilibili ranking -f json

# Search videos
opencli bilibili search --keyword "AI大模型" -f json

# Search users
opencli bilibili search-user --keyword "技术UP主" -f json

# Your feed/timeline
opencli bilibili feed -f json

# Watch history
opencli bilibili history -f json
```

### Twitter/X

```bash
# Your timeline
opencli twitter timeline -f json

# Trending topics
opencli twitter trending -f json

# Your bookmarks
opencli twitter bookmarks -f json

# Search tweets
opencli twitter search --query "claude AI" -f json

# Post a tweet (requires confirmation before use)
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
opencli youtube search --query "machine learning 2026" --limit 20 -f json
```

### HackerNews

```bash
# Top stories
opencli hackernews top --limit 20 -f json

# New stories
opencli hackernews new --limit 10 -f json
```

### Reddit

```bash
# Home feed
opencli reddit home -f json

# Subreddit hot posts
opencli reddit hot --subreddit MachineLearning -f json
opencli reddit hot --subreddit programming --limit 15 -f json

# Search Reddit
opencli reddit search --query "transformer architecture" -f json
```

### Zhihu (知乎)

```bash
# Hot list
opencli zhihu hot -f json

# Search
opencli zhihu search --keyword "大模型" -f json

# Question details
opencli zhihu question --id "QUESTION_ID" -f json
```

### Weibo (微博)

```bash
# Trending/hot search
opencli weibo hot -f json

# Post a Weibo update (requires confirmation)
# Uses Playwright browser automation
opencli weibo post --text "今天天气真好 #AI#"
```

### Xiaohongshu (小红书)

```bash
# Recommended feed
opencli xiaohongshu feed -f json

# Search notes
opencli xiaohongshu search --keyword "穿搭" -f json
```

### Xueqiu (雪球) — Stock Market

```bash
# Hot stocks/discussions
opencli xueqiu hot -f json

# Stock quote
opencli xueqiu stock --symbol SH600519   # 茅台
opencli xueqiu stock --symbol AAPL       # Apple

# Your watchlist
opencli xueqiu watchlist -f json

# Search stocks
opencli xueqiu search --keyword "新能源" -f json
```

### Yahoo Finance

```bash
# Stock quote
opencli yahoo-finance quote --symbol AAPL -f json
opencli yahoo-finance quote --symbol TSLA -f json
opencli yahoo-finance quote --symbol BTC-USD -f json
```

### BOSS直聘 (Job Search)

```bash
# Search jobs
opencli boss search --keyword "AI工程师" -f json
opencli boss search --keyword "前端开发" --city "北京" -f json
```

### V2EX

```bash
# Hot topics
opencli v2ex hot -f json

# Latest posts
opencli v2ex latest -f json

# Daily check-in
opencli v2ex checkin
```

### BBC News

```bash
# News headlines
opencli bbc news -f json
opencli bbc news --limit 15 -f json
```

### Reuters

```bash
# Search news
opencli reuters search --query "artificial intelligence" -f json
```

### 什么值得买 (Deals)

```bash
# Search deals
opencli smzdm search --keyword "机械键盘" -f json
```

### Ctrip / 携程 (Travel)

```bash
# Search attractions
opencli ctrip attractions --city "北京" -f json

# Search cities
opencli ctrip cities --keyword "云南" -f json
```

## Common Patterns

### Fetch and display trending content

```bash
# Get top 20 HackerNews stories as JSON
opencli hackernews top --limit 20 -f json

# Get Bilibili hot videos, limit 15
opencli bilibili hot --limit 15 -f json

# Get Twitter trending topics
opencli twitter trending -f json
```

### Multi-platform search workflow

```bash
# Search a topic across platforms
opencli youtube search --query "Claude Code tutorial" -f json
opencli reddit search --query "Claude Code" -f json
opencli hackernews top --limit 30 -f json  # then filter client-side
```

### Stock research workflow

```bash
# Check multiple stocks
opencli yahoo-finance quote --symbol NVDA -f json
opencli xueqiu stock --symbol SH600519 -f json
opencli xueqiu watchlist -f json
```

### Write operations (always confirm first)

```bash
# Claude MUST show content and get user confirmation before any write op

# Post tweet
opencli twitter post --text "Just discovered opencli — control 16 platforms from CLI!"

# Post Weibo
opencli weibo post --text "用Claude Code搜遍全网，太方便了！"

# V2EX check-in
opencli v2ex checkin
```

## Output Format

All read commands support `-f json` for structured output:

```json
{
  "items": [
    {
      "rank": 1,
      "title": "Show HN: I built a thing",
      "url": "https://...",
      "score": 342,
      "comments": 87,
      "author": "username"
    }
  ],
  "total": 20,
  "platform": "hackernews"
}
```

Without `-f json`, output is human-readable plain text tables.

## Agent Behavior Guidelines

When helping users with opencli:

1. **Always use `-f json`** for programmatic processing — easier to parse and display as tables
2. **For write operations** (post, reply, like, checkin): show the exact content/action to the user and ask for explicit confirmation before running the command
3. **If Chrome isn't open**: remind the user to open Chrome and log in to the target platform first
4. **Translate titles**: for non-English content (Bilibili, Zhihu, Weibo, etc.), provide English translations of titles in your response
5. **Default limits**: use `--limit 10` unless user specifies otherwise to avoid overwhelming output
6. **JSON parsing**: pipe through a formatter or parse in code when displaying results

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `opencli: command not found` | Run `npm install -g @jackwener/opencli`; check `echo $PATH` includes npm global bin |
| Chrome not being controlled | Ensure Chrome is open; verify Playwright MCP Bridge extension is enabled in `chrome://extensions` |
| Login state not recognized | Manually log in to the target site in Chrome, then retry |
| `Playwright MCP not found` error | Re-run: `claude mcp add playwright --scope user -- npx @playwright/mcp@latest` |
| `npx skills add` fails | Ensure Node.js v16+ is installed: `node --version` |
| CAPTCHA triggered | Platform bot-detection fired; wait a few minutes and retry manually |
| Empty results returned | You may not be logged in, or the platform changed its structure |

## Write Operations Risk Warning

⚠️ For all write operations (Twitter posts, Weibo updates, likes, replies):

- Content is **immediately public** once posted — the AI cannot retract it
- Rapid repeated posting may trigger **rate limits or account restrictions**
- Always **preview and confirm** content before Claude executes write commands
- Avoid automation that mimics rapid human-like posting patterns

## Credits

Built on **[jackwener/opencli](https://github.com/jackwener/opencli)** by [@jackwener](https://github.com/jackwener). The core innovation: turning major websites into CLI interfaces by reusing existing browser sessions.
