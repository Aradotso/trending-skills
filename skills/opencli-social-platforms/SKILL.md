```markdown
---
name: opencli-social-platforms
description: Use opencli CLI to control 16 social/content platforms (Bilibili, Twitter/X, YouTube, Zhihu, Reddit, HackerNews, Weibo, etc.) via natural language, reusing Chrome login sessions with no API keys required.
triggers:
  - search YouTube for videos
  - get trending on Twitter or Weibo
  - check Bilibili hot list
  - search Reddit or HackerNews
  - post a tweet or weibo
  - check stock price on Yahoo Finance or Xueqiu
  - search Zhihu or Xiaohongshu
  - browse social media platforms with Claude
---

# opencli-skill: Control 16 Social Platforms via CLI

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

opencli is a CLI tool that turns 16 major platforms into command-line interfaces by **reusing your existing Chrome login sessions**. No API keys, no re-authentication — just open Chrome, log in as usual, and interact with Bilibili, Twitter/X, YouTube, Zhihu, Reddit, HackerNews, Weibo, Xiaohongshu, V2EX, Xueqiu, BOSS直聘, BBC, Reuters, 什么值得买, Yahoo Finance, and Ctrip.

---

## Prerequisites

Before using opencli, ensure all of the following are in place:

1. **Node.js v16+** — [nodejs.org](https://nodejs.org/)
2. **Chrome browser** open and logged in to target platforms
3. **Playwright MCP Bridge** Chrome extension — [Install from Chrome Web Store](https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn)
4. **Playwright MCP** configured in Claude Code
5. **Claude Code** — [claude.ai/code](https://claude.ai/code)

---

## Installation

### Step 1 — Install opencli globally

```bash
npm install -g @jackwener/opencli
```

Verify:
```bash
opencli --version
```

### Step 2 — Install Playwright MCP Bridge in Chrome

Install the [Playwright MCP Bridge](https://chromewebstore.google.com/detail/playwright-mcp-bridge/kldoghpdblpjbjeechcaoibpfbgfomkn) extension from the Chrome Web Store. This bridges opencli to your existing Chrome sessions so login state is reused automatically.

### Step 3 — Configure Playwright MCP in Claude Code

```bash
claude mcp add playwright --scope user -- npx @playwright/mcp@latest
```

Verify:
```bash
claude mcp list
# Should show "playwright" in the list
```

### Step 4 — Install this skill

```bash
npx skills add joeseesun/opencli-skill
```

Restart Claude Code to activate.

---

## Supported Platforms & Capabilities

| Platform | Read | Search | Write |
|----------|------|--------|-------|
| Bilibili (B站) | ✅ Hot/Ranking/Feed/History | ✅ Videos/Users | — |
| Zhihu (知乎) | ✅ Hot list | ✅ | ✅ Question details |
| Weibo (微博) | ✅ Trending | — | ✅ Post (Playwright) |
| Twitter/X | ✅ Timeline/Trending/Bookmarks | ✅ | ✅ Post/Reply/Like |
| YouTube | — | ✅ | — |
| Xiaohongshu (小红书) | ✅ Recommended feed | ✅ | — |
| Reddit | ✅ Home/Hot | ✅ | — |
| HackerNews | ✅ Top stories | — | — |
| V2EX | ✅ Hot/Latest | — | ✅ Daily check-in |
| Xueqiu (雪球) | ✅ Hot/Stocks/Watchlist | ✅ | — |
| BOSS直聘 | — | ✅ Jobs | — |
| BBC | ✅ News | — | — |
| Reuters | — | ✅ | — |
| 什么值得买 | — | ✅ Deals | — |
| Yahoo Finance | ✅ Stock quotes | — | — |
| Ctrip (携程) | — | ✅ Attractions/Cities | — |

---

## Key Commands

### Bilibili (B站)

```bash
# Hot videos
opencli bilibili hot --limit 10 -f json

# Ranking
opencli bilibili ranking -f json

# Search videos
opencli bilibili search --keyword "AI大模型" -f json

# Search users
opencli bilibili search-user --keyword "林超" -f json

# Personal feed
opencli bilibili feed -f json

# Watch history
opencli bilibili history --limit 20 -f json
```

### Twitter / X

```bash
# Timeline
opencli twitter timeline -f json

# Trending topics
opencli twitter trending -f json

# Search tweets
opencli twitter search --query "claude AI" -f json

# Bookmarks
opencli twitter bookmarks -f json

# Post a tweet (requires confirmation before executing)
opencli twitter post --text "Hello from Claude Code!"

# Reply to a tweet
opencli twitter reply --tweet-id 1234567890 --text "Great post!"

# Like a tweet
opencli twitter like --tweet-id 1234567890
```

### YouTube

```bash
# Search videos
opencli youtube search --query "LLM tutorial" -f json
opencli youtube search --query "React hooks" --limit 10 -f json
```

### Zhihu (知乎)

```bash
# Hot list
opencli zhihu hot -f json

# Search
opencli zhihu search --keyword "大模型应用" -f json

# Question details
opencli zhihu question --id 123456789 -f json
```

### Weibo (微博)

```bash
# Trending / Hot search
opencli weibo hot -f json

# Post a weibo (requires confirmation)
opencli weibo post --text "今天天气真好"
```

### Reddit

```bash
# Home feed
opencli reddit home -f json

# Hot posts in a subreddit
opencli reddit hot --subreddit MachineLearning -f json
opencli reddit hot --subreddit programming --limit 20 -f json

# Search
opencli reddit search --query "transformer architecture" -f json
```

### HackerNews

```bash
# Top stories
opencli hackernews top --limit 20 -f json

# New stories
opencli hackernews new --limit 10 -f json
```

### Xueqiu (雪球) — Stocks

```bash
# Stock quote
opencli xueqiu stock --symbol SH600519    # 茅台
opencli xueqiu stock --symbol AAPL        # Apple

# Hot stocks
opencli xueqiu hot -f json

# Your watchlist
opencli xueqiu watchlist -f json

# Search stocks
opencli xueqiu search --keyword "比亚迪" -f json
```

### Yahoo Finance

```bash
# Stock quote
opencli yahoo-finance quote --symbol AAPL -f json
opencli yahoo-finance quote --symbol TSLA -f json
opencli yahoo-finance quote --symbol BTC-USD -f json
```

### Xiaohongshu (小红书)

```bash
# Recommended feed
opencli xiaohongshu feed -f json

# Search notes
opencli xiaohongshu search --keyword "咖啡推荐" -f json
```

### V2EX

```bash
# Hot topics
opencli v2ex hot -f json

# Latest topics
opencli v2ex latest -f json

# Daily check-in
opencli v2ex checkin
```

### BOSS直聘

```bash
# Search jobs
opencli boss search --keyword "前端工程师" --city "上海" -f json
opencli boss search --keyword "AI engineer" -f json
```

### BBC News

```bash
# Latest news
opencli bbc news -f json
```

### Reuters

```bash
# Search articles
opencli reuters search --query "artificial intelligence" -f json
```

### 什么值得买

```bash
# Search deals
opencli smzdm search --keyword "机械键盘" -f json
```

### Ctrip (携程)

```bash
# Search attractions
opencli ctrip attractions --city "成都" -f json

# Search cities
opencli ctrip cities --keyword "云南" -f json
```

---

## Common Output Format

All commands support `-f json` for structured JSON output. Without it, output is human-readable plain text.

```bash
# JSON output (recommended for Claude to parse)
opencli bilibili hot --limit 5 -f json

# Example JSON response structure
{
  "data": [
    {
      "title": "...",
      "author": "...",
      "views": 1234567,
      "url": "https://..."
    }
  ]
}
```

---

## Usage Patterns

### Pattern 1: Read and Summarize

When a user asks "What's trending on Bilibili?":

```bash
opencli bilibili hot --limit 20 -f json
```

Parse the JSON, format as a numbered table with titles, view counts, and authors.

### Pattern 2: Cross-Platform Search

When a user asks "Search for AI news across platforms":

```bash
opencli hackernews top --limit 10 -f json
opencli reddit hot --subreddit artificial -f json
opencli zhihu search --keyword "AI" -f json
```

Aggregate results and present unified findings.

### Pattern 3: Stock Research

When a user asks "Check Apple stock":

```bash
opencli yahoo-finance quote --symbol AAPL -f json
opencli xueqiu search --keyword "AAPL" -f json
```

### Pattern 4: Write Operations (Always Confirm First)

⚠️ **Always show the user exactly what will be posted and wait for explicit confirmation before executing any write command.**

```
User: "Post a tweet saying I just finished my project"

Claude: I'll post the following tweet:
"Just finished my project! 🎉"

Shall I proceed? (yes/no)

[After confirmation:]
opencli twitter post --text "Just finished my project! 🎉"
```

---

## How opencli Works

opencli controls Chrome via the **Playwright MCP Bridge** extension to reuse your existing authenticated sessions. This means:

- ✅ No API keys or OAuth setup required
- ✅ Works with platforms that don't have public APIs
- ✅ Uses your actual account (personalized feed, watchlist, etc.)
- ✅ Handles rate limiting by behaving like a real browser user

The flow: `Claude Code` → `Playwright MCP` → `Playwright MCP Bridge (Chrome Extension)` → `Chrome (logged-in sessions)` → `Platform websites`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `opencli: command not found` | Re-run `npm install -g @jackwener/opencli`; verify PATH includes npm global bin |
| Chrome not being controlled | Ensure Chrome is open; verify Playwright MCP Bridge extension is enabled |
| Login state not recognized | Manually log in to the target site in Chrome first, then retry |
| `Playwright MCP not found` error | Re-run: `claude mcp add playwright --scope user -- npx @playwright/mcp@latest` |
| `npx skills add` fails | Ensure Node.js v16+ is installed: `node --version` |
| JSON parse errors | Some platforms may return HTML on rate-limit; wait and retry |
| CAPTCHA triggered | Manual intervention needed in Chrome; avoid rapid repeated commands |

### Verify Full Setup

```bash
# 1. Check opencli is installed
opencli --version

# 2. Check Playwright MCP is configured
claude mcp list

# 3. Test a read command (Chrome must be open)
opencli hackernews top --limit 3 -f json
```

---

## Full Command Reference

All 55+ commands are documented at [references/commands.md](https://github.com/joeseesun/opencli-skill/blob/main/references/commands.md) in the skill repository.

---

## Write Operations Warning

Write operations include: posting tweets, weibo updates, V2EX check-ins, Twitter likes/replies.

Risks to communicate to users:
- Platform bot-detection may trigger CAPTCHAs or temporary rate limits
- Content is immediately public once posted; the AI cannot retract it
- Avoid rapid repeated posting (risk of account suspension)
- **Always confirm with the user before executing any write command**
```
