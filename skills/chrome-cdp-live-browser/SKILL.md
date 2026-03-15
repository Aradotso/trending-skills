---
name: chrome-cdp-live-browser
description: Connect AI agents to your live Chrome session via CDP for real-time tab interaction, screenshots, and JS evaluation without re-login
triggers:
  - browse the web in chrome
  - interact with my open browser tabs
  - take a screenshot of the current page
  - click elements on a webpage
  - evaluate javascript in the browser
  - read page content from chrome
  - automate my existing chrome session
  - navigate to a url in chrome
---

# Chrome CDP Live Browser Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection

Connect your AI agent to a **live, already-running Chrome session** — no fresh browser, no re-login, no automation framework. This skill uses Chrome DevTools Protocol (CDP) directly to interact with tabs you have open right now.

## What It Does

- **Reads and interacts with authenticated pages** (Gmail, GitHub, internal tools) without re-authenticating
- **Takes screenshots** of live tab state
- **Evaluates JavaScript** in page context
- **Clicks elements**, types text, navigates URLs
- **Extracts accessibility trees and HTML** for structured page analysis
- Maintains **persistent daemon per tab** — Chrome's "Allow debugging" prompt fires once, then stays silent

## Prerequisites

- Node.js 22+
- Chrome, Chromium, Brave, Edge, or Vivaldi
- Remote debugging enabled in Chrome

## Installation

### Enable Remote Debugging in Chrome

Navigate to `chrome://inspect/#remote-debugging` and toggle the switch. That's it — no flags, no restart needed.

### Install as a pi skill

```bash
pi install git:github.com/pasky/chrome-cdp-skill@v1.0.1
```

### For other agents (Claude Code, Cursor, Amp, etc.)

Clone the repo and use the `skills/chrome-cdp/` directory:

```bash
git clone https://github.com/pasky/chrome-cdp-skill.git
# Point your agent's skill loader at skills/chrome-cdp/
```

No `npm install` needed — zero runtime dependencies beyond Node.js 22+.

## Key Commands

```bash
# List all open tabs with their targetIds
scripts/cdp.mjs list

# Take a screenshot (saves to runtime dir)
scripts/cdp.mjs shot <target>

# Get accessibility tree (compact, semantic — best for understanding page structure)
scripts/cdp.mjs snap <target>

# Get full HTML or scoped to a CSS selector
scripts/cdp.mjs html <target>
scripts/cdp.mjs html <target> ".main-content"

# Evaluate JavaScript in the page
scripts/cdp.mjs eval <target> "document.title"
scripts/cdp.mjs eval <target> "window.location.href"

# Navigate to a URL and wait for load
scripts/cdp.mjs nav <target> https://example.com

# Click an element by CSS selector
scripts/cdp.mjs click <target> "button.submit"

# Click at specific pixel coordinates
scripts/cdp.mjs clickxy <target> 320 480

# Type text at the currently focused element (works in cross-origin iframes)
scripts/cdp.mjs type <target> "hello world"

# Get network resource timing
scripts/cdp.mjs net <target>

# Click "load more" repeatedly until selector disappears
scripts/cdp.mjs loadall <target> ".load-more-btn"

# Raw CDP command passthrough
scripts/cdp.mjs evalraw <target> Runtime.evaluate '{"expression":"1+1"}'

# Open a new tab (triggers Allow prompt)
scripts/cdp.mjs open https://example.com

# Stop daemon for a specific tab, or all daemons
scripts/cdp.mjs stop <target>
scripts/cdp.mjs stop
```

`<target>` is a unique prefix of the `targetId` shown by `list`.

## Typical Agent Workflow

### 1. Discover open tabs

```bash
scripts/cdp.mjs list
# Output:
# abc123  https://github.com/user/repo  GitHub - user/repo
# def456  https://mail.google.com       Gmail
# ghi789  https://app.internal.co       Internal Dashboard
```

### 2. Inspect page structure

```bash
# Accessibility tree is fastest for understanding layout
scripts/cdp.mjs snap abc123

# Scoped HTML for a specific component
scripts/cdp.mjs html abc123 "#issue-list"
```

### 3. Extract data with JavaScript

```javascript
// Using eval to collect structured data from a live page
scripts/cdp.mjs eval abc123 "
  JSON.stringify(
    Array.from(document.querySelectorAll('.issue-title'))
      .map(el => ({ title: el.textContent.trim(), href: el.closest('a')?.href }))
  )
"
```

### 4. Interact with the page

```bash
# Fill a search box and submit
scripts/cdp.mjs click abc123 "input[name='q']"
scripts/cdp.mjs type  abc123 "my search query"
scripts/cdp.mjs click abc123 "button[type='submit']"

# Wait for navigation and check result
scripts/cdp.mjs snap abc123
```

### 5. Screenshot for visual verification

```bash
scripts/cdp.mjs shot abc123
# Saves PNG to runtime dir, path printed to stdout
```

## Configuration

### Non-standard browser profile location

If Chrome stores `DevToolsActivePort` in a non-default path:

```bash
export CDP_PORT_FILE="/path/to/your/chrome/profile/DevToolsActivePort"
scripts/cdp.mjs list
```

Default search locations (auto-detected):
- macOS: `~/Library/Application Support/Google/Chrome/`
- Linux: `~/.config/google-chrome/`, `~/.config/chromium/`, `~/.config/brave-browser/`
- Also checks Vivaldi, Edge, Canary variants

### Daemon lifecycle

Daemons auto-exit after **20 minutes of inactivity**. You can manually stop them:

```bash
scripts/cdp.mjs stop          # stop all
scripts/cdp.mjs stop abc123   # stop one tab's daemon
```

## Integration Patterns

### Reading a logged-in page (e.g. GitHub notifications)

```bash
# No login needed — uses your existing session
TARGET=$(scripts/cdp.mjs list | grep github.com | awk '{print $1}' | head -1)
scripts/cdp.mjs eval $TARGET "
  JSON.stringify(
    Array.from(document.querySelectorAll('.notification-list-item'))
      .slice(0, 10)
      .map(n => n.querySelector('a')?.textContent?.trim())
  )
"
```

### Scraping a page that requires infinite scroll

```bash
# Click "load more" until the button disappears, then extract all content
scripts/cdp.mjs loadall abc123 "button.load-more"
scripts/cdp.mjs html abc123 ".results-container"
```

### Running a multi-step form automation

```bash
TARGET=abc123

# Fill form fields
scripts/cdp.mjs click  $TARGET "#first-name"
scripts/cdp.mjs type   $TARGET "Jane"
scripts/cdp.mjs click  $TARGET "#last-name"
scripts/cdp.mjs type   $TARGET "Smith"

# Select a dropdown via JS
scripts/cdp.mjs eval   $TARGET "document.querySelector('#country').value = 'US'"

# Submit
scripts/cdp.mjs click  $TARGET "button[type='submit']"

# Verify result
scripts/cdp.mjs snap   $TARGET
```

### Raw CDP for advanced use cases

```bash
# Capture full page PDF
scripts/cdp.mjs evalraw abc123 Page.printToPDF '{}'

# Get all cookies for the page
scripts/cdp.mjs evalraw abc123 Network.getCookies '{}'

# Emulate mobile viewport
scripts/cdp.mjs evalraw abc123 Emulation.setDeviceMetricsOverride \
  '{"width":375,"height":812,"deviceScaleFactor":3,"mobile":true}'
```

## Troubleshooting

### "Cannot connect to Chrome" / no output from `list`

1. Confirm remote debugging is enabled: visit `chrome://inspect/#remote-debugging`
2. Check that `DevToolsActivePort` exists:
   ```bash
   ls ~/Library/Application\ Support/Google/Chrome/DevToolsActivePort   # macOS
   ls ~/.config/google-chrome/DevToolsActivePort                        # Linux
   ```
3. Set `CDP_PORT_FILE` explicitly if using a non-standard profile

### "Allow debugging" prompt keeps appearing

This prompt fires **once per tab per daemon start**. After the first command to a tab, the daemon persists and subsequent commands skip the prompt. If it keeps appearing, the daemon may be crashing — check for Node.js version (`node --version` should be 22+).

### Timeout with many tabs open

Unlike tools that re-enumerate all targets on every command, `chrome-cdp` targets tabs by prefix ID directly. If you see slowness, use a more specific target prefix to avoid ambiguity.

### Commands work but screenshot is blank

The page may not have finished rendering. Chain an `eval` to wait:

```bash
scripts/cdp.mjs eval abc123 "
  new Promise(r => {
    if (document.readyState === 'complete') r();
    else window.addEventListener('load', r);
  })
"
scripts/cdp.mjs shot abc123
```

### Type command not working in iframes

The `type` command is specifically designed to work across **cross-origin iframes** — make sure you're using `type` (not a click+eval workaround) for iframe inputs.

## How It Works Internally

1. Reads `DevToolsActivePort` to find Chrome's debugging WebSocket port
2. On first access to a `<target>`, spawns a lightweight Node.js daemon that holds the WebSocket session open
3. Commands communicate with the daemon via a local socket
4. Daemons auto-clean after 20 minutes idle
5. No Puppeteer, no intermediary — raw CDP messages only

This architecture is why it handles 100+ open tabs reliably: target enumeration only touches the specific tab you address, not all tabs at once.
