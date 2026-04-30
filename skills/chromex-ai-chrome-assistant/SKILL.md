```markdown
---
name: chromex-ai-chrome-assistant
description: Chromex is a Chrome MV3 side-panel assistant that connects Chrome to OpenAI Codex through a local native bridge, enabling page context, tab, voice, image, and browser-control workflows.
triggers:
  - set up chromex chrome extension
  - add chromex side panel assistant
  - connect codex to chrome extension
  - build chrome side panel with codex
  - chromex native bridge setup
  - install chromex local bridge
  - chromex page context workflow
  - chromex voice image tab assistant
---

# Chromex AI Chrome Assistant

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Chromex is a Chrome MV3 side-panel extension that bridges the browser to OpenAI Codex through a local native messaging host. It supports chat with page content, selected tabs, uploaded files (PDF, DOCX, XLSX, images), voice transcription, live voice mode, screenshot workflows, and browser-control via content scripts — all without storing credentials in extension storage.

---

## Architecture

```
Chrome Extension (MV3 Side Panel)
  └─> Native Messaging Host  (packages/native-host)
        └─> Local Bridge Daemon  (packages/bridge)
              └─> codex app-server  (OpenAI Codex CLI)
```

**Packages:**

| Package | Purpose |
|---|---|
| `packages/extension` | Chrome MV3 side-panel UI (TypeScript + React) |
| `packages/bridge` | Local bridge daemon: Codex app-server, multimodal workflows |
| `packages/native-host` | Chrome Native Messaging relay process |
| `packages/shared` | Shared types, policies, profiles, helpers |

---

## Prerequisites

- Node.js 20 LTS or newer
- Google Chrome (stable or dev)
- OpenAI Codex CLI: `npm install -g @openai/codex`
- Verify: `codex --version`

---

## Install & Build

### Clone and Build

```bash
git clone https://github.com/GENEXIS-AI/chromex.git
cd chromex
npm install
npm run build
```

### Install the Native Messaging Host

```bash
# macOS / Linux
node scripts/install-native-host.mjs

# Windows (PowerShell)
node scripts/install-native-host.mjs --browser=chrome
```

If Chrome assigns a non-standard extension ID, pass it explicitly:

```bash
node scripts/install-native-host.mjs <extension-id> --browser=chrome
```

The expected public release extension ID is `menmlhahmendmkiicbjihgjhppkgaeom`.

### Load Extension in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select `packages/extension/dist`

---

## Key CLI / NPM Commands

```bash
npm run build              # Build all packages
npm run typecheck          # TypeScript type-check all packages
npm run test               # Run unit tests
npm run release:audit      # Pre-release audit checks
npm run smoke              # Browser smoke test (Playwright)
npm run smoke:install-browser  # Install Playwright Chromium runtime
```

---

## Configuration

### Windows: Codex Binary Path

If `codex` is not found on PATH, set the executable path in Chromex Settings:

- **Codex binary path:** `%APPDATA%\npm\codex.cmd`
- **Codex binary folder:** `%APPDATA%\npm`

> Do not put your workspace folder in the Codex binary field. The workspace folder and executable path are separate settings.

### App UI Language

Chromex auto-detects the browser language. To override:
**Settings → General → App UI language**

Supported locales include: `en`, `ko`, `ja`, `zh-CN`, `ar`, `fr`, `de`, `es`, `pt`, `hi`, `vi`, `th`, `tr`, `uk`, and more.

### Chat History

Conversation history is **session-only by default**. Enable persistent local chat history in:
**Settings → Privacy → Persistent local chat history**

---

## Extension Package Structure

```
packages/extension/
├── src/
│   ├── background/          # Service worker, native messaging client
│   ├── sidepanel/           # React side-panel UI
│   │   ├── components/      # Chat, attachments, pickers, markdown
│   │   ├── hooks/           # Page context, voice, tab selection
│   │   └── pages/           # Main panel, onboarding, settings
│   ├── content/             # Content scripts for page context & browser control
│   └── shared/              # Extension-local shared utilities
├── manifest.json
└── dist/                    # Built output (load this in Chrome)
```

---

## Real Code Examples

### 1. Sending a Message to the Bridge (Native Messaging)

```typescript
// packages/extension/src/background/nativeClient.ts

interface BridgeMessage {
  type: string;
  payload: unknown;
  requestId: string;
}

class NativeClient {
  private port: chrome.runtime.Port | null = null;

  connect(): void {
    this.port = chrome.runtime.connectNative("com.genexisai.chromex.bridge");
    this.port.onMessage.addListener((msg) => this.handleMessage(msg));
    this.port.onDisconnect.addListener(() => {
      this.port = null;
      console.warn("Native host disconnected", chrome.runtime.lastError);
    });
  }

  send(message: BridgeMessage): void {
    if (!this.port) this.connect();
    this.port!.postMessage(message);
  }

  private handleMessage(msg: unknown): void {
    // Dispatch to side panel via chrome.runtime.sendMessage
    chrome.runtime.sendMessage({ source: "native", data: msg });
  }
}

export const nativeClient = new NativeClient();
```

### 2. Reading Page Context from a Content Script

```typescript
// packages/extension/src/content/pageContext.ts

export interface PageContext {
  url: string;
  title: string;
  textContent: string;
  selectedText: string;
  metaDescription: string;
}

export function extractPageContext(): PageContext {
  const selectedText = window.getSelection()?.toString() ?? "";

  return {
    url: location.href,
    title: document.title,
    textContent: document.body.innerText.slice(0, 20000), // truncate for context window
    selectedText,
    metaDescription:
      document
        .querySelector('meta[name="description"]')
        ?.getAttribute("content") ?? "",
  };
}

// Listen for context requests from the background service worker
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "GET_PAGE_CONTEXT") {
    sendResponse(extractPageContext());
  }
  return true; // keep channel open for async
});
```

### 3. Requesting Page Context from the Background

```typescript
// packages/extension/src/background/contextBridge.ts

export async function getActiveTabContext(): Promise<PageContext | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return null;

  try {
    const context = await chrome.tabs.sendMessage(tab.id, {
      type: "GET_PAGE_CONTEXT",
    });
    return context as PageContext;
  } catch {
    // Content script not injected yet — inject it
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content/pageContext.js"],
    });
    return chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_CONTEXT" });
  }
}
```

### 4. Tab Picker — Selecting Multiple Tabs

```typescript
// packages/extension/src/sidepanel/hooks/useTabPicker.ts
import { useState, useEffect } from "react";

export interface TabItem {
  id: number;
  title: string;
  url: string;
  favIconUrl?: string;
}

export function useTabPicker() {
  const [tabs, setTabs] = useState<TabItem[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());

  useEffect(() => {
    chrome.tabs.query({ currentWindow: true }, (chromeTabs) => {
      setTabs(
        chromeTabs
          .filter((t) => t.id !== undefined)
          .map((t) => ({
            id: t.id!,
            title: t.title ?? "",
            url: t.url ?? "",
            favIconUrl: t.favIconUrl,
          }))
      );
    });
  }, []);

  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return { tabs, selected, toggle };
}
```

### 5. Bridge Daemon — Starting the Codex App-Server

```typescript
// packages/bridge/src/codexServer.ts
import { spawn, ChildProcess } from "child_process";
import path from "path";

export interface CodexServerConfig {
  codexBin: string;       // e.g. "codex" or full path
  workspaceDir: string;   // user workspace directory
  port: number;
}

export class CodexAppServer {
  private proc: ChildProcess | null = null;

  start(config: CodexServerConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      this.proc = spawn(
        config.codexBin,
        ["app-server", "--port", String(config.port)],
        {
          cwd: config.workspaceDir,
          env: {
            // Reduced environment allowlist — never pass full process.env
            PATH: process.env.PATH ?? "",
            HOME: process.env.HOME ?? "",
            USERPROFILE: process.env.USERPROFILE ?? "",
            OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "", // optional local fallback
          },
          stdio: ["pipe", "pipe", "pipe"],
        }
      );

      this.proc.stdout?.once("data", () => resolve());
      this.proc.on("error", reject);
      this.proc.stderr?.on("data", (d) =>
        console.error("[codex-server]", d.toString())
      );
    });
  }

  stop(): void {
    this.proc?.kill();
    this.proc = null;
  }
}
```

### 6. Shared Types — Chat Message

```typescript
// packages/shared/src/types/chat.ts

export type Role = "user" | "assistant" | "system";

export type AttachmentType =
  | "image"
  | "pdf"
  | "docx"
  | "xlsx"
  | "csv"
  | "text"
  | "screenshot";

export interface Attachment {
  type: AttachmentType;
  name: string;
  dataUrl?: string;      // base64 data URI for images/screenshots
  textContent?: string;  // extracted text for documents
  mimeType: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  attachments?: Attachment[];
  timestamp: number;
  pageContext?: {
    url: string;
    title: string;
  };
}
```

### 7. Profile Selection with `/` Picker

```typescript
// packages/shared/src/profiles/types.ts

export interface CodexProfile {
  id: string;
  name: string;
  systemPrompt: string;
  model: string;
  temperature?: number;
  skillsEnabled: boolean;
  skillPaths?: string[]; // paths to .codex/skills/*/SKILL.md
}

// packages/extension/src/sidepanel/hooks/useProfilePicker.ts
import { useState } from "react";
import type { CodexProfile } from "@chromex/shared/profiles/types";

export function useProfilePicker(profiles: CodexProfile[]) {
  const [active, setActive] = useState<CodexProfile>(profiles[0]);
  const [open, setOpen] = useState(false);

  const select = (id: string) => {
    const found = profiles.find((p) => p.id === id);
    if (found) setActive(found);
    setOpen(false);
  };

  return { active, open, setOpen, select };
}
```

### 8. Native Host Manifest (auto-generated by install script)

```json
{
  "name": "com.genexisai.chromex.bridge",
  "description": "Chromex local bridge native messaging host",
  "path": "/absolute/path/to/packages/native-host/dist/host.js",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://menmlhahmendmkiicbjihgjhppkgaeom/"
  ]
}
```

On **macOS/Linux**, this file is placed at:
- `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/` (macOS)
- `~/.config/google-chrome/NativeMessagingHosts/` (Linux)

On **Windows**, the path is written to the registry:
`HKCU\Software\Google\Chrome\NativeMessagingHosts\com.genexisai.chromex.bridge`

---

## Common Patterns

### Pattern: Page-Aware Chat Request

```typescript
// In the side-panel, combine page context with user message
async function sendPageAwareMessage(userText: string) {
  const context = await chrome.runtime.sendMessage({ type: "GET_PAGE_CONTEXT" });

  const payload = {
    type: "CHAT",
    messages: [
      {
        role: "system",
        content: `Current page: ${context.title} (${context.url})\n\n${context.textContent}`,
      },
      { role: "user", content: userText },
    ],
  };

  chrome.runtime.sendMessage({ type: "BRIDGE_SEND", payload });
}
```

### Pattern: Screenshot Attachment

```typescript
async function captureAndAttach(): Promise<Attachment> {
  const dataUrl = await chrome.tabs.captureVisibleTab({ format: "png" });
  return {
    type: "screenshot",
    name: "screenshot.png",
    dataUrl,
    mimeType: "image/png",
  };
}
```

### Pattern: YouTube Adapter — Timestamp Context

```typescript
// packages/extension/src/content/adapters/youtube.ts
export function getYouTubeContext() {
  const video = document.querySelector<HTMLVideoElement>("video");
  const title =
    document.querySelector("h1.ytd-video-primary-info-renderer")?.textContent ??
    document.title;

  return {
    site: "youtube",
    videoTitle: title.trim(),
    currentTime: video ? Math.floor(video.currentTime) : null,
    duration: video ? Math.floor(video.duration) : null,
    url: location.href,
  };
}

export function seekTo(seconds: number): void {
  const video = document.querySelector<HTMLVideoElement>("video");
  if (video) video.currentTime = seconds;
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **Native host missing or forbidden** | Run `npm run build`, then `node scripts/install-native-host.mjs --browser=chrome`, reload extension in `chrome://extensions` |
| **Different extension ID** | Run `node scripts/install-native-host.mjs <your-extension-id> --browser=chrome` |
| **`Failed to start codex app-server`** | Run `codex --version`. On Windows, set Codex binary path to `%APPDATA%\npm\codex.cmd` in Chromex Settings |
| **Model list does not load** | Confirm native bridge is connected, then sign in through app-server login flow |
| **Page context unavailable** | Open Chromex from the target tab, approve the Chrome site permission prompt |
| **Old UI still showing** | Run `npm run build`, reload the extension card in `chrome://extensions` |
| **Smoke test fails: no browser** | Run `npm run smoke:install-browser`, then `npm run smoke` |
| **Windows: codex not found on PATH** | Set Codex binary folder to `%APPDATA%\npm` in Chromex Settings |

---

## Security Notes

- **Never** store raw `OPENAI_API_KEY` in Chrome extension storage or `manifest.json`.
- Credentials flow through the local Codex app-server OAuth/login, not extension storage.
- API-key login is an optional local fallback only, never automatic.
- Native-host child processes run with a **reduced environment allowlist** (see `CodexAppServer` example above).
- Page content, microphone input, and history are only accessed for user-requested workflows.
- Read [`SECURITY.md`](https://github.com/GENEXIS-AI/chromex/blob/main/SECURITY.md) and [`PRIVACY.md`](https://github.com/GENEXIS-AI/chromex/blob/main/PRIVACY.md) before deploying modified builds.

---

## Release & Contributing

- Versioning and PR flow: [`RELEASE.md`](https://github.com/GENEXIS-AI/chromex/blob/main/RELEASE.md)
- Pre-release check: `npm run release:audit`
- License: MIT
- Built extension output: `packages/extension/dist`
```
