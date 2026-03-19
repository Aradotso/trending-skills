```markdown
---
name: collaborator-ai-canvas
description: Skill for using Collaborator, a native macOS desktop app for agentic development on an infinite canvas with terminals, files, and AI agents.
triggers:
  - "set up collaborator for agentic development"
  - "install collaborator on mac"
  - "use collaborator with AI agents"
  - "open workspace in collaborator"
  - "create terminal tile on canvas"
  - "drag files onto collaborator canvas"
  - "configure collaborator workspace"
  - "run coding agent in collaborator"
---

# Collaborator AI Canvas

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Collaborator is a native macOS desktop application (Apple Silicon only) that provides an infinite canvas where you can arrange terminal sessions, markdown notes, code files, and images side-by-side. It is purpose-built for running AI coding agents (Claude Code, Cursor, Codex, etc.) without context switching.

---

## Installation

### One-line install (recommended)

```sh
curl -fsSL https://raw.githubusercontent.com/collaborator-ai/collab-public/main/install.sh | bash
```

### Manual download

Download the latest `.dmg` from:
```
https://github.com/collaborator-ai/collab-public/releases/latest
```

> **Requirements:** macOS, Apple Silicon (arm64). Intel Macs are not supported.

---

## First Launch Quickstart

1. Open Collaborator from Applications or Spotlight.
2. Add a workspace:
   - Click the workspace dropdown in the top-left navigator sidebar.
   - Choose **"Add workspace"** and select a local project folder.
   - Keyboard shortcut: `Cmd+Shift+O`
3. Double-click empty canvas space → creates a **terminal tile**.
4. Your terminal opens in the workspace directory. Start an agent:

```sh
# Example: start Claude Code in your workspace
claude

# Example: start Codex
codex

# Example: run any shell command
npm run dev
```

5. Drag a file from the navigator sidebar onto the canvas to open it as a tile next to your terminal.

---

## Application Layout

```
┌──────────────────────────────────────────────────┐
│  Navigator (sidebar)  │  Canvas (infinite)        │
│  ┌─────────────────┐  │  ┌──────┐  ┌──────────┐  │
│  │ Workspace        │  │  │ term │  │ note.md  │  │
│  │ dropdown         │  │  └──────┘  └──────────┘  │
│  ├─────────────────┤  │                            │
│  │ File tree        │  │  ┌──────────┐             │
│  │ (expand/collapse)│  │  │ code.ts  │             │
│  └─────────────────┘  │  └──────────┘             │
└──────────────────────────────────────────────────┘
```

- **Navigator** — resizable left sidebar with file tree and workspace switcher.
- **Canvas** — infinite pan/zoom surface; holds all tiles.
- **Viewer** — opens when you click a file in the navigator; occupies the main area alongside the canvas.

---

## Canvas Controls

| Action | Input |
|---|---|
| Pan canvas | Scroll wheel, Space+drag, or middle-click+drag |
| Zoom in | `Cmd+=` or `Ctrl+scroll up` |
| Zoom out | `Cmd+-` or `Ctrl+scroll down` |
| Reset zoom | `Cmd+0` |
| Create terminal | Double-click empty canvas space |
| Open file as tile | Drag file from navigator onto canvas |
| Close viewer | `Escape` |
| Search files | `Cmd+K` |
| Add workspace | `Cmd+Shift+O` |
| Rename file | `F2` |

- Zoom range: 33%–100% with rubber-band overshoot.
- All tile positions snap to a dot grid.
- Scroll through a tile without panning: hold `Shift` while scrolling.

---

## Tile Types

### Terminal tile
- Created by **double-clicking** empty canvas space.
- Backed by a **tmux session** — persists even if you close and reopen the tile.
- Working directory is set to the active workspace path automatically.
- Use this to run agents, build tools, or any shell commands.

### Note tile (Markdown)
- Created by dragging a `.md`, `.mdx`, `.markdown`, or `.txt` file onto the canvas.
- Rich text editor with live rendering, frontmatter support, cover images, and wiki-style `[[links]]`.

### Code tile
- Created by dragging any non-markdown, non-image file onto the canvas.
- Monaco Editor with syntax highlighting and language auto-detection.
- Supports inline editing.

### Image tile
- Created by dragging `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, or `.webp` onto the canvas.
- Read-only display with metadata.

---

## Workspace Management

```
Workspace dropdown (top of navigator)
├── Switch to any workspace (click to activate)
├── Add workspace      → Cmd+Shift+O → pick a folder
└── Remove workspace   → removes from list, does NOT delete files
```

- Each workspace has its own independent file tree.
- The canvas and viewer are **shared** across workspaces.
- Workspaces are stored by absolute path in `~/.collaborator/config.json`.

---

## File Tree Operations

| Action | How |
|---|---|
| Expand/collapse folder | Click folder |
| Open file in viewer | Click file |
| Open file as canvas tile | Drag file onto canvas |
| Create new note | Navigator toolbar → new note icon (creates `Untitled.md`) |
| Create new folder | Navigator toolbar → new folder icon |
| Rename | Select file → `F2` |
| Delete | Select file → Delete key (moves to Trash) |
| Move file | Drag between folders |
| Multi-select | `Shift+click` (range), `Cmd+click` (individual) |
| Search | `Cmd+K` |

**View modes:**
- Hierarchical tree
- Chronological feed (sorted by date)

**Sort options (cycle through):**
- Created: newest / oldest
- Modified: newest / oldest
- Name: A–Z / Z–A

---

## Persistent State Files

All state is stored in `~/.collaborator/`. You can inspect or back these up manually.

### `~/.collaborator/canvas-state.json`

```json
{
  "version": 1,
  "tiles": [
    {
      "id": "tile-1742024800000-0",
      "type": "term",
      "x": 120,
      "y": 80,
      "width": 440,
      "height": 540,
      "filePath": null,
      "zIndex": 1
    },
    {
      "id": "tile-1742024900000-1",
      "type": "note",
      "x": 600,
      "y": 80,
      "width": 440,
      "height": 540,
      "filePath": "/Users/you/project/README.md",
      "zIndex": 2
    },
    {
      "id": "tile-1742025000000-2",
      "type": "code",
      "x": 1080,
      "y": 80,
      "width": 600,
      "height": 540,
      "filePath": "/Users/you/project/src/index.ts",
      "zIndex": 3
    }
  ],
  "viewport": {
    "panX": -60,
    "panY": -20,
    "zoom": 0.85
  }
}
```

- Canvas state saves **500ms after each change** (debounced).
- Tiles are saved **immediately** when created or closed.
- `type` values: `"term"`, `"note"`, `"code"`, `"image"`

### `~/.collaborator/config.json`

```json
{
  "workspaces": [
    "/Users/you/projects/my-app",
    "/Users/you/projects/another-project"
  ],
  "active_workspace": 0,
  "window_state": {
    "x": 100,
    "y": 50,
    "width": 1440,
    "height": 900,
    "isMaximized": false
  },
  "ui": {}
}
```

---

## File Tile Binding Behavior

File tiles track their source file on disk:

| Event | Tile behavior |
|---|---|
| File renamed | Tile updates to track the new path automatically |
| File deleted | Tile is **closed** |
| File content changed externally | Tile **reloads** automatically |

This means agents editing files in terminals will be reflected live in open tiles.

---

## Common Agentic Workflows

### Pattern 1: Agent + context file side by side

```
Canvas layout:
┌─────────────────┐  ┌──────────────────────┐
│  Terminal        │  │  CLAUDE.md / spec.md │
│  (claude / codex)│  │  (note tile)         │
└─────────────────┘  └──────────────────────┘
```

1. Double-click canvas → terminal tile.
2. Drag `CLAUDE.md` or your spec file onto canvas → note tile.
3. Start agent in terminal; reference the spec in the same view.

### Pattern 2: Agent + code file being edited

```
Canvas layout:
┌─────────────────┐  ┌──────────────────────┐
│  Terminal        │  │  src/main.py         │
│  (agent running) │  │  (code tile, live)   │
└─────────────────┘  └──────────────────────┘
```

Drag `src/main.py` onto canvas. As the agent edits it, the code tile updates live.

### Pattern 3: Multi-agent setup

```
Canvas layout:
┌───────────┐  ┌───────────┐  ┌──────────────┐
│ Agent 1   │  │ Agent 2   │  │ notes.md     │
│ (frontend)│  │ (backend) │  │ (shared spec)│
└───────────┘  └───────────┘  └──────────────┘
```

Each terminal tile has its own independent tmux session. Both agents run concurrently. Pan the canvas to monitor both.

### Pattern 4: Reset/restore canvas state manually

```sh
# Back up current state
cp ~/.collaborator/canvas-state.json ~/.collaborator/canvas-state.backup.json

# Reset canvas (removes all tiles, resets viewport)
cat > ~/.collaborator/canvas-state.json << 'EOF'
{
  "version": 1,
  "tiles": [],
  "viewport": {
    "panX": 0,
    "panY": 0,
    "zoom": 1.0
  }
}
EOF
```

### Pattern 5: Add a workspace via config (without GUI)

```sh
# Add a workspace path directly to config
node -e "
const fs = require('fs');
const path = require('os').homedir() + '/.collaborator/config.json';
const cfg = JSON.parse(fs.readFileSync(path, 'utf8'));
cfg.workspaces.push('/Users/you/projects/new-project');
fs.writeFileSync(path, JSON.stringify(cfg, null, 2));
console.log('Workspace added.');
"
```

---

## Viewer (File Preview Panel)

Clicking a file in the navigator opens it in the **Viewer** — a full-featured editor/reader in the main area.

| File type | Viewer behavior |
|---|---|
| `.md`, `.mdx`, `.markdown`, `.txt` | Rich text editor, frontmatter, cover images, `[[wiki links]]` |
| All other text/code files | Monaco Editor, syntax highlighting, line numbers |
| Images (png/jpg/gif/svg/webp) | Image display with metadata |

- Viewer watches for external disk changes and **reloads automatically**.
- Press `Escape` to close the viewer (when not actively editing).
- Viewer and canvas tiles are independent — the same file can be open in both simultaneously.

---

## Troubleshooting

### App won't open / "damaged" warning on macOS
```sh
# Remove quarantine flag after manual download
xattr -cr /Applications/Collaborator.app
```

### Terminal tile shows wrong working directory
- Ensure the workspace is set before creating the terminal.
- Close and re-create the terminal tile after switching workspaces.

### Tile not reloading after external file change
- Collaborator watches files via the OS. On very large directories, watchers may be exhausted.
```sh
# macOS: check and raise fs watcher limit if needed
launchctl limit maxfiles
sudo launchctl limit maxfiles 65536 200000
```

### Canvas state corrupted / app won't load canvas
```sh
# Reset canvas state
echo '{"version":1,"tiles":[],"viewport":{"panX":0,"panY":0,"zoom":1.0}}' \
  > ~/.collaborator/canvas-state.json
```

### tmux session persists after tile is closed (intended behavior)
- Terminal tiles bind to tmux sessions that live beyond the tile.
- To list and kill stale sessions:
```sh
tmux ls
tmux kill-session -t <session-name>
```

### Reinstall cleanly
```sh
# Remove app state (workspaces, canvas, config)
rm -rf ~/.collaborator/

# Reinstall
curl -fsSL https://raw.githubusercontent.com/collaborator-ai/collab-public/main/install.sh | bash
```

---

## Key Facts for AI Agents

- **macOS Apple Silicon only** — no Linux, no Windows, no Intel Mac.
- **No accounts required** — fully local, no cloud sync.
- **Anonymous analytics** via PostHog (non-identifying).
- **Source not public** — bug reports go to GitHub Issues.
- **State location:** `~/.collaborator/`
- **Canvas tile types:** `term`, `note`, `code`, `image`
- **Terminal persistence:** tmux (sessions survive tile close/reopen)
- **Zoom range:** 33%–100%
- **Grid snap:** all tile positions and sizes snap to dot grid
```
