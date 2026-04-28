```markdown
---
name: open-design-ai
description: Local-first open-source alternative to Claude Design — wire your existing coding agent (Claude Code, Codex, Cursor, Gemini CLI) into a skill-driven design workflow with 19 skills and 71 brand-grade design systems.
triggers:
  - set up open design locally
  - use open design with my coding agent
  - generate a design artifact with open design
  - add a new skill to open design
  - pick a design system in open design
  - export HTML or PDF from open design
  - run the open design daemon
  - configure open design with my API key
---

# open-design-ai

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Open Design (OD) is the open-source, local-first alternative to Anthropic's Claude Design. It turns your existing coding agent (Claude Code, Codex CLI, Cursor Agent, Gemini CLI, OpenCode, Qwen) into a design engine backed by **19 composable Skills** and **71 brand-grade Design Systems**. Artifacts render in a sandboxed iframe and export to HTML, PDF, PPTX, ZIP, or Markdown.

---

## Installation

### Prerequisites

- Node.js ≥ 18
- pnpm ≥ 8
- One supported coding agent on `PATH`: `claude`, `codex`, `cursor`, `gemini`, `opencode`, or `qwen`

### Three-command quickstart

```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

### Vercel deploy

```bash
pnpm build
vercel deploy
```

### Single-process production

```bash
pnpm build
npm start
```

---

## Environment variables

Create `.env.local` (never commit this file):

```bash
# Required only when using the Anthropic API BYOK fallback
# (not needed if you rely on a local agent CLI)
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# Optional: pin which agent the daemon prefers
OD_AGENT=claude          # claude | codex | cursor | gemini | opencode | qwen

# Optional: override the port the local daemon listens on
OD_DAEMON_PORT=4242

# Optional: working directory for on-disk project folders
OD_WORKSPACE_DIR=$HOME/.open-design/workspace
```

The daemon PATH-scans for agents automatically; `OD_AGENT` just sets priority.

---

## Architecture overview

```
Browser (Next.js)
  │  streaming SSE / WebSocket
  ▼
Local Daemon  (src/daemon/)
  │  spawns agent CLI in a real on-disk project folder
  ▼
Agent CLI  (claude / codex / gemini / …)
  │  Read · Write · Bash · WebFetch against workspace
  ▼
Skill stack  (skills/<name>/SKILL.md + seed template + checklist)
  │  structured <artifact> tag in agent output
  ▼
Sandboxed iframe preview  (srcdoc, vendored React 18 + Babel)
```

---

## Key concepts

### Skills

Each skill lives under `skills/<name>/` and follows the Claude Code `SKILL.md` convention with an extended `od:` frontmatter block:

```yaml
# skills/web-prototype/SKILL.md
---
name: web-prototype
description: Interactive single-page web prototype
od:
  mode: prototype          # prototype | deck | template
  platform: web
  scenario: landing
  preview: iframe
  design_system: linear    # default design system slug
triggers:
  - build me a landing page
  - create a web prototype
  - make a SaaS homepage
---
```

Skills are discovered at startup by scanning `skills/*/SKILL.md`.

### Design Systems

71 design systems live under `design-systems/<slug>/DESIGN.md`. Each exposes:

- A 4-colour OKLch signature palette
- Typography stack (font family + scale)
- Spacing and radius tokens
- A live `showcase.html`

Reference a system by slug anywhere in the UI or in a skill's `od.design_system` field.

### Visual Directions (no-brand fallback)

When the user has no existing brand the agent emits a direction-picker form with five curated schools:

| Slug | School |
|---|---|
| `editorial-monocle` | Editorial Monocle |
| `modern-minimal` | Modern Minimal |
| `tech-utility` | Tech Utility |
| `brutalist` | Brutalist |
| `soft-warm` | Soft Warm |

Each direction ships a deterministic OKLch palette + font stack — no model freestyle.

---

## Running a design session (programmatic)

### POST /api/design — start a session

```typescript
// src/app/api/design/route.ts (simplified excerpt)
const response = await fetch('/api/design', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    skill: 'web-prototype',          // skill slug
    designSystem: 'linear',          // design system slug (optional)
    direction: 'modern-minimal',     // visual direction (optional, no-brand fallback)
    brief: 'A SaaS pricing page for a developer tool',
    discovery: {                     // answers from the turn-1 discovery form
      surface: 'web',
      audience: 'developers',
      tone: 'technical',
      scale: 'single-page',
    },
  }),
});

// Response is a Server-Sent Events stream
const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // Parse SSE lines: data: {"type":"todo"|"artifact"|"tool_call"|"done", ...}
  for (const line of chunk.split('\n')) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      handleEvent(event);
    }
  }
}
```

### Event types

```typescript
type DesignEvent =
  | { type: 'todo';      todos: Todo[] }           // live TodoWrite plan
  | { type: 'tool_call'; tool: string; input: unknown }
  | { type: 'artifact';  html: string }             // final renderable artifact
  | { type: 'done' }
  | { type: 'error';     message: string };
```

---

## Working with Skills

### Add a custom skill

```bash
mkdir -p skills/my-skill
cat > skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: One-line description of what this skill produces
od:
  mode: prototype
  platform: web
  scenario: custom
  preview: iframe
  design_system: vercel
triggers:
  - build me a custom thing
  - create a my-skill artifact
---

# My Skill

## What you produce
A single self-contained HTML file that …

## Checklist (P0 — must ship)
- [ ] Responsive at 375 px, 768 px, 1280 px
- [ ] No external image URLs (inline SVG or CSS gradients only)
- [ ] All colours from the active design system palette
EOF
```

OD hot-reloads skills in development — no restart needed.

### Add a seed template

Seed templates pre-populate the agent's workspace before generation:

```bash
mkdir -p skills/my-skill/seed
cat > skills/my-skill/seed/index.html << 'EOF'
<!DOCTYPE html>
<!-- SEED: agent reads this file first via the pre-flight Read step -->
<html lang="en">
<head><meta charset="UTF-8" /><title>{{title}}</title></head>
<body><!-- replace with generated content --></body>
</html>
EOF
```

---

## Working with Design Systems

### Browse available systems

```typescript
import { listDesignSystems } from '@/lib/design-systems';

const systems = await listDesignSystems();
// [{ slug: 'linear', name: 'Linear', palette: ['#5E6AD2', ...], ... }, ...]
```

### Load a specific system

```typescript
import { loadDesignSystem } from '@/lib/design-systems';

const ds = await loadDesignSystem('stripe');
console.log(ds.palette);   // ['#635BFF', '#0A2540', '#00D924', '#FFFFFF']
console.log(ds.fonts);     // { display: 'Sohne', body: 'Sohne', mono: 'Sohne Mono' }
console.log(ds.tokens);    // spacing, radius, shadow tokens
```

### Add a new design system

```bash
mkdir -p design-systems/my-brand
cat > design-systems/my-brand/DESIGN.md << 'EOF'
---
name: My Brand
slug: my-brand
palette:
  - '#1A1A2E'   # primary
  - '#16213E'   # secondary
  - '#0F3460'   # accent
  - '#E94560'   # highlight
fonts:
  display: 'Inter'
  body: 'Inter'
  mono: 'JetBrains Mono'
tokens:
  radius: '8px'
  spacing-unit: '8px'
---

## My Brand Design System

Use OKLch for all colour derivations. Primary background is near-black (`#1A1A2E`).
Accent `#E94560` for CTAs only — max one per viewport.
EOF
```

---

## Export

### From the UI

After an artifact renders, the export bar offers: **HTML · PDF · PPTX · ZIP · Markdown**.

### Programmatic export

```typescript
import { exportArtifact } from '@/lib/export';

// Export to PDF (uses headless Chromium via Puppeteer)
const pdf = await exportArtifact({ html: artifactHtml, format: 'pdf' });
// pdf is a Buffer — write to disk or return as a Response

// Export to PPTX (uses pptxgenjs under the hood)
const pptx = await exportArtifact({ html: artifactHtml, format: 'pptx' });

// Export to ZIP (html + assets bundled)
const zip = await exportArtifact({ html: artifactHtml, format: 'zip' });
```

---

## Device frames

Pixel-accurate frames are shared SVG assets in `public/frames/`:

```
public/frames/
  iphone-15-pro.svg   # Dynamic Island, status bar, home indicator
  pixel-8.svg
  ipad-pro.svg
  macbook-pro.svg
  browser-chrome.svg
```

Reference in a skill's seed template:

```html
<!-- Embed the iPhone frame; agent injects screen content into #screen slot -->
<div class="device-frame">
  <img src="/frames/iphone-15-pro.svg" alt="" aria-hidden="true" />
  <div id="screen" class="device-screen">
    <!-- agent writes here -->
  </div>
</div>
```

---

## Daemon API (local agent runtime)

The daemon runs at `http://localhost:${OD_DAEMON_PORT}` (default `4242`).

```typescript
// Check which agent is active
const status = await fetch('http://localhost:4242/status').then(r => r.json());
// { agent: 'claude', version: '1.2.3', workspace: '/Users/you/.open-design/workspace' }

// List on-disk project folders
const projects = await fetch('http://localhost:4242/projects').then(r => r.json());
// [{ id: 'proj_abc', skill: 'web-prototype', createdAt: '…' }, ...]

// Read a file from a project
const file = await fetch('http://localhost:4242/projects/proj_abc/files/index.html')
  .then(r => r.text());
```

---

## Supported coding agents

| Agent | CLI binary | Notes |
|---|---|---|
| Claude Code | `claude` | Best results; native `TodoWrite` support |
| Codex CLI | `codex` | Set `OPENAI_API_KEY` in env |
| Cursor Agent | `cursor` | Requires Cursor ≥ 0.40 with agent mode |
| Gemini CLI | `gemini` | Set `GEMINI_API_KEY` in env |
| OpenCode | `opencode` | BYOK via `.opencode.json` |
| Qwen Code | `qwen` | Set `DASHSCOPE_API_KEY` in env |
| Anthropic API (fallback) | — | Set `ANTHROPIC_API_KEY`; no local CLI needed |

---

## Common patterns

### Pre-flight read enforcement

Every skill enforces a pre-flight `Read` before writing. The system prompt instructs the agent:

```
Before writing any file, read:
1. skills/<name>/SKILL.md          — constraints and checklist
2. skills/<name>/seed/index.html   — starter scaffold (if present)
3. design-systems/<slug>/DESIGN.md — palette and token contract
```

This is what prevents "AI slop" — the agent cannot freestyle colours or layouts it hasn't explicitly read.

### 5-dimensional self-critique

After emitting a draft artifact the agent is prompted to score itself on:

1. **Visual hierarchy** — does the eye flow correctly?
2. **Brand fidelity** — are all colours from the design system palette?
3. **Typographic rhythm** — consistent scale and line-height?
4. **Accessibility** — contrast ≥ 4.5:1 for body text?
5. **Completeness** — all P0 checklist items checked?

Scores below 4/5 trigger a self-revision loop before the final `<artifact>` is emitted.

### Artifact tag format

```html
<artifact type="text/html" title="My Landing Page">
<!DOCTYPE html>
<html lang="en">
  <!-- full self-contained HTML here -->
</html>
</artifact>
```

OD's parser extracts the inner HTML and renders it in a sandboxed `srcdoc` iframe with `allow-scripts` but no `allow-same-origin`.

---

## Troubleshooting

### Agent not detected

```bash
# Confirm the agent binary is on PATH
which claude   # or codex, gemini, etc.

# Force a specific agent
OD_AGENT=claude pnpm dev
```

### Daemon port conflict

```bash
OD_DAEMON_PORT=4243 pnpm dev
```

### Artifact renders blank

- Check the browser console inside the iframe (DevTools → frame selector).
- The most common cause is an external font URL blocked by CSP. Use system fonts or inline the `@font-face` declaration.

### PDF export fails

Puppeteer requires Chromium. On Linux:

```bash
sudo apt-get install -y chromium-browser
PUPPETEER_EXECUTABLE_PATH=$(which chromium-browser) pnpm build && npm start
```

### Design system not found

```bash
ls design-systems/ | grep my-brand
# If missing, ensure the folder contains DESIGN.md with a valid `slug:` field
```

### `pnpm dev` exits immediately

Confirm Node.js ≥ 18:

```bash
node --version   # must be >= 18.0.0
```

---

## Project layout

```
open-design/
├── skills/                  # 19 built-in skills (SKILL.md + seed/ + example.html)
├── design-systems/          # 71 design system definitions (DESIGN.md)
├── src/
│   ├── app/                 # Next.js App Router pages and API routes
│   ├── daemon/              # Local daemon (agent PATH scan, project runner)
│   ├── lib/
│   │   ├── design-systems/  # loadDesignSystem, listDesignSystems
│   │   ├── export/          # exportArtifact (HTML/PDF/PPTX/ZIP/MD)
│   │   └── skills/          # loadSkill, listSkills
│   └── prompts/
│       └── discovery.ts     # Turn-1 form · direction picker · self-critique prompt
├── public/frames/           # Device frame SVGs
└── docs/screenshots/        # README assets
```
```
