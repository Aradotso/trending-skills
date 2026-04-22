```markdown
---
name: open-codesign-ai-design
description: Open-source Claude Design alternative — turn prompts into prototypes/slides/PDFs locally with any AI model (Claude, GPT, Gemini, Ollama) via BYOK
triggers:
  - "set up open codesign"
  - "use open codesign to generate UI"
  - "prompt to prototype with open codesign"
  - "add a model provider to open codesign"
  - "import claude code api key into open codesign"
  - "build a design skill module for open codesign"
  - "export prototype as PDF or PPTX from open codesign"
  - "configure ollama with open codesign"
---

# Open CoDesign AI Design Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection

Open CoDesign is an MIT-licensed, local-first Electron desktop app that turns text prompts into polished HTML/JSX prototypes, slide decks, and marketing assets. It supports 20+ models via BYOK (Anthropic, OpenAI, Gemini, DeepSeek, Kimi, GLM, Ollama, OpenRouter, any OpenAI-compatible relay). No subscription, no mandatory cloud — credentials stay in `~/.config/open-codesign/config.toml`.

---

## Installation

### Pre-built binaries (fastest)

Download from [GitHub Releases](https://github.com/OpenCoworkAI/open-codesign/releases):

| Platform | File |
|---|---|
| macOS Apple Silicon | `open-codesign-*-arm64.dmg` |
| macOS Intel | `open-codesign-*-x64.dmg` |
| Windows x64 | `open-codesign-*-x64-setup.exe` |
| Linux x64 | `open-codesign-*-x64.AppImage` |

**macOS Gatekeeper bypass (unsigned v0.1.x):**
```sh
xattr -cr "/Applications/Open CoDesign.app"
```

**Homebrew (macOS):**
```sh
brew install --cask opencoworkai/tap/open-codesign
```

**Scoop (Windows):**
```sh
scoop bucket add opencoworkai https://github.com/OpenCoworkAI/scoop-bucket
scoop install open-codesign
```

### Build from source

```sh
git clone https://github.com/OpenCoworkAI/open-codesign.git
cd open-codesign
npm install
npm run build
npm start
```

Development mode with hot reload:
```sh
npm run dev
```

---

## Configuration

Credentials live in `~/.config/open-codesign/config.toml` (mode `0600`). Never commit this file.

### Manual config.toml

```toml
[providers.anthropic]
api_key = "$ANTHROPIC_API_KEY"   # set in shell; app reads env vars too
default_model = "claude-opus-4-5"

[providers.openai]
api_key = "$OPENAI_API_KEY"
default_model = "gpt-4o"

[providers.gemini]
api_key = "$GEMINI_API_KEY"
default_model = "gemini-2.0-flash"

[providers.ollama]
base_url = "http://localhost:11434"
default_model = "llama3.2"

[providers.openrouter]
api_key = "$OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
default_model = "anthropic/claude-opus-4-5"

[app]
theme = "dark"           # "light" | "dark" | "system"
locale = "en"            # "en" | "zh-CN"
save_path = "~/Documents/open-codesign"
```

### One-click import from Claude Code / Codex

Open CoDesign auto-detects:
- `~/.config/claude/config.toml` (Claude Code)
- `~/.config/codex/config.toml` (OpenAI Codex CLI)

Click **Settings → Import from Claude Code** or **Import from Codex** — providers, models, and keys transfer in a single click with no copy-paste.

---

## Core TypeScript API

Open CoDesign exposes an internal TypeScript API used by both the Electron main process and renderer. When contributing or extending:

### Provider client factory

```typescript
// src/providers/factory.ts
import { createProviderClient } from '@open-codesign/providers';

const client = createProviderClient({
  provider: 'anthropic',           // 'anthropic' | 'openai' | 'gemini' | 'ollama' | 'openrouter'
  apiKey: process.env.ANTHROPIC_API_KEY!,
  model: 'claude-opus-4-5',
  baseUrl: undefined,              // override for OpenAI-compatible relays
});

const stream = await client.streamGenerate({
  systemPrompt: 'You are a UI designer...',
  userMessage: 'Create a SaaS pricing page with three tiers',
  temperature: 0.7,
  maxTokens: 8192,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.text ?? '');
}
```

### Artifact generation

```typescript
// src/generation/artifact.ts
import { generateArtifact } from '@open-codesign/generation';

const artifact = await generateArtifact({
  prompt: 'A glassmorphism login card with email + password fields',
  outputFormat: 'html',     // 'html' | 'jsx' | 'react'
  skills: ['glassmorphism', 'landing-page'],   // built-in skill modules
  provider: 'anthropic',
  model: 'claude-opus-4-5',
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

console.log(artifact.code);   // full HTML string
console.log(artifact.title);  // auto-generated title
console.log(artifact.skills); // skills the model selected
```

### Export artifact

```typescript
import { exportArtifact } from '@open-codesign/export';

// Export to PDF
await exportArtifact({
  artifactId: 'abc123',
  format: 'pdf',              // 'pdf' | 'pptx' | 'html' | 'zip' | 'markdown'
  outputPath: './output/design.pdf',
  options: {
    pageSize: 'A4',           // pdf only
    landscape: false,
  },
});

// Export to PPTX (slide deck)
await exportArtifact({
  artifactId: 'abc123',
  format: 'pptx',
  outputPath: './output/pitch-deck.pptx',
  options: {
    slideWidth: 1280,
    slideHeight: 720,
  },
});
```

### Comment-mode patch

```typescript
import { patchArtifactRegion } from '@open-codesign/generation';

// Rewrite only the region the user pinned
const patched = await patchArtifactRegion({
  artifactId: 'abc123',
  elementSelector: '#pricing-card-pro',
  comment: 'Make this card stand out more — use a gradient border and bold CTA',
  provider: 'anthropic',
  model: 'claude-opus-4-5',
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

console.log(patched.diff);   // unified diff of the change
```

---

## Design Skill Modules

### Built-in skills (12)

`slide-deck` · `dashboard` · `landing-page` · `svg-charts` · `glassmorphism` · `editorial-typography` · `hero` · `pricing` · `footer` · `chat-ui` · `data-table` · `calendar`

Skills are selected automatically per prompt, but you can specify them explicitly:

```typescript
const artifact = await generateArtifact({
  prompt: 'Monthly revenue dashboard',
  skills: ['dashboard', 'svg-charts', 'data-table'],
  // ...
});
```

### Custom SKILL.md (project-level taste layer)

Create `SKILL.md` in your project root to teach the model your design system:

```markdown
# My Design System

## Colors
- Primary: #6366F1 (Indigo 500)
- Surface: #0F172A
- Text primary: #F8FAFC

## Typography
- Headings: Inter, weight 700, tracking -0.02em
- Body: Inter, weight 400, line-height 1.6

## Components
- Buttons: 8px border-radius, 44px min-height, always include focus ring
- Cards: 12px border-radius, 1px border rgba(255,255,255,0.08), backdrop-blur

## Rules
- Never use pure black backgrounds — use #0F172A or #020617
- Always include hover and focus states
- Prefer CSS Grid over flexbox for page-level layout
```

The app picks up `SKILL.md` automatically when it exists in the working directory.

---

## Ollama (local / offline)

```sh
# Install and pull a model
ollama pull llama3.2
ollama pull qwen2.5-coder:7b

# Verify it's running
curl http://localhost:11434/api/tags
```

In `config.toml`:
```toml
[providers.ollama]
base_url = "http://localhost:11434"
default_model = "qwen2.5-coder:7b"
```

Open CoDesign calls `/api/chat` (streaming) and `/api/tags` (model discovery) — no extra configuration needed once Ollama is running.

---

## OpenRouter / OpenAI-compatible relays

```toml
[providers.openrouter]
api_key = "$OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
default_model = "google/gemini-2.0-flash-001"
```

Any relay that implements the OpenAI Chat Completions API (`/v1/chat/completions`) works here, including SiliconFlow, Together AI, and self-hosted vLLM.

---

## Project structure (for contributors)

```
open-codesign/
├── src/
│   ├── main/           # Electron main process (Node.js)
│   │   ├── ipc/        # IPC handlers for generation, export, settings
│   │   └── providers/  # Provider clients (Anthropic, OpenAI, Gemini…)
│   ├── renderer/       # React UI (Vite + TypeScript)
│   │   ├── components/ # DesignHub, ArtifactViewer, AgentPanel, CommentMode
│   │   ├── hooks/      # useGeneration, useArtifact, useProvider
│   │   └── store/      # Zustand state slices
│   ├── generation/     # Prompt → artifact pipeline, skill routing
│   ├── export/         # PDF (Puppeteer), PPTX (pptxgenjs), ZIP, Markdown
│   └── skills/         # Built-in SKILL.md modules (12 design skills)
├── packaging/          # Homebrew, Scoop, winget manifests + auto-sync scripts
├── website/            # Docusaurus docs + marketing site
└── CONTRIBUTING.md
```

---

## IPC channel reference (Electron)

When building renderer-side features, use these IPC channels via `window.electron.invoke`:

```typescript
// Generate artifact
const artifact = await window.electron.invoke('generation:create', {
  prompt: string,
  outputFormat: 'html' | 'jsx',
  skills?: string[],
  providerId: string,
  modelId: string,
});

// List saved artifacts
const artifacts = await window.electron.invoke('artifacts:list');

// Export artifact
await window.electron.invoke('export:artifact', {
  artifactId: string,
  format: 'pdf' | 'pptx' | 'html' | 'zip' | 'markdown',
  outputPath: string,
});

// Get available models for a provider
const models = await window.electron.invoke('providers:models', {
  providerId: string,
});

// Cancel active generation
await window.electron.invoke('generation:cancel', { generationId: string });

// Patch region (comment mode)
const patch = await window.electron.invoke('generation:patch', {
  artifactId: string,
  elementSelector: string,
  comment: string,
});
```

---

## Common patterns

### Pattern 1 — Prompt → prototype → PDF pipeline

```typescript
import { generateArtifact, exportArtifact } from '@open-codesign/generation';

async function promptToPDF(prompt: string, outputPath: string) {
  // 1. Generate
  const artifact = await generateArtifact({
    prompt,
    outputFormat: 'html',
    provider: 'anthropic',
    model: 'claude-opus-4-5',
    apiKey: process.env.ANTHROPIC_API_KEY!,
  });

  // 2. Export
  await exportArtifact({
    artifactId: artifact.id,
    format: 'pdf',
    outputPath,
    options: { pageSize: 'A4', landscape: true },
  });

  return artifact;
}

await promptToPDF(
  'Quarterly business review slide deck, 8 slides, dark theme',
  './qbr-q2-2026.pdf'
);
```

### Pattern 2 — Iterative refinement loop

```typescript
import { generateArtifact, patchArtifactRegion } from '@open-codesign/generation';

let artifact = await generateArtifact({
  prompt: 'SaaS landing page hero section',
  outputFormat: 'html',
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: process.env.OPENAI_API_KEY!,
});

// Refine the CTA
artifact = await patchArtifactRegion({
  artifactId: artifact.id,
  elementSelector: '.hero-cta',
  comment: 'Make the CTA button larger, add a subtle pulse animation, change copy to "Start free — no card needed"',
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: process.env.OPENAI_API_KEY!,
});
```

### Pattern 3 — Dynamic model picker

```typescript
import { listProviderModels } from '@open-codesign/providers';

const models = await listProviderModels({
  provider: 'openrouter',
  apiKey: process.env.OPENROUTER_API_KEY!,
  baseUrl: 'https://openrouter.ai/api/v1',
});

// models is the live catalogue from the provider, not a hardcoded list
console.log(models.map(m => m.id));
```

---

## Troubleshooting

### macOS — app won't open after install
```sh
xattr -cr "/Applications/Open CoDesign.app"
# For older builds installed as lowercase:
xattr -cr "/Applications/open-codesign.app"
```

### API key not recognised
- Check `~/.config/open-codesign/config.toml` exists and has mode `0600`
- Anthropic keys start with `sk-ant-`; OpenAI keys start with `sk-`
- For Ollama, confirm `curl http://localhost:11434/api/tags` returns JSON before opening the app

### Generation hangs / no output
1. Open **Settings → Agent Panel** and check the live tool-call stream for error messages
2. Click **Cancel** (stops mid-stream cleanly, prior turns preserved)
3. Verify the selected model is available: `providers:models` IPC call or the model picker dropdown

### Ollama model not appearing
```sh
ollama list          # confirm the model is pulled
ollama ps            # confirm ollama serve is running
```
Restart the app after pulling new models — the model catalogue is fetched at startup.

### Export to PPTX produces blank slides
Ensure the artifact preview fully renders before exporting. Add a short delay or wait for the `artifact:rendered` event in the IPC layer before calling `export:artifact`.

### SQLite version history missing
History is stored at `~/Documents/open-codesign/history.sqlite` (or the `save_path` from config). If the file is missing, the app creates it on next launch. Do not delete while the app is running.

---

## Environment variables reference

```sh
ANTHROPIC_API_KEY        # Anthropic Claude
OPENAI_API_KEY           # OpenAI GPT
GEMINI_API_KEY           # Google Gemini
OPENROUTER_API_KEY       # OpenRouter
DEEPSEEK_API_KEY         # DeepSeek
SILICON_FLOW_API_KEY     # SiliconFlow relay
OPEN_CODESIGN_LOG_LEVEL  # debug | info | warn | error (default: info)
OPEN_CODESIGN_CONFIG     # override config file path
```

---

## Resources

- **Homepage:** https://opencoworkai.github.io/open-codesign/
- **Quickstart docs:** https://opencoworkai.github.io/open-codesign/quickstart
- **Releases + SBOMs:** https://github.com/OpenCoworkAI/open-codesign/releases
- **Contributing:** https://github.com/OpenCoworkAI/open-codesign/blob/main/CONTRIBUTING.md
- **Security policy:** https://github.com/OpenCoworkAI/open-codesign/blob/main/SECURITY.md
- **License:** MIT
```
