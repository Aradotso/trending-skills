```markdown
---
name: design-md-chrome-extractor
description: Chrome extension that extracts design tokens from any website and generates DESIGN.md or SKILL.md files for use with AI coding agents
triggers:
  - extract design system from website
  - generate DESIGN.md from site styles
  - create design skill for AI agent
  - extract colors and typography from webpage
  - generate SKILL.md design file
  - build design system documentation from URL
  - capture website design tokens
  - create TypeUI design blueprint
---

# TypeUI DESIGN.md Extractor (Chrome Extension)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Chrome extension that reads visual styles from any live website — typography, colors, spacing, border radius, shadows, and motion — and generates structured `DESIGN.md` or `SKILL.md` files compatible with AI coding agents like Claude Code, Cursor, Codex, and Google Stitch.

---

## What It Does

- **Auto-extracts** computed CSS styles from any active browser tab
- **Generates `DESIGN.md`** — a human-readable design system documentation file
- **Generates `SKILL.md`** — an agent-ready skill file for AI coding assistants
- Follows the open-source [TypeUI DESIGN.md](https://www.typeui.sh/design-skills) format
- Outputs downloadable markdown files for immediate use in AI workflows

---

## Installation (Developer / Load Unpacked)

```bash
# 1. Clone the repository
git clone https://github.com/bergside/design-md-chrome.git
cd design-md-chrome

# 2. Open Chrome and navigate to extensions
# chrome://extensions

# 3. Enable Developer Mode (toggle top-right)

# 4. Click "Load unpacked" and select the project folder
```

After loading:
- The extension icon appears in your Chrome toolbar
- Navigate to any website
- Click the extension icon to open the popup

---

## Project Structure

```
design-md-chrome/
├── manifest.json          # Chrome extension manifest (MV3)
├── popup.html             # Extension popup UI
├── popup.js               # Popup logic, UI interactions, file download
├── content.js             # Content script — injected into active tab to extract styles
├── background.js          # Service worker for Chrome extension messaging
├── generator.js           # DESIGN.md / SKILL.md markdown generation logic
├── tests/
│   └── run-tests.mjs      # Node.js test runner
└── README.md
```

---

## How Style Extraction Works

The content script (`content.js`) is injected into the active tab and uses `getComputedStyle` to sample elements across the DOM, collecting:

| Token Category | What Gets Extracted |
|---|---|
| Typography | Font families, sizes, weights, line heights, letter spacing |
| Colors | Background, text, border, and accent colors (deduped) |
| Spacing | Margin/padding values across key elements |
| Border Radius | Extracted from buttons, cards, inputs |
| Shadows | `box-shadow` values from elevated elements |
| Motion | `transition` and `animation` properties |

### Example: Content Script Extraction Pattern

```javascript
// content.js — core extraction logic pattern
function extractDesignTokens() {
  const elements = document.querySelectorAll(
    'h1, h2, h3, p, a, button, input, [class*="card"], [class*="container"]'
  );

  const tokens = {
    typography: new Set(),
    colors: new Set(),
    spacing: new Set(),
    borderRadius: new Set(),
    shadows: new Set(),
    motion: new Set(),
  };

  elements.forEach((el) => {
    const styles = window.getComputedStyle(el);

    // Typography
    tokens.typography.add({
      fontFamily: styles.fontFamily,
      fontSize: styles.fontSize,
      fontWeight: styles.fontWeight,
      lineHeight: styles.lineHeight,
      letterSpacing: styles.letterSpacing,
    });

    // Colors
    tokens.colors.add(styles.color);
    tokens.colors.add(styles.backgroundColor);
    tokens.colors.add(styles.borderColor);

    // Spacing
    tokens.spacing.add(styles.padding);
    tokens.spacing.add(styles.margin);

    // Radius
    tokens.borderRadius.add(styles.borderRadius);

    // Shadows
    if (styles.boxShadow !== 'none') {
      tokens.shadows.add(styles.boxShadow);
    }

    // Motion
    if (styles.transition !== 'all 0s ease 0s') {
      tokens.motion.add(styles.transition);
    }
  });

  return {
    typography: [...tokens.typography],
    colors: [...tokens.colors].filter(Boolean),
    spacing: [...tokens.spacing].filter(Boolean),
    borderRadius: [...tokens.borderRadius].filter(Boolean),
    shadows: [...tokens.shadows],
    motion: [...tokens.motion],
    url: window.location.href,
    title: document.title,
  };
}

// Send extracted tokens back to popup via Chrome messaging
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extractStyles') {
    sendResponse(extractDesignTokens());
  }
});
```

---

## Triggering Extraction from Popup

```javascript
// popup.js — trigger content script and receive tokens
async function runExtraction() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Inject content script if not already present
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js'],
  });

  // Request extraction
  chrome.tabs.sendMessage(tab.id, { action: 'extractStyles' }, (tokens) => {
    if (chrome.runtime.lastError) {
      console.error('Extraction failed:', chrome.runtime.lastError.message);
      return;
    }
    displayTokens(tokens);
    window.__extractedTokens = tokens; // store for generation
  });
}

document.getElementById('btn-extract').addEventListener('click', runExtraction);
```

---

## Generating DESIGN.md

```javascript
// generator.js — generate DESIGN.md from extracted tokens
function generateDesignMd(tokens) {
  const { url, title, typography, colors, spacing, borderRadius, shadows, motion } = tokens;

  return `# DESIGN.md — ${title}

## Mission
Define a consistent, accessible design system for ${title} (${url}).

## Brand
- **Product**: ${title}
- **URL**: ${url}
- **Audience**: Web users
- **Surface**: Web application

## Style Foundations

### Typography
${typography.map(t => `- Font: ${t.fontFamily}, Size: ${t.fontSize}, Weight: ${t.fontWeight}`).join('\n')}

### Colors
${colors.slice(0, 20).map(c => `- \`${c}\``).join('\n')}

### Spacing
${[...new Set(spacing)].slice(0, 10).map(s => `- ${s}`).join('\n')}

### Border Radius
${[...new Set(borderRadius)].slice(0, 8).map(r => `- ${r}`).join('\n')}

### Shadows
${shadows.slice(0, 6).map(s => `- ${s}`).join('\n')}

### Motion
${motion.slice(0, 6).map(m => `- ${m}`).join('\n')}

## Accessibility
- Follow WCAG 2.2 AA standards
- Minimum contrast ratio: 4.5:1 for text, 3:1 for UI components
- All interactive elements must have visible focus states
- Support keyboard navigation throughout

## Writing Tone
- Implementation-ready and precise
- Avoid ambiguity in component descriptions
- Use design token references over raw values

## Rules: Do
- Use extracted color tokens consistently
- Apply spacing scale from extracted values
- Maintain border radius consistency per component type
- Use motion values for transitions

## Rules: Don't
- Do not introduce new colors outside the extracted palette
- Do not use arbitrary spacing values
- Do not remove focus indicators
- Do not use motion that violates prefers-reduced-motion

## Quality Gates
- [ ] All colors reference extracted tokens
- [ ] Typography scale applied consistently
- [ ] WCAG AA contrast validated
- [ ] Keyboard navigation verified
`;
}
```

---

## Generating SKILL.md

```javascript
// generator.js — generate agent-ready SKILL.md
function generateSkillMd(tokens) {
  const { url, title } = tokens;

  return `---
name: ${title.toLowerCase().replace(/\s+/g, '-')}-design-system
description: Design system skill extracted from ${title} (${url})
triggers:
  - implement ${title} design
  - use ${title} styles
  - apply ${title} design system
  - build UI matching ${title}
---

# ${title} Design System Skill

Extracted design system from ${url}.

## Style Foundations
[...generated token documentation...]

## Component Patterns
[...inferred patterns from extracted styles...]
`;
}
```

---

## Downloading Generated Files

```javascript
// popup.js — download as .md file
function downloadMarkdown(content, filename) {
  const blob = new Blob([content], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Usage
document.getElementById('btn-download-design').addEventListener('click', () => {
  const tokens = window.__extractedTokens;
  const content = generateDesignMd(tokens);
  downloadMarkdown(content, 'DESIGN.md');
});

document.getElementById('btn-download-skill').addEventListener('click', () => {
  const tokens = window.__extractedTokens;
  const content = generateSkillMd(tokens);
  downloadMarkdown(content, 'SKILL.md');
});
```

---

## Using Generated Files with AI Agents

### Claude Code
```bash
# Place DESIGN.md in your project root
cp ~/Downloads/DESIGN.md ./DESIGN.md

# Reference in CLAUDE.md
echo "Follow the design system defined in DESIGN.md" >> CLAUDE.md
```

### Cursor
```
# Add to .cursorrules
Follow the design system documented in DESIGN.md for all UI implementations.
```

### Codex / OpenAI
```bash
# Pass as context file
codex --context DESIGN.md "build a login page"
```

### Google Stitch
Upload `DESIGN.md` as a project reference document in the Stitch workspace.

---

## Running Tests

```bash
# Run the test suite locally (Node.js required)
node tests/run-tests.mjs
```

### Writing Tests

```javascript
// tests/run-tests.mjs
import { generateDesignMd, generateSkillMd } from '../generator.js';

const mockTokens = {
  url: 'https://example.com',
  title: 'Example Site',
  typography: [
    { fontFamily: 'Inter, sans-serif', fontSize: '16px', fontWeight: '400', lineHeight: '1.5', letterSpacing: '0px' }
  ],
  colors: ['rgb(0, 0, 0)', 'rgb(255, 255, 255)', 'rgb(59, 130, 246)'],
  spacing: ['16px', '24px', '32px'],
  borderRadius: ['4px', '8px', '9999px'],
  shadows: ['0 1px 3px rgba(0,0,0,0.1)'],
  motion: ['all 0.2s ease'],
};

// Test DESIGN.md generation
const designMd = generateDesignMd(mockTokens);
console.assert(designMd.includes('## Mission'), 'Missing Mission section');
console.assert(designMd.includes('Inter, sans-serif'), 'Missing font family');
console.assert(designMd.includes('WCAG'), 'Missing accessibility section');
console.log('✅ DESIGN.md generation tests passed');

// Test SKILL.md generation
const skillMd = generateSkillMd(mockTokens);
console.assert(skillMd.includes('---'), 'Missing YAML frontmatter');
console.assert(skillMd.includes('triggers:'), 'Missing triggers');
console.log('✅ SKILL.md generation tests passed');
```

---

## Manifest Configuration (MV3)

```json
{
  "manifest_version": 3,
  "name": "TypeUI DESIGN.md Extractor",
  "version": "1.0.0",
  "description": "Extract design tokens from any website and generate DESIGN.md / SKILL.md",
  "permissions": [
    "activeTab",
    "scripting",
    "downloads"
  ],
  "action": {
    "default_popup": "popup.html",
    "default_icon": "icon.png"
  },
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [],
  "host_permissions": [
    "<all_urls>"
  ]
}
```

> Note: `"content_scripts"` is intentionally empty — scripts are injected programmatically via `chrome.scripting.executeScript` for on-demand extraction only.

---

## Available Extension Actions

| Button | Action |
|---|---|
| **Auto-extract** | Injects content script and collects computed styles |
| **Generate DESIGN.md** | Produces design system documentation from tokens |
| **Generate SKILL.md** | Produces agent-ready skill file from tokens |
| **Refresh** | Re-runs extraction for current page state |
| **Download** | Saves the generated file locally |
| **?** (Explain) | Shows how the file was generated with TypeUI reference |

---

## Troubleshooting

### Extension not extracting styles
```
Problem: Extension shows no data after clicking extract
Fix: Refresh the page, then re-click extract. Some SPAs need full load.
```

```javascript
// Check if content script is reachable
chrome.tabs.sendMessage(tabId, { action: 'ping' }, (response) => {
  if (chrome.runtime.lastError) {
    // Re-inject the content script
    chrome.scripting.executeScript({
      target: { tabId },
      files: ['content.js'],
    });
  }
});
```

### CORS / CSP blocked pages
```
Problem: Extension cannot inject scripts on chrome://, file://, or CSP-restricted pages
Fix: Navigate to a public HTTP/HTTPS website. Extension requires scripting access.
```

### Empty color palette
```
Problem: Colors list is empty or shows only defaults
Fix: Scroll through the page before extracting — lazy-loaded components need to render first.
```

### Download not triggering
```
Problem: Download button clicks but no file appears
Fix: Check Chrome's download settings. Ensure the "downloads" permission is in manifest.json.
```

### Tests failing in Node.js
```bash
# Ensure you're using Node.js 18+ for ESM support
node --version  # should be v18+

# Run with explicit ESM flag if needed
node --experimental-vm-modules tests/run-tests.mjs
```

---

## Curated Design Skills

Browse pre-extracted design systems at [typeui.sh/design-skills](https://www.typeui.sh/design-skills) — ready-to-use SKILL.md files for popular products and design systems.

---

## License

MIT — open source, free to use and modify.
```
