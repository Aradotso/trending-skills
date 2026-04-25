```markdown
---
name: garden-skills-web-design-engineer
description: AI agent skill that transforms AI-generated web pages from functional to stunning using design systems, oklch colors, curated typography, and anti-cliché rules
triggers:
  - make this web page look better
  - improve the design of this HTML page
  - create a beautiful landing page
  - design a stunning web interface
  - apply web design engineer skill
  - transform this from functional to stunning
  - use garden skills for web design
  - generate a high quality web page design
---

# Web Design Engineer Skill (garden-skills)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

An AI agent skill that elevates AI-generated HTML/CSS/JavaScript artifacts from generic and functional to polished and visually compelling. Inspired by Claude Design's system prompt, this skill encodes design taste, anti-pattern rules, typography pairings, and a structured six-step workflow directly into your agent's decision-making process.

---

## What This Skill Does

- **Bans AI design clichés** — no purple-pink gradients, no emoji icons, no Inter font, no fabricated testimonials
- **Enforces oklch color theory** — perceptually uniform color derivation for harmonious palettes
- **Provides curated font × color pairings** — six validated visual systems for common use cases
- **Defines a six-step workflow** — from requirements gathering through verification
- **Ships a pattern library** — ready-to-use code templates for advanced UI patterns

---

## Installation

Copy the skill directory into your project root:

```
your-project/
├── .agents/skills/web-design-engineer/
│   ├── SKILL.md
│   └── references/
│       └── advanced-patterns.md
└── ...
```

For Cursor or tools using `.claude/`:

```
your-project/
├── .claude/skills/web-design-engineer/
│   ├── SKILL.md
│   └── references/
│       └── advanced-patterns.md
└── ...
```

Clone directly from GitHub:

```bash
git clone https://github.com/ConardLi/garden-skills.git _tmp_garden
mkdir -p .agents/skills/web-design-engineer/references
cp _tmp_garden/.agents/skills/web-design-engineer/SKILL.md .agents/skills/web-design-engineer/
cp _tmp_garden/.agents/skills/web-design-engineer/references/advanced-patterns.md .agents/skills/web-design-engineer/references/
rm -rf _tmp_garden
```

The agent picks up the skill automatically when your request involves visual or interactive front-end work.

---

## The Six-Step Workflow

When this skill is active, the agent follows this structured process:

```
1. Understand requirements  →  Ask only when information is insufficient
2. Gather design context    →  Examine existing code/screenshots; never start from nothing
3. Declare design system    →  Colors, fonts, spacing, motion — written in Markdown before any code
4. Show v0 draft early      →  Placeholders + layout + tokens; allow course correction
5. Full build               →  Components, states, motion; pause at key decision points
6. Verify                   →  Pre-delivery checklist: no console errors, no rogue hues
```

---

## Design System Declaration (Step 3)

Before writing any code, the agent declares the design system in plain Markdown:

```markdown
## Design System

### Colors (oklch)
- Primary:    oklch(55% 0.18 265)   /* blue-violet */
- Surface:    oklch(98% 0.005 265)  /* near-white with hue tint */
- On-surface: oklch(18% 0.02 265)   /* near-black */
- Accent:     oklch(72% 0.14 145)   /* muted green */

### Typography
- Display:  "Space Grotesk", sans-serif  — weights 400 600 700
- Body:     "Inter", sans-serif          — weights 400 500
- Mono:     "JetBrains Mono", monospace  — weight 400

### Spacing scale (rem)
0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 / 4 / 6 / 8 / 12

### Motion
- Easing:    cubic-bezier(0.16, 1, 0.3, 1)
- Duration:  150ms micro / 300ms element / 600ms page
```

---

## oklch Color System

Colors use the perceptually uniform `oklch(lightness% chroma hue)` space. Unlike HSL, equal lightness values *look* equally bright across all hues.

```css
:root {
  /* Primary scale — same lightness steps, perceptually consistent */
  --color-primary-900: oklch(25% 0.18 265);
  --color-primary-700: oklch(40% 0.20 265);
  --color-primary-500: oklch(55% 0.22 265);  /* base */
  --color-primary-300: oklch(72% 0.16 265);
  --color-primary-100: oklch(93% 0.05 265);

  /* Surface tokens */
  --color-surface:        oklch(98% 0.005 265);
  --color-surface-raised: oklch(100% 0 0);
  --color-on-surface:     oklch(18% 0.02 265);
  --color-on-surface-dim: oklch(48% 0.04 265);

  /* Semantic */
  --color-success: oklch(62% 0.17 145);
  --color-warning: oklch(75% 0.18 75);
  --color-danger:  oklch(58% 0.22 25);
}
```

Derive tints and shades by adjusting only the `lightness%` value while keeping `chroma` and `hue` fixed:

```css
/* Hover state: lighten by ~8% */
--color-primary-hover: oklch(63% 0.22 265);

/* Disabled: desaturate (reduce chroma) */
--color-primary-disabled: oklch(55% 0.06 265);
```

---

## Curated Font × Color Pairings

| Style | Primary Color (oklch) | Display Font | Body Font | Use Case |
|---|---|---|---|---|
| Modern tech | `oklch(55% 0.20 265)` | Space Grotesk | Inter | SaaS, dev tools |
| Elegant editorial | `oklch(42% 0.12 55)` | Newsreader | Outfit | Content, blogs |
| Premium brand | `oklch(22% 0.03 265)` | Sora | Plus Jakarta Sans | Luxury, finance |
| Lively consumer | `oklch(60% 0.22 22)` | Plus Jakarta Sans | Outfit | E-commerce, social |
| Minimal professional | `oklch(52% 0.14 195)` | Outfit | Space Grotesk | Dashboards, B2B |
| Artisan warmth | `oklch(58% 0.14 60)` | Caveat | Newsreader | Food, education |

Load Google Fonts for a pairing:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- Modern tech pairing -->
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&family=JetBrains+Mono&display=swap" rel="stylesheet">
```

---

## Anti-Cliché Rules

The skill **explicitly bans** the following patterns. The agent will refuse to use them:

```
BANNED colors/gradients:
  - Purple-pink-blue gradient backgrounds (linear-gradient with purple + pink + blue)
  - Neon glow effects as primary design elements
  - #3b82f6 (Tailwind blue-500) as default primary

BANNED layout patterns:
  - Left-border accent cards (border-left: 4px solid var(--primary))
  - Feature grids with emoji + heading + one-line description
  - "Trusted by X companies" fake logo walls
  - Fabricated testimonials with stock avatar initials
  - Fabricated statistics ("10,000+ users", "99.9% uptime")

BANNED typography:
  - Inter, Roboto, Arial, Fraunces, system-ui as display fonts
  - All-caps headings as the primary typographic move
  - Font size hero text above 96px on desktop

BANNED components:
  - Emoji as icon substitutes in production UI
  - Poorly approximated SVG icons (stick to [icon] placeholders)
  - Blue primary buttons as default without justification
```

---

## Placeholder Philosophy

When assets are missing, use honest text markers rather than broken SVGs:

```html
<!-- Icon placeholder -->
<span class="icon-placeholder" aria-label="settings icon">[icon: settings]</span>

<!-- Image placeholder -->
<div class="img-placeholder" role="img" aria-label="Hero product shot">
  <span>Hero image — 1440 × 800 — product in use</span>
</div>

<!-- Logo placeholder -->
<div class="logo-placeholder">[Logo: CompanyName]</div>
```

```css
.icon-placeholder {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.25em;
  height: 1.25em;
  background: oklch(92% 0.03 265);
  border-radius: 3px;
  font-size: 0.6em;
  color: oklch(45% 0.08 265);
  font-family: "Space Grotesk", sans-serif;
  vertical-align: middle;
}

.img-placeholder {
  background: oklch(94% 0.02 265);
  display: flex;
  align-items: center;
  justify-content: center;
  color: oklch(55% 0.04 265);
  font-size: 0.875rem;
  font-style: italic;
}
```

---

## Complete Page Template

A minimal but well-structured starting point using the skill's conventions:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
  <style>
    /* ── Design Tokens ───────────────────────────────────────── */
    :root {
      --c-primary:      oklch(55% 0.20 265);
      --c-primary-dim:  oklch(40% 0.18 265);
      --c-surface:      oklch(98% 0.005 265);
      --c-surface-2:    oklch(95% 0.008 265);
      --c-on-surface:   oklch(18% 0.02 265);
      --c-on-dim:       oklch(48% 0.04 265);
      --c-border:       oklch(88% 0.01 265);

      --font-display: "Space Grotesk", sans-serif;
      --font-body:    "Inter", sans-serif;

      --sp-1: 0.25rem;  --sp-2: 0.5rem;  --sp-3: 0.75rem;
      --sp-4: 1rem;     --sp-6: 1.5rem;  --sp-8: 2rem;
      --sp-12: 3rem;    --sp-16: 4rem;   --sp-24: 6rem;

      --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
      --dur-fast: 150ms;
      --dur-base: 300ms;
    }

    /* ── Reset ───────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { font-size: 16px; -webkit-font-smoothing: antialiased; }
    body {
      background: var(--c-surface);
      color: var(--c-on-surface);
      font-family: var(--font-body);
      line-height: 1.6;
    }

    /* ── Typography scale ────────────────────────────────────── */
    h1, h2, h3, h4 {
      font-family: var(--font-display);
      font-weight: 700;
      line-height: 1.15;
      letter-spacing: -0.02em;
    }
    h1 { font-size: clamp(2.5rem, 6vw, 4.5rem); }
    h2 { font-size: clamp(1.75rem, 4vw, 3rem); }
    h3 { font-size: clamp(1.25rem, 2.5vw, 1.75rem); }

    /* ── Layout utilities ────────────────────────────────────── */
    .container {
      width: min(1200px, 100% - var(--sp-8) * 2);
      margin-inline: auto;
    }
    .section { padding-block: var(--sp-24); }

    /* ── Nav ─────────────────────────────────────────────────── */
    .nav {
      position: sticky; top: 0; z-index: 100;
      background: oklch(98% 0.005 265 / 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--c-border);
      padding-block: var(--sp-4);
    }
    .nav-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .nav-logo {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.125rem;
      color: var(--c-on-surface);
      text-decoration: none;
    }
    .nav-links {
      display: flex; gap: var(--sp-6);
      list-style: none;
    }
    .nav-links a {
      color: var(--c-on-dim);
      text-decoration: none;
      font-size: 0.9375rem;
      transition: color var(--dur-fast) ease;
    }
    .nav-links a:hover { color: var(--c-on-surface); }

    /* ── Button ──────────────────────────────────────────────── */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: var(--sp-2);
      padding: var(--sp-3) var(--sp-6);
      border: none;
      border-radius: 6px;
      font-family: var(--font-display);
      font-weight: 600;
      font-size: 0.9375rem;
      cursor: pointer;
      text-decoration: none;
      transition:
        background var(--dur-fast) ease,
        transform  var(--dur-fast) ease,
        box-shadow var(--dur-fast) ease;
    }
    .btn:active { transform: scale(0.98); }

    .btn-primary {
      background: var(--c-primary);
      color: white;
    }
    .btn-primary:hover {
      background: var(--c-primary-dim);
      box-shadow: 0 4px 16px oklch(55% 0.20 265 / 0.35);
    }

    .btn-ghost {
      background: transparent;
      color: var(--c-on-surface);
      border: 1px solid var(--c-border);
    }
    .btn-ghost:hover { background: var(--c-surface-2); }

    /* ── Hero ────────────────────────────────────────────────── */
    .hero {
      padding-block: var(--sp-24);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: var(--sp-16);
      align-items: center;
    }
    .hero-eyebrow {
      font-size: 0.8125rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--c-primary);
      margin-bottom: var(--sp-4);
    }
    .hero-heading { margin-bottom: var(--sp-6); }
    .hero-body {
      font-size: 1.125rem;
      color: var(--c-on-dim);
      margin-bottom: var(--sp-8);
      max-width: 44ch;
    }
    .hero-actions { display: flex; gap: var(--sp-3); flex-wrap: wrap; }

    .hero-visual {
      aspect-ratio: 4/3;
      background: var(--c-surface-2);
      border-radius: 12px;
      border: 1px solid var(--c-border);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--c-on-dim);
      font-size: 0.875rem;
      font-style: italic;
    }

    /* ── Card ────────────────────────────────────────────────── */
    .card {
      background: white;
      border: 1px solid var(--c-border);
      border-radius: 10px;
      padding: var(--sp-8);
      transition:
        transform var(--dur-base) var(--ease-out-expo),
        box-shadow var(--dur-base) var(--ease-out-expo);
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 32px oklch(18% 0.02 265 / 0.08);
    }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: var(--sp-6);
    }

    /* ── Footer ──────────────────────────────────────────────── */
    .footer {
      background: var(--c-on-surface);
      color: oklch(70% 0.02 265);
      padding-block: var(--sp-12);
    }
    .footer a { color: oklch(70% 0.02 265); text-decoration: none; }
    .footer a:hover { color: white; }
  </style>
</head>
<body>

  <header class="nav">
    <div class="container nav-inner">
      <a href="/" class="nav-logo">Brand</a>
      <nav>
        <ul class="nav-links">
          <li><a href="#features">Features</a></li>
          <li><a href="#pricing">Pricing</a></li>
          <li><a href="#docs">Docs</a></li>
        </ul>
      </nav>
      <a href="#signup" class="btn btn-primary">Get started</a>
    </div>
  </header>

  <main>
    <section class="section">
      <div class="container hero">
        <div>
          <p class="hero-eyebrow">New in 2026</p>
          <h1 class="hero-heading">Your compelling headline here</h1>
          <p class="hero-body">
            A concise value proposition in two sentences. Focus on the outcome
            the user gets, not the features you ship.
          </p>
          <div class="hero-actions">
            <a href="#signup" class="btn btn-primary">Start for free</a>
            <a href="#demo" class="btn btn-ghost">Watch demo</a>
          </div>
        </div>
        <div class="hero-visual">
          [Hero image — 800 × 600 — product screenshot]
        </div>
      </div>
    </section>

    <section class="section" id="features">
      <div class="container">
        <h2 style="margin-bottom: var(--sp-12)">Key features</h2>
        <div class="card-grid">
          <article class="card">
            <div style="margin-bottom: var(--sp-4); color: var(--c-primary); font-weight: 600;">
              [icon: feature-1] Feature one
            </div>
            <p style="color: var(--c-on-dim); font-size: 0.9375rem;">
              Describe the concrete benefit in one or two sentences.
            </p>
          </article>
          <article class="card">
            <div style="margin-bottom: var(--sp-4); color: var(--c-primary); font-weight: 600;">
              [icon: feature-2] Feature two
            </div>
            <p style="color: var(--c-on-dim); font-size: 0.9375rem;">
              Describe the concrete benefit in one or two sentences.
            </p>
          </article>
          <article class="card">
            <div style="margin-bottom: var(--sp-4); color: var(--c-primary); font-weight: 600;">
              [icon: feature-3] Feature three
            </div>
            <p style="color: var(--c-on-dim); font-size: 0.9375rem;">
              Describe the concrete benefit in one or two sentences.
            </p>
          </article>
        </div>
      </div>
    </section>
  </main>

  <footer class="footer">
    <div class="container">
      <p style="font-size: 0.875rem;">
        &copy; 2026 Brand. Built with care.
      </p>
    </div>
  </footer>

</body>
</html>
```

---

## Motion Patterns

```css
/* Entrance animation — staggered children */
@keyframes fade-up {
  from {
    opacity: 0;
    transform: translateY(24px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-in {
  animation: fade-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* Apply stagger via custom property */
.card:nth-child(1) { --delay: 0ms; }
.card:nth-child(2) { --delay: 80ms; }
.card:nth-child(3) { --delay: 160ms; }
.card:nth-child(4) { --delay: 240ms; }
.card { animation-delay: var(--delay, 0ms); }

/* Intersection Observer trigger */
const observer = new IntersectionObserver(
  (entries) => entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add("animate-in");
      observer.unobserve(e.target);
    }
  }),
  { threshold: 0.15 }
);
document.querySelectorAll(".card").forEach(el => observer.observe(el));
```

---

## Supported Output Types

| Type | What the skill produces |
|---|---|
| Landing pages | Marketing sites, product pages, portfolios |
| Interactive prototypes | Clickable app mockups with device frames |
| Slide decks | 1920×1080 HTML presentations with keyboard nav |
| Dashboards | Data visualizations with Chart.js or D3.js |
| Animations | CSS/JS motion design, timeline-driven demos |
| Design systems | Token exploration pages, component variant galleries |

---

## Pre-Delivery Verification Checklist

Before delivering any page, the agent runs through:

```
□ No console errors or warnings
□ oklch values used for all colors (no stray hex codes except white/black)
□ No banned fonts loaded (Inter, Roboto, Arial, Fraunces, system-ui as display)
□ No purple-pink-blue gradients present
□ No emoji used as functional icons
□ No fabricated data (stats, testimonials, logo walls)
□ All placeholder assets clearly marked with [bracket notation]
□ Hover and focus states present on all interactive elements
□ No inline styles that contradict design tokens
□ Page renders correctly at 375px, 768px, and 1440px
□ Images have meaningful alt text or role="presentation"
□ Font weights actually loaded from Google Fonts link
```

---

## Troubleshooting

**Agent keeps using Inter or system-ui fonts**
Ensure `SKILL.md` is in the correct skill directory for your tool. Verify the agent acknowledges the skill at the start of the session. Explicitly say "apply the web design engineer skill."

**Colors look inconsistent across browsers**
`oklch()` has excellent support in modern browsers (Chrome 111+, Firefox 113+, Safari 15.4+). For older targets, provide a fallback:
```css
.btn-primary {
  background: #5b4fcf; /* fallback */
  background: oklch(55% 0.20 265);
}
```

**Agent ignores the six-step workflow**
Prompt explicitly: "Follow the six-step design workflow from the web design engineer skill — start by declaring the design system before writing any code."

**v0 draft never shown**
Ask: "Show me a v0 draft with placeholders before building the full page."

**oklch colors not rendering (older Safari)**
Check Safari version. Safari 15.4+ supports oklch. For Safari 15.3 and below, add CSS `@supports` fallbacks:
```css
@supports not (color: oklch(0% 0 0)) {
  :root {
    --c-primary: #5b4fcf;
    --c-surface: #f9f9fe;
  }
}
```

---

## References

- `SKILL.md` — Main skill definition (~400 lines)
- `references/advanced-patterns.md` — Code template library (~520 lines)
- `demo/demo1.html` — Space museum without skill (baseline)
- `demo/demo1-with-skill.html` — Space museum with skill applied
- `demo/demo2-with-skill.html` — Photographer portfolio with skill applied
- `prompt/system.md` — Original Claude Design system prompt (reference only)
```
