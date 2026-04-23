```markdown
---
name: web-design-skill
description: Transform AI-generated web pages from functional to stunning using structured design system prompts, oklch color theory, curated font pairings, and anti-cliché rules.
triggers:
  - make this page look better
  - improve the design of this webpage
  - transform this into a beautiful UI
  - apply web design skill to this
  - generate a stunning landing page
  - use design system for this HTML
  - make the UI less generic
  - apply professional design to this component
---

# Web Design Engineer Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A reusable **SKILL.md** for AI coding agents (Claude Code, Cursor, Codex, etc.) that injects professional design taste into AI-generated HTML/CSS/JS artifacts. Inspired by the Claude Design system prompt, this skill transforms "functional but generic" output into editorial, high-quality visual work.

---

## What This Skill Does

When installed, this skill gives your AI agent:

- **Anti-cliché enforcement** — blocks overused AI design patterns (purple-pink gradients, emoji icons, Inter font, fake testimonials)
- **Design system declaration** — forces the agent to define tokens (colors, fonts, spacing, motion) in Markdown *before* writing code
- **oklch color theory** — perceptually uniform color derivation instead of arbitrary hex values
- **Curated font × color pairings** — six validated visual system starting points
- **Six-step structured workflow** — from requirements gathering to delivery verification
- **Placeholder philosophy** — `[icon]` markers instead of poorly drawn inline SVGs

---

## Installation

Copy the skill directory into your project:

```
your-project/
├── .agents/skills/web-design-engineer/
│   ├── SKILL.md                    # Main skill (~400 lines)
│   └── references/
│       └── advanced-patterns.md   # Code template library (~520 lines)
└── src/
    └── ...
```

For Claude Code specifically:
```
your-project/
└── .claude/skills/web-design-engineer/
    ├── SKILL.md
    └── references/
        └── advanced-patterns.md
```

Clone the repo and copy:

```bash
git clone https://github.com/ConardLi/web-design-skill.git
cp -r web-design-skill/.agents/skills/web-design-engineer your-project/.agents/skills/
```

Or for Claude Code:

```bash
cp -r web-design-skill/.agents/skills/web-design-engineer your-project/.claude/skills/
```

---

## The Six-Step Workflow

When triggered, the agent follows this structured process:

```
1. Understand requirements  →  Ask only when truly insufficient
2. Gather design context    →  Look at existing code/screenshots; never start blind
3. Declare design system    →  Colors, fonts, spacing, motion — in Markdown, before code
4. Show v0 draft early      →  Layout + tokens + placeholders; let user course-correct
5. Full build               →  Components, states, motion; pause at key decisions
6. Verify                   →  Pre-delivery checklist; no console errors, no rogue hues
```

---

## Core Design Principles

### oklch Color System

Colors must be derived in oklch space for perceptual uniformity. Same `L` value = same perceived brightness across hues.

```css
:root {
  /* Primary palette — blue-violet */
  --color-primary-50:  oklch(97% 0.02 270);
  --color-primary-100: oklch(93% 0.05 270);
  --color-primary-200: oklch(86% 0.10 270);
  --color-primary-300: oklch(76% 0.15 270);
  --color-primary-400: oklch(65% 0.20 270);
  --color-primary-500: oklch(55% 0.24 270);  /* brand */
  --color-primary-600: oklch(46% 0.22 270);
  --color-primary-700: oklch(38% 0.18 270);
  --color-primary-800: oklch(28% 0.13 270);
  --color-primary-900: oklch(18% 0.08 270);

  /* Neutrals — warm-tinted to avoid clinical grey */
  --color-neutral-50:  oklch(98% 0.005 80);
  --color-neutral-100: oklch(95% 0.008 80);
  --color-neutral-200: oklch(88% 0.010 80);
  --color-neutral-500: oklch(58% 0.010 80);
  --color-neutral-900: oklch(14% 0.008 80);

  /* Semantic */
  --color-bg:      var(--color-neutral-50);
  --color-surface: oklch(100% 0 0);
  --color-text:    var(--color-neutral-900);
  --color-muted:   var(--color-neutral-500);
  --color-accent:  var(--color-primary-500);
}
```

### Typography System

```css
/* Import — never use Inter, Roboto, Arial, system-ui, or Fraunces */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Newsreader:ital,opsz,wght@0,6..72,300..800;1,6..72,300..800&display=swap');

:root {
  --font-display: 'Newsreader', Georgia, serif;
  --font-body:    'Space Grotesk', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;

  /* Fluid type scale */
  --text-xs:   clamp(0.70rem, 0.65rem + 0.25vw, 0.75rem);
  --text-sm:   clamp(0.85rem, 0.80rem + 0.25vw, 0.875rem);
  --text-base: clamp(1.00rem, 0.95rem + 0.25vw, 1.0625rem);
  --text-lg:   clamp(1.15rem, 1.05rem + 0.50vw, 1.25rem);
  --text-xl:   clamp(1.30rem, 1.15rem + 0.75vw, 1.5rem);
  --text-2xl:  clamp(1.60rem, 1.35rem + 1.25vw, 2.00rem);
  --text-3xl:  clamp(2.00rem, 1.60rem + 2.00vw, 2.75rem);
  --text-4xl:  clamp(2.50rem, 1.90rem + 3.00vw, 3.75rem);
  --text-5xl:  clamp(3.00rem, 2.10rem + 4.50vw, 5.00rem);
}
```

### Spacing & Layout Tokens

```css
:root {
  /* 4px base grid */
  --space-1:  0.25rem;
  --space-2:  0.50rem;
  --space-3:  0.75rem;
  --space-4:  1.00rem;
  --space-6:  1.50rem;
  --space-8:  2.00rem;
  --space-12: 3.00rem;
  --space-16: 4.00rem;
  --space-24: 6.00rem;
  --space-32: 8.00rem;

  /* Layout */
  --max-w-content: 72rem;
  --max-w-prose:   65ch;
  --max-w-narrow:  42rem;

  /* Grid */
  --grid-cols-12: repeat(12, 1fr);
  --grid-gap:     clamp(1rem, 2vw, 2rem);

  /* Radius */
  --radius-sm:   0.25rem;
  --radius-md:   0.5rem;
  --radius-lg:   1.0rem;
  --radius-full: 9999px;

  /* Motion */
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in-out:   cubic-bezier(0.4, 0, 0.2, 1);
  --duration-fast: 150ms;
  --duration-base: 250ms;
  --duration-slow: 400ms;
}
```

---

## Curated Color × Font Pairings

| Style | Primary Color | Fonts | Best For |
|---|---|---|---|
| Modern tech | `oklch(55% 0.24 270)` blue-violet | Space Grotesk + Inter | SaaS, dev tools |
| Elegant editorial | `oklch(42% 0.12 55)` warm brown | Newsreader + Outfit | Content, blogs |
| Premium brand | `oklch(20% 0.04 270)` near-black | Sora + Plus Jakarta Sans | Luxury, finance |
| Lively consumer | `oklch(62% 0.22 25)` coral | Plus Jakarta Sans + Outfit | E-commerce, social |
| Minimal professional | `oklch(52% 0.14 200)` teal-blue | Outfit + Space Grotesk | Dashboards, B2B |
| Artisan warmth | `oklch(58% 0.14 65)` caramel | Caveat + Newsreader | Food, education |

---

## Anti-Cliché Checklist

Before delivering any artifact, the agent must verify it does NOT contain:

```
✗ Purple-to-pink-to-blue gradient backgrounds
✗ Left-border accent cards  
✗ Inter / Roboto / Arial / Fraunces / system-ui fonts
✗ Emoji used as icons (🚀 ✨ 💡 🎯 etc.)
✗ Fabricated statistics ("10,000+ users", "99.9% uptime")
✗ Fake logo walls ("Trusted by Google, Apple, Meta...")
✗ Dummy testimonials with generated avatar circles
✗ Rounded rectangles with drop shadows as the only card style
✗ Generic hero: big centered headline + subtitle + two buttons
✗ Hamburger menus that don't actually function
✗ Horizontal scrolling carousels on desktop
✗ "Glassmorphism" cards on gradient backgrounds
```

---

## Real Code Examples

### Design System Declaration (Step 3 output)

Before writing any code, the agent outputs a declaration like:

```markdown
## Design System: Nordic Photographer Portfolio

**Concept:** Paper-warm minimalism. Ink on vellum. Film-era restraint.

**Colors (oklch):**
- Background: oklch(96% 0.008 80)   — warm paper
- Surface:    oklch(100% 0 0)        — pure white panels  
- Ink:        oklch(14% 0.008 80)    — near-black
- Accent:     oklch(52% 0.18 200)    — muted teal
- Muted:      oklch(58% 0.008 80)    — mid-grey

**Typography:**
- Display: Instrument Serif, italic weight, optical size 32–72
- UI:      Space Grotesk 300–600
- Caption: Space Grotesk 300, tracked +0.08em

**Spacing:** 4px base, major sections 6rem–12rem vertical rhythm
**Motion:** 400ms ease-out-expo; Ken Burns hero at 24s cycle
**Radius:** 0 (zero) — no rounded corners; editorial flatness
```

### Component: Editorial Hero

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio — Mira Høst</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      oklch(96% 0.008 80);
      --ink:     oklch(14% 0.008 80);
      --accent:  oklch(52% 0.18 200);
      --muted:   oklch(58% 0.008 80);

      --font-display: 'Instrument Serif', Georgia, serif;
      --font-ui:      'Space Grotesk', sans-serif;

      --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
    }

    body {
      background: var(--bg);
      color: var(--ink);
      font-family: var(--font-ui);
      font-weight: 300;
      line-height: 1.6;
    }

    /* Navigation */
    .nav {
      position: fixed;
      top: 0; left: 0; right: 0;
      z-index: 100;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.5rem 3rem;
      mix-blend-mode: difference;
      color: white;
    }

    .nav__logo {
      font-family: var(--font-display);
      font-style: italic;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
    }

    .nav__links {
      display: flex;
      gap: 2.5rem;
      list-style: none;
      font-size: 0.8125rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .nav__links a {
      color: inherit;
      text-decoration: none;
    }

    /* Hero */
    .hero {
      position: relative;
      height: 100svh;
      overflow: hidden;
      display: grid;
      grid-template-columns: 1fr 1fr;
    }

    .hero__image {
      grid-column: 1 / -1;
      position: absolute;
      inset: 0;
      background: oklch(30% 0.02 200);
      /* Ken Burns */
      animation: kenburns 24s ease-in-out infinite alternate;
    }

    @keyframes kenburns {
      from { transform: scale(1.00) translate(0, 0); }
      to   { transform: scale(1.08) translate(-2%, -1%); }
    }

    .hero__content {
      position: relative;
      z-index: 2;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      padding: 4rem 3rem;
      grid-column: 1 / 2;
      color: oklch(96% 0.008 80);
    }

    .hero__eyebrow {
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      opacity: 0.7;
      margin-bottom: 1rem;
    }

    .hero__title {
      font-family: var(--font-display);
      font-style: italic;
      font-size: clamp(3rem, 6vw, 5.5rem);
      line-height: 1.0;
      letter-spacing: -0.02em;
      margin-bottom: 1.5rem;
    }

    .hero__meta {
      display: flex;
      align-items: center;
      gap: 2rem;
      font-size: 0.8125rem;
      opacity: 0.65;
    }

    .hero__counter {
      position: absolute;
      bottom: 4rem;
      right: 3rem;
      z-index: 2;
      color: oklch(96% 0.008 80);
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      opacity: 0.5;
    }

    /* Section rail */
    .section {
      display: grid;
      grid-template-columns: 1fr 3fr;
      gap: 3rem;
      padding: 6rem 3rem;
      border-top: 1px solid oklch(80% 0.008 80);
    }

    .section__label {
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      padding-top: 0.25rem;
    }

    .section__number {
      font-family: var(--font-display);
      font-style: italic;
      font-size: 0.875rem;
      color: var(--accent);
      display: block;
      margin-bottom: 0.5rem;
    }
  </style>
</head>
<body>

  <nav class="nav">
    <div class="nav__logo">Mira Høst</div>
    <ul class="nav__links">
      <li><a href="#work">Work</a></li>
      <li><a href="#about">About</a></li>
      <li><a href="#contact">Contact</a></li>
    </ul>
  </nav>

  <section class="hero">
    <div class="hero__image"></div>
    <div class="hero__content">
      <p class="hero__eyebrow">Documentary · Landscape · Portrait</p>
      <h1 class="hero__title">Light<br>finds<br>form.</h1>
      <div class="hero__meta">
        <span>Oslo, Norway</span>
        <span>Est. 2014</span>
      </div>
    </div>
    <div class="hero__counter">001 / 012</div>
  </section>

  <section class="section" id="work">
    <div class="section__label">
      <span class="section__number">01</span>
      Selected Work
    </div>
    <div>
      <!-- [photo grid] -->
    </div>
  </section>

</body>
</html>
```

### Component: oklch Dashboard Card

```css
.card {
  background: oklch(100% 0 0);
  border: 1px solid oklch(90% 0.008 80);
  padding: var(--space-6);
  /* No border-radius unless explicitly designed */
  border-radius: var(--radius-md);
  transition: box-shadow var(--duration-base) var(--ease-out-expo),
              transform    var(--duration-base) var(--ease-out-expo);
}

.card:hover {
  box-shadow: 0 8px 32px oklch(14% 0.008 80 / 0.08);
  transform: translateY(-2px);
}

.card__metric {
  font-family: var(--font-display);
  font-size: var(--text-4xl);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.03em;
  color: var(--color-accent);
  line-height: 1;
}

.card__label {
  font-size: var(--text-sm);
  color: var(--color-muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: var(--space-2);
}
```

### Placeholder Philosophy

```html
<!-- CORRECT: honest placeholders -->
<div class="icon-slot">[icon: arrow-right]</div>
<div class="avatar-slot">[photo: team member Sarah]</div>
<figure class="image-slot">[image: hero — coastal landscape, wide format]</figure>

<!-- WRONG: never do this -->
<svg><!-- badly drawn attempt at an icon --></svg>
<div style="background: #ccc; border-radius: 50%; width: 40px; height: 40px;"></div>
```

---

## Supported Output Types

| Type | Description |
|---|---|
| Web pages & landing pages | Marketing sites, product pages, portfolios |
| Interactive prototypes | Clickable mockups with device frames |
| Slide decks | HTML presentations (1920×1080, keyboard nav) |
| Data visualizations | Dashboards with Chart.js or D3.js |
| Animations | CSS/JS motion design, timeline-driven demos |
| Design systems | Token exploration, component variants |

---

## Troubleshooting

**Agent ignores the skill and produces generic output**
- Confirm the skill file is in the correct directory for your tool (`.agents/skills/` vs `.claude/skills/`)
- Explicitly reference the skill in your prompt: "Using the web-design-engineer skill, create..."
- Ensure `SKILL.md` is not empty or truncated

**oklch colors not rendering**
- oklch is supported in Chrome 111+, Firefox 113+, Safari 16.4+
- Add a hex fallback for older browsers:
```css
color: #4f46e5; /* fallback */
color: oklch(55% 0.24 270);
```

**Fonts not loading in self-contained HTML files**
- Ensure internet access for Google Fonts CDN, or self-host via `@font-face`
- The agent should always include `rel="preconnect"` for performance

**Design system declaration step is skipped**
- Remind the agent: "Before writing code, declare the design system in Markdown"
- This is Step 3 of the workflow and must happen before Step 4

**Agent uses emoji as icons despite the rule**
- The anti-cliché list explicitly bans this; re-invoke with: "Do not use emoji as icons — use `[icon: name]` placeholders instead"

---

## File Reference

```
web-design-skill/
├── .agents/skills/web-design-engineer/
│   ├── SKILL.md                    # Main skill definition (~400 lines)
│   └── references/
│       └── advanced-patterns.md   # Code templates & patterns (~520 lines)
├── demo/
│   ├── demo1.html                  # Space museum — without skill
│   ├── demo1-with-skill.html       # Space museum — with skill
│   └── demo2-with-skill.html       # Photographer portfolio — with skill
└── prompt/
    └── system.md                   # Claude Design reference system prompt
```

---

## License

MIT — freely copy, modify, and embed in any project.
```
