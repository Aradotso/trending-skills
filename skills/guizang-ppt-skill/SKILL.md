```markdown
---
name: guizang-ppt-skill
description: Generate single-file HTML magazine-style horizontal swipe PPT presentations with WebGL fluid backgrounds, editorial typography, and 10 layout templates
triggers:
  - "make a magazine style PPT"
  - "generate a horizontal swipe presentation"
  - "create an editorial magazine slide deck"
  - "帮我做一份杂志风 PPT"
  - "生成电子杂志风格演示文稿"
  - "electronic ink style presentation slides"
  - "create a single file HTML presentation"
  - "build a web PPT with WebGL background"
---

# Magazine Web PPT (guizang-ppt-skill)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code / Agent skill for generating **single-file HTML horizontal-swipe PPT** presentations with an "electronic magazine × electronic ink" aesthetic — like *Monocle* magazine with code.

## What This Skill Does

- Generates a **single `.html` file** — no build step, no server, open directly in browser
- **WebGL fluid/dispersion background** visible on hero pages, restrained on content pages
- **Horizontal left-right pagination**: keyboard ← → / scroll wheel / touch swipe / bottom dots / ESC index
- **5 color theme presets**: Ink Classic / Indigo Porcelain / Forest Ink / Kraft Paper / Dune
- **10 page layout templates**: Cover, Chapter, Data Headline, Left-Text-Right-Image, Image Grid, Pipeline, Question, Big Quote, Before/After, Mixed Layout
- **Triple font hierarchy**: serif headlines + sans-serif body + monospace metadata

## Installation

### Method 1: Ask Your AI Agent

```
帮我安装 guizang-ppt-skill 这个 Claude Code skill。请按下面步骤做:
1. 确保 ~/.claude/skills/ 目录存在(不存在就创建)
2. 执行 git clone https://github.com/op7418/guizang-ppt-skill.git ~/.claude/skills/magazine-web-ppt
3. 验证: ls ~/.claude/skills/magazine-web-ppt/ 应该看到 SKILL.md、assets/、references/ 三项
4. 告诉我安装好了
```

### Method 2: Manual

```bash
git clone https://github.com/op7418/guizang-ppt-skill.git ~/.claude/skills/magazine-web-ppt
ls ~/.claude/skills/magazine-web-ppt/
# Should see: SKILL.md  assets/  references/
```

## Repository Structure

```
magazine-web-ppt/
├── SKILL.md              ← Skill main file: workflow, principles, common errors
├── README.md
├── assets/
│   └── template.html     ← Complete runnable seed HTML (CSS + WebGL + pagination JS)
└── references/
    ├── components.md     ← Component manual (fonts, colors, grid, icons, callout, stat, pipeline)
    ├── layouts.md        ← 10 page layout skeletons (paste-ready)
    ├── themes.md         ← 5 color theme presets (choose, don't customize)
    └── checklist.md      ← Quality checklist (P0 / P1 / P2 / P3 graded)
```

## 6-Step Workflow

### Step 1: Requirements Clarification (6 Questions)

Ask the user:
1. **Audience** — Who's in the room? (investors, peers, public)
2. **Duration** — How many minutes? (determines slide count ~1 slide/min)
3. **Content** — Raw materials: outline, doc, bullet points?
4. **Images** — URLs provided or use placeholder patterns?
5. **Theme** — Which of the 5 presets? (default: Ink Classic)
6. **Hard constraints** — Max slides, forbidden content, language?

### Step 2: Copy Template

```bash
cp ~/.claude/skills/magazine-web-ppt/assets/template.html ./my-presentation.html
```

Then edit the top of the file:
- Change `<title>Your Title</title>`
- Replace the `:root{}` theme variables (6 lines only)

### Step 3: Fill Content from Layouts

Read `references/layouts.md` for the 10 skeleton templates. Copy paste and fill.

**Pre-fill checklist:**
- [ ] Run class name pre-check: every class used in layout HTML must exist in `template.html` CSS
- [ ] Plan hero/non-hero rhythm before writing (hero pages ~every 3-4 slides)

### Step 4: Self-Check

Open `references/checklist.md` and verify all **P0** items pass before delivering.

### Step 5: Preview

```bash
open my-presentation.html
# or
python3 -m http.server 8080
# then visit http://localhost:8080/my-presentation.html
```

### Step 6: Iterate

Use inline styles for per-slide tweaks:

```html
<!-- Adjust font size -->
<h1 style="font-size: 4.5rem;">Bigger Title</h1>

<!-- Adjust slide height constraint -->
<div class="slide" style="--slide-padding: 3rem;">

<!-- Tighten line height -->
<p style="line-height: 1.4;">Dense paragraph</p>
```

## Theme Presets

**Never use custom hex values.** Only select from these 5 presets in `references/themes.md`.

| Theme | CSS Variables Snippet | Best For |
|-------|----------------------|----------|
| 🖋 Ink Classic | `--bg: #F5F2ED; --fg: #1A1A1A; --accent: #2C2C2C` | Default, business, launches |
| 🌊 Indigo Porcelain | `--bg: #EEF2F7; --fg: #1B2A4A; --accent: #2D5BE3` | Tech, AI, research |
| 🌿 Forest Ink | `--bg: #F0F4F0; --fg: #1A2E1A; --accent: #2D6A2D` | Nature, sustainability, culture |
| 🍂 Kraft Paper | `--bg: #F4EDE0; --fg: #2E1F0E; --accent: #8B4513` | Retro, humanities, indie |
| 🌙 Dune | `--bg: #F5F0E8; --fg: #2A1F0A; --accent: #C4860A` | Art, design, gallery |

To switch themes, replace only the `:root {}` block at the top of `template.html`:

```css
:root {
  --bg: #EEF2F7;
  --fg: #1B2A4A;
  --accent: #2D5BE3;
  --accent-light: #E8EDFB;
  --muted: #6B7A99;
  --border: #C9D3E8;
}
```

## The 10 Layout Templates

Paste these from `references/layouts.md` — shown here with key structure:

### 1. Hero Cover (开场封面)

```html
<section class="slide hero" data-slide="1">
  <canvas class="webgl-bg"></canvas>
  <div class="slide-content cover-layout">
    <div class="eyebrow mono">VOL.01 · 2026</div>
    <h1 class="display-title">Your Big<br>Main Title</h1>
    <p class="subtitle">Subtitle or tagline goes here</p>
    <div class="meta-row">
      <span class="mono">SPEAKER NAME</span>
      <span class="mono">EVENT · DATE</span>
    </div>
  </div>
</section>
```

### 2. Chapter Curtain (章节幕封)

```html
<section class="slide hero chapter" data-slide="2">
  <div class="chapter-number mono">02</div>
  <div class="chapter-content">
    <h2 class="chapter-title">Chapter Title</h2>
    <p class="chapter-sub">Brief chapter description</p>
  </div>
</section>
```

### 3. Data Headline (数据大字报)

```html
<section class="slide data-hero" data-slide="3">
  <div class="slide-content">
    <div class="stat-block">
      <span class="stat-number display-title">73%</span>
      <span class="stat-label">of something important</span>
    </div>
    <p class="stat-context">Supporting context sentence explaining the data point above.</p>
  </div>
</section>
```

### 4. Left Text Right Image (左文右图)

```html
<section class="slide split-layout" data-slide="4">
  <div class="split-left">
    <div class="eyebrow mono">SECTION LABEL</div>
    <h2>Section Heading</h2>
    <p>Body copy goes here. Keep it to 3-4 lines maximum for readability at presentation distance.</p>
    <ul class="bullet-list">
      <li>Key point one</li>
      <li>Key point two</li>
      <li>Key point three</li>
    </ul>
  </div>
  <div class="split-right">
    <img src="https://example.com/image.jpg" alt="Description" class="slide-image">
  </div>
</section>
```

### 5. Image Grid (图片网格)

```html
<section class="slide grid-layout" data-slide="5">
  <div class="grid-header">
    <h2>Grid Section Title</h2>
  </div>
  <div class="image-grid grid-2x2">
    <figure class="grid-item">
      <img src="img1.jpg" alt="Item 1">
      <figcaption class="mono">CAPTION ONE</figcaption>
    </figure>
    <figure class="grid-item">
      <img src="img2.jpg" alt="Item 2">
      <figcaption class="mono">CAPTION TWO</figcaption>
    </figure>
    <figure class="grid-item">
      <img src="img3.jpg" alt="Item 3">
      <figcaption class="mono">CAPTION THREE</figcaption>
    </figure>
    <figure class="grid-item">
      <img src="img4.jpg" alt="Item 4">
      <figcaption class="mono">CAPTION FOUR</figcaption>
    </figure>
  </div>
</section>
```

### 6. Pipeline / Process (流程图)

```html
<section class="slide pipeline-layout" data-slide="6">
  <h2>Process Title</h2>
  <div class="pipeline">
    <div class="pipeline-step">
      <div class="step-number mono">01</div>
      <div class="step-content">
        <h3>Step One</h3>
        <p>Brief description</p>
      </div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
      <div class="step-number mono">02</div>
      <div class="step-content">
        <h3>Step Two</h3>
        <p>Brief description</p>
      </div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
      <div class="step-number mono">03</div>
      <div class="step-content">
        <h3>Step Three</h3>
        <p>Brief description</p>
      </div>
    </div>
  </div>
</section>
```

### 7. Suspense Question (悬念问题)

```html
<section class="slide hero question-slide" data-slide="7">
  <div class="slide-content centered">
    <div class="question-mark mono">?</div>
    <h2 class="question-text">What if everything<br>you knew was wrong?</h2>
    <p class="question-sub mono">We're about to find out.</p>
  </div>
</section>
```

### 8. Big Quote (大引用)

```html
<section class="slide quote-slide" data-slide="8">
  <div class="slide-content">
    <blockquote class="big-quote">
      <p>"The best way to predict the future is to invent it."</p>
      <footer class="quote-attribution mono">— ALAN KAY · 1971</footer>
    </blockquote>
  </div>
</section>
```

### 9. Before / After Comparison

```html
<section class="slide compare-layout" data-slide="9">
  <h2>Before vs After</h2>
  <div class="compare-grid">
    <div class="compare-panel before">
      <div class="panel-label mono">BEFORE</div>
      <ul class="compare-list negative">
        <li>Pain point one</li>
        <li>Pain point two</li>
        <li>Pain point three</li>
      </ul>
    </div>
    <div class="compare-divider"></div>
    <div class="compare-panel after">
      <div class="panel-label mono accent">AFTER</div>
      <ul class="compare-list positive">
        <li>Improvement one</li>
        <li>Improvement two</li>
        <li>Improvement three</li>
      </ul>
    </div>
  </div>
</section>
```

### 10. Mixed Text + Image (图文混排)

```html
<section class="slide mixed-layout" data-slide="10">
  <div class="mixed-header">
    <h2>Mixed Layout Title</h2>
    <p class="lead">Opening paragraph that frames the content below.</p>
  </div>
  <div class="mixed-body">
    <img src="wide-image.jpg" alt="Main visual" class="mixed-image">
    <div class="mixed-caption">
      <p>Caption or supplementary text explaining the image above.</p>
      <span class="mono">SOURCE · 2026</span>
    </div>
  </div>
</section>
```

## Core CSS Classes Reference

From `references/components.md`:

```css
/* Typography */
.display-title    /* Serif, hero-scale (5-8rem) headline */
.mono             /* Monospace, metadata/labels/numbers */
.eyebrow          /* Small mono label above heading */
.lead             /* Large sans-serif intro paragraph */
.subtitle         /* Secondary heading text */

/* Layout */
.slide            /* Base slide container (100vw × 100vh) */
.slide.hero       /* Hero variant — WebGL bg visible */
.slide-content    /* Inner content wrapper with padding */
.split-layout     /* Two-column 50/50 or 40/60 */
.centered         /* Flex center both axes */

/* Components */
.stat-number      /* Giant number display */
.stat-label       /* Label under stat number */
.big-quote        /* Blockquote with editorial styling */
.callout          /* Highlighted aside box */
.pipeline         /* Horizontal step process */
.bullet-list      /* Styled unordered list */
.image-grid       /* CSS grid image container */
.compare-grid     /* Two-panel before/after */

/* Navigation (auto-generated) */
.slide-dots       /* Bottom pagination dots */
.slide-index      /* ESC overlay with all slides */
```

## Navigation Controls

Built into `template.html` — no configuration needed:

| Input | Action |
|-------|--------|
| `→` / `Space` | Next slide |
| `←` | Previous slide |
| Mouse wheel down | Next slide |
| Mouse wheel up | Previous slide |
| Touch swipe left | Next slide |
| Touch swipe right | Previous slide |
| `ESC` | Toggle slide index overlay |
| Click dot | Jump to slide |

## Image Handling Rules

**Critical:** Images crop from bottom only — top and sides must be compositionally complete.

```html
<!-- Correct: object-fit cover with top anchor -->
<img src="photo.jpg" class="slide-image" 
     style="object-fit: cover; object-position: top center;">

<!-- For portrait images in landscape slides -->
<img src="portrait.jpg" class="slide-image"
     style="object-fit: cover; object-position: center top; height: 100%;">
```

If no images are provided, use structured placeholder patterns:

```html
<!-- Placeholder with label -->
<div class="img-placeholder" style="background: var(--accent-light); 
     aspect-ratio: 16/9; display:flex; align-items:center; 
     justify-content:center;">
  <span class="mono" style="color: var(--muted);">IMAGE · 16:9</span>
</div>
```

## Quality Checklist (P0 — Must Pass)

From `references/checklist.md`:

- [ ] All CSS classes used in HTML exist in `template.html` — no undefined classes
- [ ] No slide exceeds viewport height (no scrollbar appears within a slide)
- [ ] Hero/non-hero rhythm maintained (not 5+ non-hero slides in a row)
- [ ] All `<img>` tags have `alt` attributes
- [ ] WebGL canvas only present on `.hero` slides
- [ ] Font stack loads: serif for display, sans for body, mono for metadata
- [ ] Navigation works: keyboard, scroll, touch, dots all functional
- [ ] Theme uses only preset variables — no raw hex values in content HTML
- [ ] `<title>` updated from template default
- [ ] File opens correctly with `file://` protocol (no server required)

## Common Errors & Fixes

### Slide Content Overflows

```html
<!-- Problem: too much text in one slide -->
<!-- Fix: reduce content OR add font-size override -->
<div class="slide-content" style="font-size: 0.9em;">

<!-- Or split into two slides -->
```

### WebGL Background Not Showing

```html
<!-- Canvas must be direct child of .hero slide -->
<section class="slide hero">
  <canvas class="webgl-bg"></canvas>  <!-- ← must be here -->
  <div class="slide-content">...</div>
</section>
```

### Classes Not Styled

```
ERROR: Using .grid-3col but template only has .grid-2x2 and .grid-3x1
FIX: Check references/components.md for available classes, or add inline styles
```

### Images Cropping Wrong

```html
<!-- Fix portrait image in landscape slide -->
<img src="tall.jpg" style="object-fit:cover; object-position: top; 
     width:100%; height:100%;">
```

### Dots Not Updating

The pagination dots are auto-generated from `[data-slide]` attributes. Ensure every `<section class="slide">` has `data-slide="N"` with sequential numbers starting from 1.

```html
<section class="slide hero" data-slide="1">
<section class="slide" data-slide="2">
<section class="slide" data-slide="3">
```

## Design Principles

1. **Restraint over flashiness** — WebGL only on hero slides
2. **Structure over decoration** — hierarchy via type scale + grid whitespace, not shadows/cards
3. **Images are first-class** — crop bottom only, keep top/sides compositionally whole
4. **Rhythm via hero slides** — alternate hero/non-hero, ~every 3-4 slides
5. **Theme lock** — 5 presets only, no custom hex, aesthetic protection > freedom

## Slide Count Guidelines

| Talk Length | Suggested Slides |
|-------------|-----------------|
| 5 min | 5-7 slides |
| 10 min | 10-12 slides |
| 20 min | 18-22 slides |
| 30 min | 25-30 slides |

Rule of thumb: 1 slide per minute, with hero/chapter slides as breathing room.

## Full Minimal Example

Complete working 3-slide presentation:

```html
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Magazine PPT</title>
<style>
/* Copy full CSS from assets/template.html */
:root {
  --bg: #F5F2ED;
  --fg: #1A1A1A;
  --accent: #2C2C2C;
  --accent-light: #ECEAE6;
  --muted: #888580;
  --border: #D8D4CE;
}
/* ... rest of template CSS ... */
</style>
</head>
<body>

<!-- Slide 1: Hero Cover -->
<section class="slide hero" data-slide="1">
  <canvas class="webgl-bg"></canvas>
  <div class="slide-content cover-layout">
    <div class="eyebrow mono">VOL.01 · 2026</div>
    <h1 class="display-title">The Future<br>of Work</h1>
    <p class="subtitle">How AI is folding organizations</p>
    <div class="meta-row">
      <span class="mono">GUIZANG</span>
      <span class="mono">SHANGHAI · APR 2026</span>
    </div>
  </div>
</section>

<!-- Slide 2: Data Headline -->
<section class="slide data-hero" data-slide="2">
  <div class="slide-content">
    <div class="stat-block">
      <span class="stat-number display-title">1人</span>
      <span class="stat-label">可以做到过去10人的工作</span>
    </div>
    <p class="stat-context">AI agent 的出现让个体效能产生了指数级变化。</p>
  </div>
</section>

<!-- Slide 3: Big Quote -->
<section class="slide quote-slide" data-slide="3">
  <div class="slide-content">
    <blockquote class="big-quote">
      <p>"The company of one is the company of the future."</p>
      <footer class="quote-attribution mono">— A NEW WORKING METHOD · 2026</footer>
    </blockquote>
  </div>
</section>

<!-- Navigation dots auto-generated by JS -->
<div class="slide-dots"></div>

<script>
/* Copy full JS from assets/template.html */
</script>
</body>
</html>
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Blank white page | Check browser console; likely a JS error in template |
| Fonts look wrong | Ensure Google Fonts link in `<head>` from template |
| Can't advance slides | Verify `data-slide` attributes are sequential from 1 |
| WebGL flickering | Only one `<canvas class="webgl-bg">` per hero slide |
| File too large | Compress images before embedding as data URIs |
| Slide content cut off | Reduce font size or split slide; never scroll within slide |

## Resources

- Full template: `~/.claude/skills/magazine-web-ppt/assets/template.html`
- All 10 layouts: `~/.claude/skills/magazine-web-ppt/references/layouts.md`
- Component library: `~/.claude/skills/magazine-web-ppt/references/components.md`
- Theme presets: `~/.claude/skills/magazine-web-ppt/references/themes.md`
- Quality checklist: `~/.claude/skills/magazine-web-ppt/references/checklist.md`
- GitHub: https://github.com/op7418/guizang-ppt-skill
- Creator: [@op7418](https://x.com/op7418)
```
