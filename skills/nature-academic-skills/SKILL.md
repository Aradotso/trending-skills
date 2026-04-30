```markdown
---
name: nature-academic-skills
description: Generate publication-ready Nature-journal matplotlib figures and polish academic prose to Nature style standards using Claude skills.
triggers:
  - "create a Nature figure"
  - "make a publication-ready plot"
  - "polish this academic writing to Nature style"
  - "scientific figure for my paper"
  - "Nature journal manuscript polishing"
  - "multi-panel matplotlib figure"
  - "academic prose editing Nature standard"
  - "convert draft to Nature style writing"
---

# Nature Academic Skills

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A collection of Claude skills for producing academic work at *Nature*-journal standard — covering scientific figures (`nature-figure`) and manuscript prose polishing (`nature-polishing`).

---

## What This Project Does

`nature-skills` provides two stable skills that enforce rules derived from **primary sources** (published *Nature* papers, official author guidelines, structured writing curricula):

| Skill | Purpose |
|-------|---------|
| `nature-figure` | Multi-panel matplotlib figures matching *Nature* visual standards |
| `nature-polishing` | Academic prose polishing to *Nature* prose conventions |

---

## Installation

### For Claude Code / Cursor / Codex agents

Clone the repository into your project's `.claude/skills/` or equivalent skills directory:

```bash
git clone https://github.com/Yuan1z0825/nature-skills.git .claude/skills/nature-skills
```

Or copy the relevant `SKILL.md` files directly into your agent's context:

```bash
# For figure generation only
cp nature-skills/nature-figure/SKILL.md .claude/skills/nature-figure.md

# For prose polishing only
cp nature-skills/nature-polishing/SKILL.md .claude/skills/nature-polishing.md
```

The agent will automatically load `SKILL.md` files from its skills directory and activate the appropriate skill based on trigger keywords.

### Python dependencies (for nature-figure)

```bash
pip install matplotlib numpy scipy
```

Optional for SVG post-processing:

```bash
pip install cairosvg   # SVG → PDF conversion
pip install svgutils   # panel assembly
```

---

## Skill 1: nature-figure

### Trigger phrases

"Nature figure", "publication plot", "scientific figure", "multi-panel figure"

### Mandatory rcParams (always include first)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# REQUIRED: must appear before any figure creation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'  # text stays as <text> nodes, not paths
```

### Output policy

```python
# Primary output: SVG (always)
fig.savefig('figure1.svg', bbox_inches='tight', dpi=300)

# Secondary output: PNG raster preview (always include alongside SVG)
fig.savefig('figure1.png', bbox_inches='tight', dpi=300)
```

### Nature colour palette

```python
NATURE_PALETTE = {
    'blue':       '#4878CF',
    'green':      '#6ACC65',
    'red':        '#D65F5F',
    'purple':     '#B47CC7',
    'cyan':       '#77BEDB',
    'orange':     '#EE854A',
    'pink':       '#D0759F',
    'yellow':     '#C4AD66',
    'light_blue': '#8ABBE5',
    'dark_green': '#3A9E5F',
}

COLORS = list(NATURE_PALETTE.values())
```

### Typography rules

| Element | Size | Weight |
|---------|------|--------|
| Panel label (a, b, c…) | 8 pt | bold |
| Axis title | 7 pt | normal |
| Tick labels | 6 pt | normal |
| Legend text | 6 pt | normal |
| Figure title (if any) | 8 pt | bold |

```python
FONT_SIZES = {
    'panel_label': 8,
    'axis_title':  7,
    'tick_label':  6,
    'legend':      6,
}
```

### Complete multi-panel figure example

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- Mandatory rcParams ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

NATURE_PALETTE = {
    'blue':   '#4878CF',
    'green':  '#6ACC65',
    'red':    '#D65F5F',
    'orange': '#EE854A',
}
COLORS = list(NATURE_PALETTE.values())

# --- Figure layout (Nature single-column: 89 mm; double: 183 mm) ---
fig = plt.figure(figsize=(7.2, 4.0))  # 183 mm wide ≈ 7.2 inches
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45, hspace=0.4)

# Panel a: bar chart (overview)
ax_a = fig.add_subplot(gs[0, 0])
categories = ['Control', 'Treatment A', 'Treatment B']
values = [0.42, 0.67, 0.81]
errors = [0.05, 0.04, 0.06]
bars = ax_a.bar(categories, values, color=COLORS[:3],
                width=0.6, linewidth=0.8, edgecolor='white')
ax_a.errorbar(categories, values, yerr=errors,
              fmt='none', color='black', capsize=3, linewidth=0.8)
ax_a.set_ylabel('Accuracy', fontsize=7)
ax_a.set_ylim(0, 1.0)
ax_a.tick_params(labelsize=6)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.18, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=8, fontweight='bold', va='top')

# Panel b: trend lines (deviation)
ax_b = fig.add_subplot(gs[0, 1])
epochs = np.arange(1, 51)
for i, label in enumerate(['Model A', 'Model B', 'Model C']):
    loss = 1.0 * np.exp(-0.08 * epochs) + 0.05 * np.random.randn(50) * 0
    loss = 1.0 * np.exp(-0.08 * epochs) + i * 0.05
    ax_b.plot(epochs, loss, color=COLORS[i], linewidth=1.2, label=label)
ax_b.set_xlabel('Epoch', fontsize=7)
ax_b.set_ylabel('Loss', fontsize=7)
ax_b.tick_params(labelsize=6)
ax_b.legend(fontsize=6, frameon=False, loc='upper right')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.18, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=8, fontweight='bold', va='top')

# Panel c: scatter (relationship)
ax_c = fig.add_subplot(gs[0, 2])
np.random.seed(42)
x = np.random.randn(60)
y = 0.7 * x + 0.5 * np.random.randn(60)
ax_c.scatter(x, y, color=COLORS[0], alpha=0.7, s=18,
             linewidths=0.3, edgecolors='white')
m, b = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax_c.plot(xline, m * xline + b, color=COLORS[2], linewidth=1.2, linestyle='--')
ax_c.set_xlabel('Feature score', fontsize=7)
ax_c.set_ylabel('Outcome', fontsize=7)
ax_c.tick_params(labelsize=6)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.text(-0.18, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=8, fontweight='bold', va='top')

plt.savefig('figure1.svg', bbox_inches='tight', dpi=300)
plt.savefig('figure1.png', bbox_inches='tight', dpi=300)
plt.show()
```

### Supported chart types

| Type | Use case |
|------|----------|
| Stacked / grouped bar | Comparing categories with subgroups |
| Horizontal ablation bar | Ablation studies, feature importance |
| Trend / line | Training curves, time-series |
| Sequential heatmap | Expression matrices, correlation |
| Diverging z-score heatmap | Z-score, signed deviation from mean |
| Bubble scatter | Three-variable relationships |
| Radar / polar | Multi-metric model comparison |
| 3D sphere illustration | Conceptual/anatomical diagrams |
| Fill-between area | Confidence intervals, variance bands |
| Log-scale bar | Dynamic-range comparisons |
| GridSpec multi-panel | Combined overview figures |

### Three-level panel information hierarchy

```
Overview  →  Deviation  →  Relationship
   (a)           (b)            (c)
```

**Rule:** No two panels may answer the same scientific question.

---

## Skill 2: nature-polishing

### Trigger phrases

"Nature style", "polish", "academic writing", "manuscript editing"

### 12-step polishing workflow

```
1. Sentence split       — Split into individual sentences; count words each
2. Section ID           — Identify section: Abstract / Intro / Results / Discussion / Methods
3. Hourglass check      — Verify structure follows broad → specific → broad
4. Tense audit          — Results = past; Discussion = hedging present; Methods = past
5. Sentence edit        — Enforce ≤ 30 words per sentence
6. Vocabulary upgrade   — Replace weak/vague terms with precise scientific vocabulary
7. Template check       — Remove formulaic openers ("In this study, we…")
8. Citation audit       — Cite only verified sources; use correct attribution type
9. House style          — British English, journal formatting conventions
10. Overclaim detection — Flag absolutes, unwarranted causation, unverified "first" claims
11. Proofreading        — Final grammar, punctuation, consistency check
12. Plain-text output   — Return clean prose only, no markup
```

### Core rules quick reference

| Domain | Rule |
|--------|------|
| Sentence length | Every sentence ≤ 30 words; count each individually |
| Hedging | *demonstrate* (strong evidence) → *suggest* → *may reflect* (weak evidence) |
| Results tense | Past tense + quantitative detail ("increased by 23%", "P = 0.003") |
| Discussion tense | Hedging present tense + mechanism ("This may reflect…") |
| Citation integrity | Cite only sources personally read and verified |
| Overclaim | Flag: absolutes, unwarranted causation, scope expansion, unverified "first" |
| British English | signalling, colour, analyse, programme, modelling, behaviour |
| Vocabulary | Replace: "get" → "obtain", "big" → "substantial", "shows" → "demonstrates" |

### Hedging vocabulary scale

```
Strong  ──────────────────────────────────────  Weak
demonstrate  →  show  →  suggest  →  indicate  →  may reflect  →  is consistent with
```

### Sentence length enforcement

```python
def check_sentence_length(text: str) -> list[dict]:
    """Flag sentences exceeding 30 words."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    results = []
    for i, sent in enumerate(sentences, 1):
        words = len(sent.split())
        results.append({
            'index': i,
            'words': words,
            'over_limit': words > 30,
            'sentence': sent[:80] + ('…' if len(sent) > 80 else '')
        })
    return [r for r in results if r['over_limit']]

# Usage
violations = check_sentence_length(my_paragraph)
for v in violations:
    print(f"Sentence {v['index']}: {v['words']} words — OVER LIMIT")
    print(f"  → {v['sentence']}")
```

### Example: before / after polishing

**Before (draft):**
> In this study, we developed a new deep learning model that can very effectively identify cancer biomarkers from genomic data, which is a very important problem in the field of precision medicine, and our results show that it works much better than all existing methods.

**After (Nature style):**
> We developed a deep learning model for cancer biomarker identification from genomic data. The model achieved an area under the curve of 0.94 (95% CI 0.91–0.97), surpassing all benchmarked methods by a margin of 8–14 percentage points. These findings suggest a pathway towards improved precision oncology screening.

Changes applied:
- Removed "In this study, we" opener
- Split 54-word sentence into three ≤ 30-word sentences
- Added quantitative detail (AUC, CI)
- Replaced "much better than all" with hedged quantified comparison
- Replaced "works" with "achieved"; "show" with "suggest"

---

## Common Patterns

### Pattern 1: Ultra-wide single panel (Nature Methods style)

```python
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(7.2, 2.5))
# ... plot content ...
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('figure_wide.svg', bbox_inches='tight', dpi=300)
fig.savefig('figure_wide.png', bbox_inches='tight', dpi=300)
```

### Pattern 2: Legend as separate axis (no overlap)

```python
fig = plt.figure(figsize=(7.2, 3.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[3, 3, 3, 1])
ax_leg = fig.add_subplot(gs[0, 3])
ax_leg.axis('off')
handles = [mpl.patches.Patch(color=c, label=l)
           for c, l in zip(COLORS, labels)]
ax_leg.legend(handles=handles, loc='center left',
              fontsize=6, frameon=False)
```

### Pattern 3: Diverging heatmap (z-score)

```python
import matplotlib.colors as mcolors

data = np.random.randn(8, 12)  # pre-computed z-scores
vmax = np.abs(data).max()
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Z-score', pad=0.02)
```

### Pattern 4: Prose polishing prompt template

```
Polish the following [Results / Discussion / Methods] section text to Nature journal style.
Apply all 12 steps: sentence splitting, tense audit, ≤30-word limit, hedging calibration,
vocabulary upgrade, British English, and overclaim detection.
Return only the polished plain text.

---
[PASTE TEXT HERE]
---
```

---

## Troubleshooting

### Figure text renders as paths in SVG

**Cause:** `svg.fonttype` not set before figure creation.

```python
# Fix: set BEFORE any plt call
plt.rcParams['svg.fonttype'] = 'none'
```

### Arial not found — fallback to DejaVu Sans

```python
# Correct fallback chain (already in mandatory rcParams)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
# DejaVu Sans is always available in matplotlib; visually similar to Arial
```

### Sentence word count — last sentence most likely to fail

The 12-step workflow notes: **the last sentence of a paragraph is the most likely to exceed 30 words.** Always count it explicitly.

### Heatmap colour scale not centred at zero

```python
# Use TwoSlopeNorm, not just vmin/vmax
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
```

### SVG file too large

```python
# Reduce embedded raster elements; avoid imshow in SVG for large matrices
# Use PNG for heatmaps > 50×50 cells
if data.shape[0] * data.shape[1] > 2500:
    fig.savefig('heatmap.png', bbox_inches='tight', dpi=300)
else:
    fig.savefig('heatmap.svg', bbox_inches='tight', dpi=300)
```

---

## Reference Files (for agents with file access)

```
nature-figure/
├── SKILL.md                   # Loaded automatically by agent
├── README.md
└── references/
    ├── api.md                 # PALETTE, helper signatures, validation rules
    ├── design-theory.md       # Typography, layout, export policy, anti-redundancy
    ├── common-patterns.md     # Ultra-wide panels, legend axes, print-safe bars
    ├── tutorials.md           # End-to-end walkthroughs
    └── chart-types.md         # Radar, 3D, scatter, fill_between, log-scale

nature-polishing/
├── SKILL.md                   # 25 rules + 12-step workflow
└── README.md
```

---

## Candidate Skills (not yet built — contributions welcome)

| Candidate | Scope |
|-----------|-------|
| `nature-stats` | Effect sizes, CI, p-value formatting, sample size statements |
| `nature-response` | Peer-review response letters, point-by-point replies |
| `nature-methods` | Methods reproducibility checklist, forbidden phrases |
| `nature-cover` | Cover letter ≤ 500 words, hook paragraph, fit-to-journal argument |
| `nature-data` | Data availability statements, FAIR metadata standards |
```
