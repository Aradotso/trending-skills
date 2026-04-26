```markdown
---
name: mckinsey-pptx-generator
description: Generate McKinsey-style PowerPoint presentations via Claude Code plugin with 40 slide templates and intelligent template selection
triggers:
  - "create a McKinsey style presentation"
  - "generate a pptx slide deck"
  - "make a consulting style PowerPoint"
  - "build a business review deck"
  - "create slides from Excel data"
  - "generate a strategy presentation"
  - "make a McKinsey deck from my data"
  - "create a professional slide deck"
---

# McKinsey PPTX Generator

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code plugin that generates McKinsey-style `.pptx` files from natural language instructions. It ships 40 professional slide templates and a subagent (`mckinsey-slide-agent`) that picks the right template for each slide and explains its reasoning.

---

## What It Does

- Accepts plain-language requests ("make a Q4 business review deck")
- Reads source files: `.xlsx`, `.csv`, `.docx`, `.pdf`, `.md`, `.txt`
- Selects from 40 consulting-grade slide templates (cover, chart, matrix, roadmap, dashboard, etc.)
- Explains template selection rationale per slide
- Writes a finished `.pptx` to `output/` in the working directory

---

## Installation (Claude Code Plugin)

### Step 1 — Add from marketplace

```
/plugin marketplace add seulee26/mckinsey-pptx
```

### Step 2 — Install

```
/plugin install axlabs-mckinsey-pptx@axlabs
```

### Step 3 — **Restart Claude Code** (critical)

The plugin only loads after a full restart:

```
/exit
```

Then relaunch Claude Code. Verify:

```
/agents
```

`mckinsey-slide-agent` must appear in the list.

### Step 4 — Install Python dependencies

Tell Claude:

```
이 플러그인이 쓸 파이썬 라이브러리들 설치해줘.
```

Or manually:

```bash
pip install python-pptx openpyxl python-docx pdfplumber pandas
```

### Optional — PDF preview (macOS)

```bash
brew install --cask libreoffice && brew install poppler
```

---

## Project Structure

```
axlabs-mckinsey-pptx/
├── templates/          # 40 slide template definitions
├── agents/
│   └── mckinsey_slide_agent.py   # subagent: picks templates, defends choices
├── generators/
│   ├── slide_builder.py          # python-pptx slide construction
│   ├── chart_builder.py          # chart rendering helpers
│   └── data_reader.py            # xlsx/csv/pdf/docx ingestion
├── themes/
│   └── mckinsey.py               # colour palette, fonts, spacing
└── output/                       # generated .pptx files land here
```

---

## Recommended Folder Layout for Your Project

```
my-deck/
├── inputs/
│   ├── revenue.xlsx
│   ├── roadmap.md
│   └── research.pdf
└── output/          ← auto-created; .pptx files appear here
```

Launch Claude Code **from inside** `my-deck/` so the agent can read `inputs/`:

```bash
cd ~/Desktop/my-deck
claude
```

---

## Core Usage — Natural Language Requests

### Minimal request

```
Q4 사업 리뷰 데크 만들어줘. 매출 1,200억, 전년 대비 14% 성장, KPI 지연 2건.
```

### With source files

```
inputs/revenue.xlsx의 '월별매출' 시트 숫자로 Q4 리뷰 덱 7슬라이드 만들어줘.
경영진 대상, 결론은 투자 승인 요청.
```

### English deck

```
Create a 10-slide market entry kickoff deck in English.
Use inputs/market_research.pdf for context and inputs/roadmap.md for the timeline.
Show the 12-week plan as a Gantt chart.
```

### Let the agent find the story

```
inputs/5yr_revenue.csv만 보고, 이 숫자들이 말하는 이야기를 찾아서
5슬라이드로 보여줘. 어떤 메시지가 중요한지 네가 판단해줘.
```

---

## Template Catalogue (40 templates)

Key categories the subagent chooses from:

| Category | Templates include |
|---|---|
| Cover / Title | `title-dark`, `title-light`, `title-split` |
| Executive Summary | `exec-summary-3up`, `exec-summary-bullets` |
| Charts | `bar-chart`, `line-trend`, `waterfall`, `stacked-bar` |
| KPI / Dashboard | `kpi-4up`, `kpi-dashboard`, `scorecard` |
| Matrix / 2×2 | `bcg-matrix`, `priority-matrix`, `risk-matrix` |
| Comparison | `side-by-side`, `3-column-compare`, `before-after` |
| Process / Roadmap | `timeline`, `gantt`, `swim-lane`, `process-flow` |
| Text / Insight | `single-insight`, `bullets-3`, `quote-callout` |
| Appendix | `data-table`, `footnote-slide` |

---

## Python API — Direct Usage

### Build a slide deck programmatically

```python
from generators.slide_builder import DeckBuilder
from themes.mckinsey import McKinseyTheme

theme = McKinseyTheme()
deck = DeckBuilder(theme=theme, output_dir="output")

# Cover slide
deck.add_slide(
    template="title-dark",
    data={
        "title": "Q4 2025 Business Review",
        "subtitle": "Board of Directors",
        "date": "December 2025",
    }
)

# KPI dashboard from dict
deck.add_slide(
    template="kpi-4up",
    data={
        "kpis": [
            {"label": "Revenue",       "value": "₩120B", "delta": "+14%", "trend": "up"},
            {"label": "Gross Margin",  "value": "38%",   "delta": "+2pp", "trend": "up"},
            {"label": "NPS",           "value": "42",    "delta": "+7",   "trend": "up"},
            {"label": "Delayed KPIs",  "value": "2",     "delta": "",     "trend": "flat"},
        ]
    }
)

# Bar chart from pandas DataFrame
import pandas as pd

df = pd.DataFrame({
    "Month": ["Jan","Feb","Mar","Apr","May","Jun"],
    "Revenue": [8.2, 9.1, 10.4, 11.0, 10.8, 12.3],
})

deck.add_slide(
    template="bar-chart",
    data={
        "title": "Monthly Revenue Trend (₩B)",
        "dataframe": df,
        "x_col": "Month",
        "y_col": "Revenue",
        "highlight_last": True,
    }
)

# 2×2 priority matrix
deck.add_slide(
    template="priority-matrix",
    data={
        "title": "Initiative Prioritisation",
        "x_label": "Effort",
        "y_label": "Impact",
        "items": [
            {"label": "CRM Upgrade",     "x": 0.3, "y": 0.9},
            {"label": "ERP Migration",   "x": 0.9, "y": 0.6},
            {"label": "Brand Refresh",   "x": 0.4, "y": 0.4},
            {"label": "Market Research", "x": 0.2, "y": 0.5},
        ]
    }
)

# Gantt / roadmap
deck.add_slide(
    template="gantt",
    data={
        "title": "12-Week Launch Roadmap",
        "phases": [
            {"name": "Discovery",  "start": 1, "end": 3,  "owner": "Strategy"},
            {"name": "Build",      "start": 3, "end": 8,  "owner": "Engineering"},
            {"name": "Pilot",      "start": 7, "end": 10, "owner": "Operations"},
            {"name": "Scale",      "start": 10,"end": 12, "owner": "All"},
        ]
    }
)

path = deck.save("q4-review")
print(f"Saved → {path}")   # output/q4-review.pptx
```

---

### Read source data before building

```python
from generators.data_reader import DataReader

reader = DataReader()

# Excel — specify sheet and range
df = reader.read_excel(
    path="inputs/revenue.xlsx",
    sheet="월별매출",
    usecols="A:F",
    nrows=12,
)

# CSV
df = reader.read_csv("inputs/5yr_revenue.csv")

# PDF — returns extracted text per page
pages = reader.read_pdf("inputs/research.pdf", page_range=(3, 7))

# Word document
text = reader.read_docx("inputs/brief.docx")
```

---

### Invoke the subagent directly (Python)

```python
from agents.mckinsey_slide_agent import McKinseySlideAgent

agent = McKinseySlideAgent()

# Ask agent to select a template and explain why
result = agent.select_template(
    slide_intent="Show five-year revenue growth with a CAGR callout",
    available_data={"type": "time_series", "rows": 5, "metric": "revenue"},
)

print(result["template"])    # e.g. "line-trend"
print(result["rationale"])   # agent's explanation
print(result["alternatives"]) # runner-up templates

# Build the slide with agent's choice
from generators.slide_builder import DeckBuilder
deck = DeckBuilder()
deck.add_slide(template=result["template"], data={"dataframe": df, "title": "Revenue 2020–2025"})
deck.save("revenue-story")
```

---

## Iterative Refinement via Chat

After the first draft, keep the conversation going:

```
슬라이드 4를 다른 레이아웃으로 바꿔줘. 숫자 비교가 더 잘 보이게.
```

```
슬라이드 2의 세 번째 불릿을 "NPS 42점(업계 평균 +15)"로 수정.
```

```
전체 톤을 더 단정하게. 이모지랑 느낌표 다 빼줘.
```

```
이 덱 영문판도 같이 만들어줘. 구조랑 숫자는 똑같이.
```

```
표지에 inputs/logo.png 넣어줘.
```

---

## Theme Customisation

```python
from themes.mckinsey import McKinseyTheme

# Default: McKinsey deep-navy palette
theme = McKinseyTheme()

# Override brand colours
theme.primary   = "#003087"   # your brand navy
theme.accent    = "#E8000D"   # your brand red
theme.font_body = "Pretendard"

# Override slide dimensions (default: widescreen 16:9)
theme.slide_width_emu  = 9144000   # 10 inches
theme.slide_height_emu = 5143500   # 5.625 inches

from generators.slide_builder import DeckBuilder
deck = DeckBuilder(theme=theme)
```

---

## Common Patterns

### Pattern A — Monthly business review

```python
deck.add_slide("title-dark",         data={"title": "Q4 Business Review", "date": "Dec 2025"})
deck.add_slide("exec-summary-3up",   data={"points": ["Revenue beat +14%", "2 KPI delays", "3 new markets"]})
deck.add_slide("bar-chart",          data={"title": "Monthly Revenue", "dataframe": monthly_df})
deck.add_slide("kpi-4up",            data={"kpis": kpi_list})
deck.add_slide("risk-matrix",        data={"title": "Risk Register", "items": risks})
deck.add_slide("timeline",           data={"title": "H1 2026 Roadmap", "phases": roadmap})
deck.add_slide("single-insight",     data={"headline": "Recommend: approve ₩5B expansion budget"})
deck.save("q4-review")
```

### Pattern B — Strategy one-pager

```python
deck.add_slide("title-split",        data={"title": "Indonesia Market Entry", "subtitle": "Strategic Assessment"})
deck.add_slide("bcg-matrix",         data={"title": "Portfolio View", "items": portfolio})
deck.add_slide("side-by-side",       data={"left": build_case, "right": partner_case})
deck.add_slide("gantt",              data={"title": "18-Month Roadmap", "phases": phases})
deck.save("indonesia-strategy")
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `mckinsey-slide-agent` not in `/agents` | Restart Claude Code completely (`/exit` then relaunch) |
| `ModuleNotFoundError: python_pptx` | `pip install python-pptx openpyxl python-docx pdfplumber pandas` |
| PDF table extraction garbled | Supply the same data as `.xlsx`; agent prefers Excel over PDF tables |
| `.doc` file not readable | Open in Word and resave as `.docx` |
| Slide layout looks wrong | Ask: "슬라이드 N 다른 레이아웃으로 바꿔줘" — agent picks from alternatives |
| Logo not appearing | Confirm path is relative to working directory: `inputs/logo.png` |
| Output folder missing | Created automatically on first `deck.save()`; check working directory |
| LibreOffice preview fails | Preview is optional — `.pptx` is still valid; open with PowerPoint/Keynote |

---

## Key Facts for AI Agents

- **Entry point for chat:** just describe the deck in Korean or English — no command needed
- **Entry point for code:** `from generators.slide_builder import DeckBuilder`
- **Template selector:** `from agents.mckinsey_slide_agent import McKinseySlideAgent`
- **Output:** always `output/<name>.pptx` relative to the Claude Code working directory
- **Data privacy:** files are processed locally; only the conversation text reaches Anthropic servers
- **No design knowledge required:** the subagent handles all template selection and layout decisions
```
