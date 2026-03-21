```markdown
---
name: dontbesilent-dbskill
description: Claude Code skill for dontbesilent's business diagnosis toolkit — routes to diagnosis, benchmark, content, unblock, and deconstruct skills built from 12,307 tweets
triggers:
  - diagnose my business model
  - help me find a benchmark competitor
  - diagnose my content strategy
  - I'm stuck and can't execute
  - deconstruct this concept for me
  - run business diagnosis
  - use dbs skill
  - commercial diagnosis
---

# dontbesilent-dbskill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Business diagnosis toolkit for Claude Code. Distilled from 12,307 tweets into 4,176 structured knowledge atoms across 6 diagnostic skills.

---

## Installation

```bash
npx skills add dontbesilent2025/dbskill
```

Or manually:

```bash
git clone https://github.com/dontbesilent2025/dbskill.git /tmp/dbskill \
  && cp -r /tmp/dbskill/skills/dbs* ~/.claude/skills/ \
  && rm -rf /tmp/dbskill
```

After installation, use `/dbs` inside Claude Code.

---

## Skills Overview

| Command | Purpose |
|---|---|
| `/dbs` | Main entry point — auto-routes to the right tool |
| `/dbs-diagnosis` | Business model diagnosis. Dissolves problems rather than answering them |
| `/dbs-benchmark` | Benchmark analysis. Five filters to eliminate noise |
| `/dbs-content` | Content creation diagnosis. Five-dimension detection |
| `/dbs-unblock` | Execution diagnosis. Adlerian framework |
| `/dbs-deconstruct` | Concept deconstruction. Wittgensteinian audit |

### Workflow

```
diagnosis (is the business model right?)
    ↓
benchmark (who to model after?)
    ↓
content (how to do content?)
    ↓
unblock (can't move forward?)

deconstruct (deconstruct concepts at any step)
```

Skills auto-recommend next steps. For example, if `diagnosis` detects psychological blockers, it suggests `unblock`.

---

## Usage

### Main Router

```
/dbs I'm running a consulting business but clients keep ghosting after the first call
```

The `/dbs` router reads the input and dispatches to the correct skill automatically.

---

### `/dbs-diagnosis` — Business Model Diagnosis

**What it does:** Applies 6 axioms + a dissolution funnel to identify whether you have a real business problem or a false framing problem.

```
/dbs-diagnosis
My SaaS has 200 signups but nobody converts to paid. We have a free tier.
```

**Typical output structure:**
1. Axiom scan — which of the 6 axioms are violated
2. Problem dissolution — is this a real problem or a framing error
3. Signal classification — structural vs. psychological vs. execution
4. Recommended next skill

**When to use:**
- Revenue is flat but you can't identify why
- You're about to pivot and want a second opinion
- The business "feels wrong" but you can't articulate it

---

### `/dbs-benchmark` — Benchmark Analysis

**What it does:** Finds who to model after using five filters to exclude misleading comparisons.

```
/dbs-benchmark
I run a Mandarin tutoring service for adult professionals in Europe. Who should I benchmark?
```

**Five filters applied:**
1. Same customer type (not same product)
2. Same distribution channel
3. Same pricing tier
4. Similar company age/stage
5. Accessible operational data

**When to use:**
- Deciding on pricing
- Choosing a growth channel
- Evaluating whether your metrics are good or bad

---

### `/dbs-content` — Content Diagnosis

**What it does:** Five-dimension detection for content that isn't working.

```
/dbs-content
My Twitter posts get impressions but zero engagement. I post about productivity and tools.
```

**Five dimensions checked:**
1. Audience-message fit
2. Format vs. platform norms
3. Value signal clarity
4. Call-to-action coherence
5. Consistency and cadence

**When to use:**
- Content gets views but no conversions
- You don't know what to post
- Engagement dropped suddenly

---

### `/dbs-unblock` — Execution Unblock

**What it does:** Diagnoses why you're not executing using the Adlerian framework (goal conflict, not willpower failure).

```
/dbs-unblock
I know exactly what I need to do but I keep procrastinating on cold outreach for 3 weeks.
```

**Framework applied:**
- Adlerian teleology: behavior serves a hidden goal
- Signal detection: what does avoiding this action protect?
- Reframe: surface the competing commitment
- Micro-action prescription

**When to use:**
- You know what to do but aren't doing it
- Procrastination on specific high-value tasks
- Execution works in some areas but breaks down in others

---

### `/dbs-deconstruct` — Concept Deconstruction

**What it does:** Wittgensteinian audit of vague business concepts to find where language is causing confusion.

```
/dbs-deconstruct
What does "product-market fit" actually mean for a services business?
```

**Process:**
1. List all uses of the term in circulation
2. Find family resemblances and contradictions
3. Identify which definition the user is actually operating with
4. Replace with concrete observables

**When to use:**
- A concept feels important but you can't act on it
- Team alignment is failing around a shared term
- You're using a term from one domain in another domain

---

## Knowledge Base (Standalone Use)

The knowledge base is fully open. You don't need the Skills installed to use it.

### Directory Structure

```
知识库/
├── 原子库/                     # Structured knowledge database
│   ├── atoms.jsonl             # 4,176 knowledge atoms (full)
│   ├── atoms_2024Q4.jsonl
│   ├── atoms_2025Q1.jsonl
│   └── README.md
│
├── Skill知识包/                 # Distilled methodology docs
│   ├── diagnosis_公理与诊断框架.md
│   ├── diagnosis_问题消解案例库.md
│   ├── benchmark_对标方法论.md
│   ├── benchmark_平台运营知识.md
│   ├── content_内容创作方法论.md
│   ├── content_平台特性与案例.md
│   ├── unblock_心理诊断框架.md
│   ├── unblock_信号案例库.md
│   ├── deconstruct_语言与概念框架.md
│   └── deconstruct_解构案例库.md
│
└── 高频概念词典.md
```

### Knowledge Atom Format

```json
{
  "id": "2024Q4_042",
  "knowledge": "判断一个生意能不能做，必要条件之一是你能不能说出这个产品的颜色",
  "original": "判断一个生意能不能做，必要条件之一是你能不能说出这个产品的颜色...",
  "url": "https://x.com/dontbesilent/status/...",
  "date": "2024-10-01",
  "topics": ["商业模式与定价", "语言与思维"],
  "skills": ["dbs-diagnosis", "dbs-deconstruct"],
  "type": "anti-pattern",
  "confidence": "high"
}
```

### Field Reference

| Field | Values | Notes |
|---|---|---|
| `type` | `principle` / `method` / `case` / `anti-pattern` / `insight` / `tool` | Filter for cases: ~700+ entries |
| `confidence` | `high` / `medium` / `low` | Use `high` for system prompts |
| `topics` | 10 categories (multi-select) | e.g. `"心理与执行力"` has 296 atoms |
| `skills` | One or more skill IDs | Cross-reference to skill docs |

---

## Using the Knowledge Base Without Installing Skills

### Scenario 1: Add business diagnosis to your own AI system prompt

Copy the content of `知识库/Skill知识包/diagnosis_公理与诊断框架.md` into your system prompt. Your AI gets the 6-axiom framework + dissolution funnel immediately.

### Scenario 2: RAG pipeline

```python
import json

atoms = []
with open("知识库/原子库/atoms.jsonl") as f:
    for line in f:
        atoms.append(json.loads(line))

# Filter high-confidence diagnosis atoms
diagnosis_atoms = [
    a for a in atoms
    if "dbs-diagnosis" in a["skills"] and a["confidence"] == "high"
]

# Filter only cases and anti-patterns
cases = [a for a in atoms if a["type"] in ("case", "anti-pattern")]
print(f"{len(cases)} case/anti-pattern atoms available")
```

### Scenario 3: Topic-based filtering

```python
# Get all execution/psychology atoms
unblock_atoms = [
    a for a in atoms
    if "心理与执行力" in a["topics"]
]
# ~296 atoms

# Get all pricing atoms
pricing_atoms = [
    a for a in atoms
    if "商业模式与定价" in a["topics"]
]
```

### Scenario 4: Build a standalone chatbot

```python
# System prompt = methodology doc
with open("知识库/Skill知识包/diagnosis_公理与诊断框架.md") as f:
    system_prompt = f.read()

# RAG = atoms.jsonl loaded into your vector DB
# No Claude Code required
```

---

## Common Patterns

### Pattern: Full workflow for a new venture

```
1. /dbs-diagnosis  → validate business model logic
2. /dbs-benchmark  → find comparable operators to study
3. /dbs-content    → design content strategy
4. /dbs-unblock    → address execution resistance
```

### Pattern: Stuck on a single concept

```
/dbs-deconstruct What does "niche down" mean for a B2B agency?
```

Use this before any strategy discussion where the key term feels fuzzy.

### Pattern: Content not converting

```
/dbs-content
Platform: LinkedIn
Post type: text posts about leadership
Current result: 200 impressions, 1-2 likes, 0 DMs
Goal: inbound leads for $5k/month coaching
```

The more specific the input, the more targeted the five-dimension output.

### Pattern: Cross-skill signal handoff

Skills auto-detect when a different skill would be more useful:

- `diagnosis` → detects avoidance language → recommends `unblock`
- `benchmark` → finds no valid comparables → recommends `diagnosis` (may be category creation)
- `content` → detects unclear positioning → recommends `diagnosis`
- `unblock` → detects concept confusion → recommends `deconstruct`

---

## Troubleshooting

**`/dbs` not found after installation**

Check that files were copied to the correct location:
```bash
ls ~/.claude/skills/ | grep dbs
```

Expected output:
```
dbs
dbs-benchmark
dbs-content
dbs-deconstruct
dbs-diagnosis
dbs-unblock
```

If missing, re-run the manual install:
```bash
git clone https://github.com/dontbesilent2025/dbskill.git /tmp/dbskill \
  && cp -r /tmp/dbskill/skills/dbs* ~/.claude/skills/ \
  && rm -rf /tmp/dbskill
```

**Knowledge base files not loading**

Skills read knowledge packages from relative paths. If you moved skill files, ensure the `知识库/Skill知识包/` directory is co-located or update path references in the SKILL.md files.

**Getting generic responses instead of framework-driven ones**

Provide more context. The skills route based on signal words. Instead of:
```
/dbs-diagnosis my business isn't working
```

Use:
```
/dbs-diagnosis
Business: B2B SaaS, $99/month, 50 paying customers, churn is 15%/month
Problem: can't figure out if this is a product problem or a sales problem
What I've tried: improved onboarding, added a success call
```

**Atoms.jsonl is large — only need a subset**

```bash
# Extract only high-confidence principle atoms
grep '"confidence": "high"' 知识库/原子库/atoms.jsonl | \
  grep '"type": "principle"' > principles_high.jsonl
```

---

## License

CC BY-NC 4.0. Free for personal use, learning, research, non-commercial projects. Commercial use requires separate authorization. Contact the author at [@dontbesilent](https://x.com/dontbesilent).
```
