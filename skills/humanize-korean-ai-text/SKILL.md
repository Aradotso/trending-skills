```markdown
---
name: humanize-korean-ai-text
description: Remove AI-generated Korean text patterns and rewrite in natural human style using the im-not-ai Claude Code skill harness
triggers:
  - "AI 티 없애줘"
  - "GPT 문체 제거해줘"
  - "사람이 쓴 것처럼 윤문해줘"
  - "remove AI writing patterns from Korean text"
  - "humanize this Korean AI-generated text"
  - "번역투 제거해줘"
  - "한글 AI 윤문해줘"
  - "make this Korean text sound more natural and human"
---

# Humanize KR — 한글 AI 티 제거기 (im-not-ai)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code skill harness that removes AI-generated text patterns from Korean writing. It detects and rewrites translation-style phrasing, mechanical structures, AI clichés, and unnatural rhythms — preserving all facts, figures, and meaning while restoring natural Korean voice.

## What It Does

- Detects 40+ AI-tell patterns across 10 categories (translation tone, AI clichés, hedging, bullet overuse, etc.)
- Rewrites only detected spans — untouched areas stay exactly as-is
- Enforces 4 core rules: meaning preservation, evidence-based edits only, genre consistency, no over-editing (30% change warning, 50% hard stop)
- Two modes: **Fast** (≤5,000 chars, ~3 min, single-agent) and **Strict** (8,000+ chars or `--strict`, 5-agent pipeline)
- Outputs `final.md` (rewritten) + `summary.md` (metrics, before/after, quality grade A–D)

## Installation

```bash
git clone https://github.com/epoko77-ai/im-not-ai.git
cd im-not-ai
claude
```

**Critical:** You must launch `claude` from inside the `im-not-ai` directory. The skill agents are loaded from `.claude/` — running from another directory gives you plain Claude Code with no skill loaded.

## Key Commands

### Natural Language (Easiest)

Paste AI-written Korean text and use any trigger phrase:

```
이 AI 글 자연스럽게 윤문해줘:

[paste ChatGPT / Claude / Gemini draft here]
```

Any of these phrases activate the skill automatically:
- `AI 티 없애줘`
- `GPT 문체 제거해줘`
- `사람이 쓴 것처럼 윤문해줘`
- `번역투 제거`
- `한글 AI 윤문`

### Slash Commands

```bash
# Basic humanize
/humanize [text or file path]

# With options (append as natural language)
/humanize [text] 장르: 칼럼
/humanize [text] 강도: 적극
/humanize [text] 최소심각도: S1

# Redo with specific focus
/humanize-redo "번역투만 다시"
/humanize-redo "관용구만 다시"
```

### Strict Mode

```bash
# Force strict mode (5-agent pipeline)
/humanize [text] --strict

# Strict mode triggers automatically for 8,000+ character inputs
```

## Output Files

All output lands in `_workspace/{date-number}/`:

### Fast Mode Output
| File | Content |
|------|---------|
| `01_input.txt` | Original text, unchanged |
| `final.md` | Rewritten humanized text |
| `summary.md` | Metrics, category detections (before/after), 6-point self-check, grade, change highlights |

### Strict Mode Output
| File | Content |
|------|---------|
| `01_input.txt` | Original text |
| `02_detection.json` | AI-tell detection report (position, type, severity) |
| `03_rewrite.md` | Rewritten text |
| `04_fidelity_audit.json` | 13-point meaning-preservation audit |
| `05_naturalness_review.json` | Naturalness re-measurement |
| `final.md` + `summary.md` | Final output + quality summary |

## AI-Tell Pattern Categories

| ID | Category | Example Patterns |
|----|----------|-----------------|
| A | 번역투 | `~를 통해`, `~에 있어서`, `~되어진다`, `가지고 있다` |
| B | 영어 과다 | Excessive parenthetical English, translatable terms left in English |
| C | 구조적 패턴 | Mechanical `첫째/둘째/셋째`, excessive bullets/headings/emoji |
| D | AI 관용구 | `결론적으로`, `시사하는 바가 크다`, `주목할 만하다`, `혁신적인` |
| E | 리듬 균일성 | Low sentence-length variance, repeated sentence endings |
| F | 수식·중복 | `매우`, `정말`, double-synonym modifiers, `~적/~성/~화` overuse |
| G | Hedging | `~할 수 있을 것으로 보인다` (stacked hedges) |
| H | 접속사 남발 | Consecutive sentence-initial `또한/따라서/즉/나아가` |
| I | 형식명사 과다 | `것이다`, `점`, `수`, `바`, `~할 필요가 있다` |
| J | 시각 장식 | Excessive **bold**, "quotes", em-dash (—) overuse |

## Severity Levels

- **S1 Critical** — Even one occurrence signals AI authorship. Always remove.
- **S2 Strong** — 1–2 occurrences acceptable; 3+ repetitions → remove.
- **S3 Weak** — Only problematic when overlapping with other patterns.

## Quality Grades (Post-Rewrite)

- **A** — S1: 0, S2: ≤2, improvement ≥70%
- **B** — S1: 0, S2: ≤4, improvement ≥50%
- **C** — S1: 1–2 or 2+ over-editing signals → triggers round 2
- **D** — S1: 3+ or severe over-editing → escalate for human review

## 4 Core Rules

The skill enforces these constraints on every run:

1. **Meaning unchanged** — Facts, claims, numbers, proper nouns, direct quotes: 100% original.
2. **Evidence-based only** — Only detected spans get edited. Undetected sections stay untouched.
3. **Genre preserved** — Columns stay columns, reports stay reports.
4. **No over-editing** — >30% change rate triggers warning; >50% forces hard stop.

## Do-NOT Edit List

These elements are never modified regardless of detected patterns:
- Numbers, units, dates
- Proper nouns, person names, product names, model names
- Direct quotes inside double quotation marks (`"..."`)
- Legal/regulatory text
- Academic terminology (when unavoidable)

## Refinement Commands

After getting results, request adjustments in plain Korean:

```
# Re-run a specific section
"이 문단만 다시 윤문해줘"

# Target a specific category
"번역투만 더 손봐줘"
"관용구만 다시"

# Adjust intensity
"윤문 강도 낮춰줘"          # conservative — critical patterns only
"원문 톤을 더 살려줘"        # lower change rate ceiling
"2차 윤문해줘"              # run another pass on current output
```

## Running Multiple Texts

Each run creates a new `_workspace/{date-number}/` folder, so results never mix. Within the same Claude Code session, paste a new text and request humanization again.

```
# First text
AI 티 없애줘:
[text 1]

# After results saved to _workspace/20260426-001/...

# Second text — same session, new folder created automatically
사람이 쓴 것처럼 윤문해줘:
[text 2]
# → saved to _workspace/20260426-002/
```

## Agent Architecture Reference

### Fast Mode (Default, ≤5,000 chars)
```
Input text
    ↓
[humanize-monolith]   # detect → rewrite → self-verify in one call
    ↓                 # 4–5 tool calls cap, ~3 min
final.md + summary.md
```

### Strict Mode (`--strict` or 8,000+ chars auto-upgrade)
```
Input text
    ↓
[ai-tell-detector]           # span-level detection JSON
    ↓
[korean-style-rewriter]      # surgical rewrite from findings
    ↓
[Parallel validation]
    ├─ [content-fidelity-auditor]   # 13-item meaning audit
    └─ [naturalness-reviewer]       # re-detect residual AI-tells
    ↓
[Orchestrator]
    ├─ accept              → final.md + summary.md
    ├─ rewrite_round_2     → second pass (max 3 rounds)
    ├─ rollback_and_rewrite → revert problematic edits
    └─ hold_and_report     → flag for human review
```

## Common Patterns & Examples

### Pattern A: 번역투 (Translation Tone) — Most Common
```
# Before (AI-written)
AI 기술을 통해 효율을 높일 수 있다.
이에 있어서 중요한 점은 데이터다.
보고서에 의해 확인된 사실이다.

# After (humanized)
AI로 효율을 높인다.
여기서 핵심은 데이터다.
보고서가 확인한 사실이다.
```

### Pattern D: AI 관용구 (AI Clichés)
```
# Before
결론적으로, 이는 시사하는 바가 크다.
주목할 만한 혁신적인 접근법이다.

# After
(removed — adds no meaning)
새로운 접근법이다.
```

### Pattern G: Hedging 남용
```
# Before
이는 효과가 있을 수 있을 것으로 보인다.

# After
효과가 있다. / 효과적일 수 있다.
```

### Pattern C: 기계적 구조
```
# Before
첫째, 비용이 절감된다. 둘째, 속도가 빨라진다. 셋째, 품질이 높아진다.

# After
비용이 줄고 속도도 빨라지며 품질도 올라간다.
```

## Web Service Extension (Optional)

The `humanize-web-architect` agent designs a Next.js 15 + Vercel web app. Activate with:

```
humanize-web-architect 웹 서비스 설계해줘
```

UX flow: Input → Detection highlight → Left/right diff → Copy humanized output.
Roadmap: v0 MVP (anonymous) → v1 (login + history) → v2 (Pro/Team + API) → v3 (Chrome Extension) → v4 (Japanese/Chinese).

## Plugin Install (Third-Party Fork)

The [`gaebalai/im-not-ai`](https://github.com/gaebalai/im-not-ai) fork packages this as a Claude Code Plugin:

```bash
# One-line install to a target project
./scripts/install.sh --target ~/my-project

# Or via plugin registry
/plugin install humanize-korean@epoko77-ai-plugins
```

Official plugin support planned for v1.6.

## Taxonomy Management

To add or review new AI-tell patterns:

```
korean-ai-tell-taxonomist 새 패턴 심사해줘: [pattern description]
```

This agent manages the SSOT taxonomy in `.claude/skills/humanize-korean/references/ai-tell-taxonomy.md` and promotes validated patterns through review.

## Troubleshooting

**Skill not activating**
- Confirm you launched `claude` from inside the `im-not-ai` directory
- Check `.claude/` directory exists and contains skill definitions

**Results taking too long (>10 min)**
- Fast mode should complete in ~3 min for ≤5,000 chars
- If using strict mode on short text, switch to fast: remove `--strict` flag
- v1.5 fixed the 25-min wall-clock issue from v1.2–v1.4 by reverting to monolith fast path

**Over-editing warning triggered**
- Request lower intensity: `"윤문 강도 낮춰줘"` or `"원문 톤을 더 살려줘"`
- Hard stop at 50% change rate is by design — request human review or reduce scope

**Grade C or D result**
- Grade C auto-triggers round 2 rewrite
- Grade D escalates to human review — use `/humanize-redo` with specific focus area

**Meaning changed in output**
- Report the specific span: `"이 부분 의미가 바뀌었어: [original span]"`
- The content-fidelity-auditor's 13-point checklist should catch this; if it didn't, strict mode with `--strict` gives more thorough auditing

## Key Reference Files

```
.claude/skills/humanize-korean/references/
├── ai-tell-taxonomy.md      # Full 40+ pattern taxonomy with prescriptions
├── rewriting-playbook.md    # Rewriting rules and examples per category
├── quick-rules.md           # Slim rulebook for monolith fast path (~150 lines)
└── web-service-spec.md      # Web service architecture spec

.claude/commands/
├── humanize.md              # /humanize command definition
└── humanize-redo.md         # /humanize-redo command definition
```
```
