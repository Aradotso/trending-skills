```markdown
---
name: humanize-korean-ai-text
description: Remove AI-generated writing patterns from Korean text using the im-not-ai Claude Code skill harness
triggers:
  - "AI 티 없애줘"
  - "GPT 문체 제거해줘"
  - "사람이 쓴 것처럼 윤문해줘"
  - "번역투 제거해줘"
  - "한글 AI 윤문"
  - "remove AI tells from Korean text"
  - "humanize this Korean writing"
  - "AI가 쓴 글 자연스럽게 만들어줘"
---

# Humanize Korean AI Text (im-not-ai)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code skill harness that removes AI-generated writing patterns from Korean text while preserving meaning 100%. Targets 10 categories × 40+ sub-patterns (translation artifacts, mechanical structure, AI clichés, hedging overuse, etc.) with surgical span-level edits.

## What It Does

- Detects and removes **번역투** (translation artifacts): "~를 통해", "~에 있어서", "~에 의해"
- Eliminates **AI 관용구**: "결론적으로", "시사하는 바가 크다", "주목할 만하다"
- Fixes **기계적 병렬**: "첫째/둘째/셋째" lists, excessive bullets/headings/emoji
- Reduces **피동태 남용**, **접속사 남발** (또한/따라서/즉/나아가), **형식명사 과다**
- Two modes: **Fast** (monolith, ≤5,000 chars, ~3 min) and **Strict** (5-agent pipeline, 8,000+ chars)
- 4 iron rules: meaning unchanged, evidence-based edits only, genre preserved, ≤30% change rate

## Installation

```bash
git clone https://github.com/epoko77-ai/im-not-ai.git
cd im-not-ai
claude
```

**Critical:** Always launch `claude` from inside the `im-not-ai` directory. Skills only load from the local `.claude/` folder.

```bash
# Verify Claude Code is installed
claude --version

# Always run from project root
cd im-not-ai && claude
```

## Key Commands

### Method A — Natural Language (Easiest)

Paste AI-generated Korean text and ask naturally:

```
이 AI 글 자연스럽게 윤문해줘:

[ChatGPT / Claude / Gemini 초안 붙여넣기]
```

Any of these phrases trigger the skill automatically:
- `AI 티 없애줘`
- `GPT 문체 제거해줘`
- `사람이 쓴 것처럼 윤문해줘`
- `번역투 제거`
- `한글 AI 윤문`

### Method B — Slash Command

```bash
# Basic usage
/humanize [텍스트 또는 파일 경로]

# With options
/humanize draft.txt 장르: 칼럼
/humanize draft.txt 강도: 적극
/humanize draft.txt 최소심각도: S1

# Redo with specific focus
/humanize-redo "번역투만 다시"
/humanize-redo "관용구 카테고리만"
```

### Method C — Strict Mode (Explicit)

```bash
# Force strict 5-agent pipeline
/humanize --strict [텍스트]

# Auto-upgrades to strict for 8,000+ character inputs
```

## Output Files

All runs write to `_workspace/{date-number}/`:

### Fast Mode Output
```
_workspace/20260426-001/
├── 01_input.txt        # Original text (unchanged)
├── final.md            # Humanized result
└── summary.md          # Metrics, detected patterns, grade, change highlights
```

### Strict Mode Output
```
_workspace/20260426-001/
├── 01_input.txt              # Original text
├── 02_detection.json         # AI-tell detection report (spans, categories, severity)
├── 03_rewrite.md             # Humanized draft
├── 04_fidelity_audit.json    # Content fidelity audit (13-point checklist)
├── 05_naturalness_review.json # Naturalness re-measurement
├── final.md                  # Final humanized text
└── summary.md                # Score, grade, key changes
```

## Detection Categories

| ID | Category | Key Patterns |
|----|----------|-------------|
| A | 번역투 | `~를 통해`, `~에 대해`, `~에 있어서`, `~되어진다`, `가지고 있다` |
| B | 영어 과다 | Unnecessary English terms, excessive parenthetical bilingual |
| C | 구조적 패턴 | `첫째/둘째/셋째`, excessive bullets, headings, emoji |
| D | AI 관용구 | `결론적으로`, `시사하는 바가 크다`, `혁신적인`, `주목할 만하다` |
| E | 리듬 균일 | Low sentence-length variance, repeated endings |
| F | 수식·중복 | `매우`, `정말`, double synonyms, `-적/-성/-화` overuse |
| G | Hedging | `~할 수 있을 것으로 보인다` stacked hedges |
| H | 접속사 남발 | Sentence-initial `또한/따라서/즉/나아가` chains |
| I | 형식명사 | `것이다`, `점`, `수`, `바`, `~할 필요가 있다` |
| J | 시각 장식 | Excessive **bold**, "quotes", em-dash overuse |

## Severity Levels

- **S1 결정적**: Remove unconditionally — one occurrence confirms AI authorship
- **S2 강함**: Allow 1–2 occurrences; flag 3+
- **S3 약함**: Only problematic when combined with other patterns

## Quality Grades (Post-Humanization)

| Grade | Criteria |
|-------|----------|
| **A** | S1=0, S2≤2, ≥70% pattern improvement |
| **B** | S1=0, S2≤4, ≥50% improvement |
| **C** | S1=1–2 or 2+ over-edit signals → triggers round 2 |
| **D** | S1≥3 or severe over-edit → human review recommended |

## Refinement Commands

After initial humanization, iterate with natural language:

```
# Re-process a specific paragraph
이 문단만 다시 윤문해줘

# Target a specific category
번역투만 더 손봐줘
관용구만 다시 처리해줘

# Adjust intensity
윤문 강도 낮춰줘          # Conservative — only S1 patterns
원문 톤을 더 살려줘        # Lower change rate cap
2차 윤문해줘              # Re-polish current result
```

## 7-Agent Architecture

| Agent | Mode | Role |
|-------|------|------|
| `humanize-monolith` | Fast (default) | Single-call detect + rewrite + self-verify (4–5 tool calls cap) |
| `ai-tell-detector` | Strict | JSON detection report with span positions |
| `korean-style-rewriter` | Strict | Evidence-based surgical rewriting |
| `content-fidelity-auditor` | Strict | 13-point meaning-equivalence audit |
| `naturalness-reviewer` | Strict | Residual AI-tell + over-edit judgment, grades A–D |
| `korean-ai-tell-taxonomist` | Standalone | SSOT taxonomy management, pattern promotion |
| `humanize-web-architect` | Optional | Next.js 15 + Vercel web service design |

## What Is Never Changed (Do-NOT List)

```
- Numbers, units, dates
- Proper nouns, person names, product names, model names  
- Direct quotations (inside 큰따옴표)
- Legal or regulatory text
- Academic technical terms (when unavoidable)
```

## Real Usage Example

**Input (AI-generated):**
```
인공지능 기술을 통해 다양한 분야에서 혁신적인 변화가 일어나고 있다. 
첫째, 의료 분야에서의 활용이 주목할 만하다. 둘째, 교육 분야에 있어서도 
중요한 시사하는 바가 크다. 결론적으로, 이러한 기술적 발전은 사회 전반에 
걸쳐 패러다임 시프트를 가져올 것으로 보인다.
```

**After humanization:**
```
인공지능이 여러 분야를 바꾸고 있다. 의료에서는 진단 보조가 실용화됐고, 
교육에서는 개인 맞춤 학습이 가능해졌다. 기술 속도를 보면 사회 전반의 
변화는 이미 시작됐다.
```

**summary.md excerpt:**
```
## 탐지 결과
- A (번역투): "~를 통해" ×1 [S1], "~에 있어서" ×1 [S1] → 제거
- C (구조적): "첫째/둘째" ×2 [S2] → 산문 통합
- D (AI 관용구): "결론적으로" ×1 [S1], "시사하는 바가 크다" ×1 [S1],
                 "주목할 만하다" ×1 [S1], "혁신적인" ×1 [S2] → 제거
- G (Hedging): "~것으로 보인다" ×1 [S2] → 단정형

변경률: 43% | 등급: B
```

## Reference Files

```
.claude/skills/humanize-korean/references/
├── ai-tell-taxonomy.md      # Full 40+ sub-pattern SSOT with prescriptions
├── rewriting-playbook.md    # Rewrite recipes per category
├── quick-rules.md           # Slim rulebook for monolith fast path (~150 lines)
└── web-service-spec.md      # Next.js 15 web service expansion spec

.claude/commands/
├── humanize.md              # /humanize slash command definition
└── humanize-redo.md         # /humanize-redo slash command definition
```

## Over-Edit Safeguards

```
Change rate < 30%  → proceed normally
Change rate 30–50% → warning logged in summary.md
Change rate > 50%  → forced stop, human review required
```

## Running Multiple Documents

Each run creates a new timestamped workspace — no cross-contamination:

```bash
# Run 1 → _workspace/20260426-001/
# Run 2 → _workspace/20260426-002/
# Run 3 → _workspace/20260426-003/
```

Within the same Claude Code session, just paste a new text and request again.

## Web Service Extension (Optional)

The `humanize-web-architect` agent designs a full web app:

- **Stack**: Next.js 15 App Router + Vercel Fluid Compute + AI Gateway
- **UX Flow**: Input → Detection highlights → Left/right diff → Copy humanized output
- **Roadmap**: v0 MVP (anonymous) → v1 (auth + history) → v2 (Pro/Team + API) → v3 (Chrome Extension) → v4 (Japanese/Chinese)

Spec: `.claude/skills/humanize-korean/references/web-service-spec.md`

## Troubleshooting

**Skill not activating:**
```bash
# Make sure you're in the project directory
pwd  # must show .../im-not-ai
ls .claude/  # must exist
claude  # restart from correct directory
```

**Fast mode taking too long (>5 min):**
- Input may have exceeded 5,000 chars → auto-upgraded to strict mode (expected)
- For very long texts, use `--strict` explicitly so you set expectations correctly

**Grade C/D result:**
```
# Ask for a second round explicitly
2차 윤문해줘

# Or target the remaining issues
S1 패턴만 다시 처리해줘
```

**Over-edit warning triggered:**
```
원문 톤을 더 살려줘
# or
윤문 강도 낮춰줘
```

**Content changed (meaning altered):**
- The `content-fidelity-auditor` (strict mode) will catch this and trigger rollback automatically
- In fast mode, check `summary.md` self-verification section — 6-point checklist included

## Version Notes (v1.5)

v1.2–v1.4 are deprecated. v1.5 rolls back to v1.1's simple structure + adds the monolith fast path. Root cause of previous slowness: agent-to-agent context reload overhead, not model choice. Fast mode wall-clock target: **2–3 minutes** for ≤5,000 chars (was 25 min in v1.4).
```
