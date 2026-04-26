```markdown
---
name: humanize-korean-ai-text
description: Remove AI-written tells from Korean text using the im-not-ai Claude Code skill — detects 40+ patterns across 10 categories and rewrites naturally while preserving meaning.
triggers:
  - "AI 티 없애줘"
  - "GPT 문체 제거해줘"
  - "사람이 쓴 것처럼 윤문해줘"
  - "번역투 제거해줘"
  - "한글 AI 윤문해줘"
  - "remove AI tells from Korean text"
  - "humanize this Korean writing"
  - "AI가 쓴 한국어 글 자연스럽게 고쳐줘"
---

# Humanize Korean AI Text (im-not-ai)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

`im-not-ai` is a Claude Code skill that removes AI-written tells from Korean text. It detects 40+ patterns across 10 categories (번역투, 기계적 구조, AI 관용구, etc.) and rewrites text to sound naturally human — without changing facts, claims, numbers, or proper nouns.

## Installation

```bash
git clone https://github.com/epoko77-ai/im-not-ai.git
cd im-not-ai
claude
```

> **Critical:** Always launch `claude` from inside the `im-not-ai` directory. The skills are loaded from `.claude/` relative to your working directory. Launching from elsewhere gives you plain Claude Code with no humanize skills.

## Two Operating Modes

### Fast Mode (default, ≤5,000 chars, ~2–3 min)
Single `humanize-monolith` agent call. Detect → rewrite → self-verify in one pass. Capped at 4–5 tool calls.

### Strict Mode (`--strict` flag or auto-escalated for 8,000+ chars)
5-agent pipeline: detector → rewriter → parallel (fidelity auditor + naturalness reviewer) → orchestrator. More thorough, produces intermediate JSON audit files.

## Usage

### Method A — Natural Language (easiest)

Just paste your AI-written Korean text and say:

```
이 AI 글 자연스럽게 윤문해줘:

[ChatGPT / Claude / Gemini 초안 붙여넣기]
```

Any of these trigger phrases activate the skill automatically:
- "AI 티 없애줘"
- "GPT 문체 제거해줘"
- "사람이 쓴 것처럼 윤문해줘"
- "번역투 제거"
- "한글 AI 윤문"

### Method B — Slash Command

```
/humanize [텍스트 또는 파일 경로]
```

With options appended in natural language:
```
/humanize article.md 장르: 칼럼 강도: 적극 최소심각도: S1
```

Redo with specific focus:
```
/humanize-redo "번역투만 다시"
/humanize-redo "관용구 카테고리만 재처리"
```

### Method C — File Path Input

```
이 파일 윤문해줘: ./drafts/blog-post.md
```

## Output Files

All runs write to `_workspace/{date-number}/`:

### Fast Mode Output
```
_workspace/20260426-001/
├── 01_input.txt        # 원문 그대로
├── final.md            # 윤문본
└── summary.md          # 메트릭, 탐지 결과, 자체검증 6항, 등급, 변경 하이라이트
```

### Strict Mode Output
```
_workspace/20260426-002/
├── 01_input.txt
├── 02_detection.json       # span 단위 AI 티 탐지 리포트
├── 03_rewrite.md           # 윤문본
├── 04_fidelity_audit.json  # 의미 동등성 감사 (13항 체크리스트)
├── 05_naturalness_review.json  # 잔존 AI 티 + 과윤문 판정
├── final.md
└── summary.md
```

## Detection Categories

| ID | Category | Example Patterns |
|----|----------|-----------------|
| A | 번역투 | "~를 통해", "~에 있어서", "~되어진다", "가지고 있다" |
| B | 영어 과다 | 불필요한 괄호 영어 병기, 번역 가능한 영어 그대로 사용 |
| C | AI 구조 패턴 | "첫째/둘째/셋째", 과도한 불릿·헤딩·이모지 |
| D | AI 관용구 | "결론적으로", "시사하는 바가 크다", "혁신적인" |
| E | 리듬 균일성 | 문장 길이 표준편차 낮음, 동일 종결어미 반복 |
| F | 수식·중복 | "매우", "정말", "~적/~성/~화" 남발 |
| G | Hedging | "~할 수 있을 것으로 보인다" 다중 완곡 |
| H | 접속사 남발 | 문두 "또한/따라서/즉/나아가" 연속 |
| I | 형식명사 과다 | "것이다", "~할 필요가 있다" |
| J | 시각 장식 | 과도한 **볼드**, 따옴표, 대시(—) 남발 |

Full taxonomy with 40+ sub-patterns: `.claude/skills/humanize-korean/references/ai-tell-taxonomy.md`

## Severity Levels

- **S1 결정적** — Remove unconditionally. One occurrence alone signals AI writing.
- **S2 강함** — 1–2 occurrences tolerated; remove when 3+ appear.
- **S3 약함** — Only problematic when stacked with other patterns.

## Quality Grades (Post-Rewrite)

| Grade | Criteria |
|-------|----------|
| **A** | S1: 0건, S2: ≤2건, improvement ≥70% |
| **B** | S1: 0건, S2: ≤4건, improvement ≥50% |
| **C** | S1: 1–2건 or 2+ over-rewrite signals → triggers round 2 |
| **D** | S1: 3건+ or severe over-rewrite → human review recommended |

## Rewrite Controls & Guardrails

The skill enforces these limits automatically:
- **30% change rate** — warning issued
- **50% change rate** — forced stop

These are **never modified**. If output feels too conservative, ask for a higher-intensity pass instead.

## Refinement Commands

After receiving output, refine with natural language — no special syntax needed:

```
이 문단만 다시 윤문해줘
번역투만 더 손봐줘
윤문 강도 낮춰줘
원문 톤을 더 살려줘
2차 윤문해줘
관용구 카테고리만 재처리해줘
```

Partial reruns and "2차 윤문" automatically escalate to strict mode.

## What Is NEVER Changed

The skill is hardcoded to preserve:
- Numbers, units, dates
- Proper nouns, names, product names, model names
- Direct quotations (content inside 큰따옴표)
- Legal/regulatory text
- Academic technical terms (when unavoidable)

## Common Before/After Patterns

```
# 번역투 (Category A)
Before: "AI 기술을 통해 효율을 높일 수 있다"
After:  "AI로 효율을 높인다"

Before: "이에 있어서 중요한 점은"
After:  "여기서 중요한 건"

Before: "~에 의해 생성된"
After:  "~가 만든"

# AI 관용구 (Category D)
Before: "결론적으로, 이는 시사하는 바가 크다"
After:  (삭제 또는 구체적 문장으로 대체)

# Hedging (Category G)
Before: "도움이 될 수 있을 것으로 판단된다"
After:  "도움이 된다"

# 형식명사 (Category I)
Before: "검토할 필요가 있다"
After:  "검토해야 한다"
```

## Strict Mode: Force or Check

Force strict mode explicitly:
```
이 글 --strict 옵션으로 윤문해줘:
[텍스트]
```

Text over 8,000 characters auto-escalates to strict mode regardless of flags.

## Agent Reference

| Agent | Mode | Role |
|-------|------|------|
| `humanize-monolith` | Fast (default) | All-in-one: detect + rewrite + self-verify |
| `ai-tell-detector` | Strict | Span-level JSON detection report |
| `korean-style-rewriter` | Strict | Surgery-based rewriting with change-rate monitor |
| `content-fidelity-auditor` | Strict | 13-point meaning equivalence audit |
| `naturalness-reviewer` | Strict | Residual AI tells + over-rewrite check, grades A–D |
| `korean-ai-tell-taxonomist` | Standalone | Taxonomy (SSOT) management, new pattern review |
| `humanize-web-architect` | Optional | Next.js 15 + Vercel web service design |

## Key Reference Files

```
.claude/skills/humanize-korean/references/
├── ai-tell-taxonomy.md      # Full 40+ pattern taxonomy (SSOT)
├── rewriting-playbook.md    # Prescriptions per pattern
├── quick-rules.md           # Slim rulebook for monolith (~150 lines)
└── web-service-spec.md      # Web service architecture spec

.claude/commands/
├── humanize.md              # /humanize command definition
└── humanize-redo.md         # /humanize-redo command definition
```

## Updating the Taxonomy

To propose new AI-tell patterns, invoke the taxonomist agent:
```
새로운 AI 티 패턴 추가 검토해줘: [패턴 설명]
```

The `korean-ai-tell-taxonomist` agent validates, categorizes, assigns severity, and promotes to SSOT if approved.

## Troubleshooting

**Skill not activating (behaves like plain Claude Code)**
→ You launched `claude` outside the `im-not-ai` directory. `cd im-not-ai` first, then `claude`.

**Run taking 20+ minutes**
→ You may be on v1.2–v1.4 code. Pull latest (`git pull`) — v1.5 defaults to the fast monolith path and targets 2–3 min for ≤5,000 chars.

**Output grade C or D, want better results**
→ Ask for "2차 윤문해줘" — this auto-escalates to strict mode for a more thorough pass.

**Over-rewrite warning (30%+ changed)**
→ Say "윤문 강도 낮춰줘" or "원문 톤을 더 살려줘" to rerun with conservative settings.

**Proper noun or number changed incorrectly**
→ This is a bug. Report the specific span and say "원문 복원해줘 — [해당 부분]". The Do-NOT list is hardcoded; any change to protected content triggers a rollback.

**Want only specific categories fixed**
→ "번역투(A)만 윤문해줘" or "관용구(D)와 형식명사(I)만 처리해줘" — strict mode handles category-scoped reruns.

**`_workspace/` getting cluttered**
→ Each run creates a new dated subfolder automatically. Safe to delete old ones: `rm -rf _workspace/2026042*`
```
