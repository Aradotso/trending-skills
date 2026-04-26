```markdown
---
name: humanize-korean-ai-text
description: Remove AI-tell patterns from Korean text using the im-not-ai Claude Code skill harness — detects 40+ patterns across 10 categories and rewrites with surgical precision while preserving meaning.
triggers:
  - "AI 티 없애줘"
  - "GPT 문체 제거해줘"
  - "사람이 쓴 것처럼 윤문해줘"
  - "번역투 제거해줘"
  - "한글 AI 윤문해줘"
  - "remove AI writing patterns from Korean text"
  - "humanize this Korean AI-generated text"
  - "ChatGPT 글 자연스럽게 고쳐줘"
---

# Humanize KR — 한글 AI 티 제거기 (im-not-ai)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code skill harness that removes AI-tell patterns from Korean text — fixing translation-style phrasing, mechanical structure, and AI clichés while **never changing facts, numbers, or meaning**.

## What It Does

Korean AI text (ChatGPT, Claude, Gemini output) contains distinctive patterns inherited from English translation. This skill detects and rewrites them:

| Before (AI-tell) | After (Natural Korean) |
|---|---|
| "AI 기술을 **통해** 효율을 높**일 수 있다**" | "AI로 효율을 높인다" |
| "**결론적으로**, 이는 **시사하는 바가 크다**" | *(삭제)* |
| "이에 **있어서** 중요한 **점은**" | "여기서 중요한 건" |
| "~**에 의해** 생성된" | "~가 만든" |

**4 Inviolable Rules:**
1. **의미 불변** — Facts, claims, numbers, proper nouns, direct quotes: 100% preserved
2. **근거 기반** — Only detected spans get edited; undetected passages untouched
3. **장르 유지** — Column stays column; report stays report
4. **과윤문 금지** — >30% change rate = warning; >50% = forced stop

## Installation

### Prerequisites

[Claude Code](https://claude.com/claude-code) must be installed:

```bash
# Verify Claude Code is available
claude --version
```

### Clone and Enter

```bash
git clone https://github.com/epoko77-ai/im-not-ai.git
cd im-not-ai

# Launch Claude Code from INSIDE this directory (critical!)
claude
```

> ⚠️ You must run `claude` from within the `im-not-ai` directory. Running from elsewhere means the skills won't load.

## Usage — Three Methods

### Method A: Natural Language (Easiest)

Paste any of these trigger phrases + your text:

```
이 AI 글 자연스럽게 윤문해줘:

[paste your ChatGPT/Claude/Gemini output here]
```

Any of these phrases trigger the skill automatically:
- `"AI 티 없애줘"`
- `"GPT 문체 제거해줘"`
- `"사람이 쓴 것처럼 윤문해줘"`
- `"번역투 제거"`
- `"한글 AI 윤문"`

### Method B: Slash Command

```
/humanize [text or file path]

# With options (natural language at end)
/humanize ./draft.md 장르: 칼럼 강도: 적극 최소심각도: S1

# Redo with specific focus
/humanize-redo "번역투만 다시"
/humanize-redo "관용구 카테고리만 재처리"
```

### Method C: Strict Mode (for long text or precision work)

```bash
# Force strict mode explicitly
/humanize --strict [text]

# Auto-triggers for 8,000+ character input
# Uses 5-agent pipeline instead of monolith
```

## Two Processing Modes

### Fast Mode (Default, ≤5,000 chars, ~3 min)

Single `humanize-monolith` agent call: detect → rewrite → self-verify in one pass.

```
_workspace/{date-number}/
├── 01_input.txt      # Original text, untouched
├── final.md          # Rewritten output
└── summary.md        # Metrics, before/after, grade, highlights
```

### Strict Mode (`--strict` or 8,000+ chars auto-upgrade)

5-agent pipeline with separate audit files:

```
_workspace/{date-number}/
├── 01_input.txt            # Original
├── 02_detection.json       # AI-tell spans (position, category, severity)
├── 03_rewrite.md           # Rewritten draft
├── 04_fidelity_audit.json  # 13-point meaning-preservation audit
├── 05_naturalness_review.json  # Residual AI-tell + over-rewrite check
├── final.md                # Accepted final
└── summary.md              # Score, grade, key changes
```

## AI-Tell Pattern Taxonomy

### 10 Categories × 40+ Patterns

| ID | Category | Key Patterns |
|----|----------|-------------|
| **A** | 번역투 | `~를 통해`, `~에 대해`, `~에 있어서`, 이중 피동 `~되어진다`, `가지고 있다` |
| **B** | 영어 인용 과다 | 과도한 괄호 병기, 번역 가능한 영어 그대로 사용 |
| **C** | 구조적 AI 패턴 | 기계적 `첫째/둘째/셋째`, 과도한 불릿·헤딩·이모지 |
| **D** | AI 특유 관용구 | `결론적으로`, `시사하는 바가 크다`, `주목할 만하다`, `혁신적인` |
| **E** | 리듬 균일성 | 문장 길이 표준편차 낮음, 동일 종결어미 반복 |
| **F** | 수식·중복 | `매우`, `정말`, 동의어 이중 수식, `~적/~성/~화` 남발 |
| **G** | Hedging 남용 | `~할 수 있을 것으로 보인다` 다중 완곡 |
| **H** | 접속사 남발 | 문두 `또한/따라서/즉/나아가` 연속 |
| **I** | 형식명사 과다 | `것이다`, `점`, `수`, `바`, `~할 필요가 있다` |
| **J** | 시각 장식 남용 | 과도한 **볼드**, `"따옴표"`, 대시(—) 남발 |

### Severity Levels

- **S1 결정적**: Appears once → AI confirmed. Always remove.
- **S2 강함**: 1-2 occurrences OK; 3+ must be removed.
- **S3 약함**: Only problematic when co-occurring with other patterns.

### Quality Grades (Post-Rewrite)

| Grade | Criteria |
|-------|----------|
| **A** | S1: 0, S2: ≤2, improvement ≥70% |
| **B** | S1: 0, S2: ≤4, improvement ≥50% |
| **C** | S1: 1-2 or 2 over-rewrite signals → trigger round 2 |
| **D** | S1: 3+ or severe over-rewrite → human review required |

## The 7 Agents

| Agent | Mode | Role |
|-------|------|------|
| `humanize-monolith` | **Fast (default)** | Single-call detect + rewrite + self-verify (4-5 tool calls capped) |
| `ai-tell-detector` | Strict | JSON detection report with span positions |
| `korean-style-rewriter` | Strict | Surgical rewrite from findings, monitors change rate |
| `content-fidelity-auditor` | Strict | 13-point meaning-equivalence audit, triggers rollback |
| `naturalness-reviewer` | Strict | Re-runs detection, grades A-D, flags over-rewrite |
| `korean-ai-tell-taxonomist` | Separate command | Manages SSOT taxonomy, promotes new patterns |
| `humanize-web-architect` | Optional | Next.js 15 + Vercel web service design |

## Refinement Commands

After getting results, speak naturally to refine:

```
# Redo a specific section
"이 문단만 다시 윤문해줘"

# Target a specific category
"번역투만 더 손봐줘"
"관용구만 다시"

# Adjust intensity
"윤문 강도 낮춰줘"          # Conservative — S1 patterns only
"원문 톤을 더 살려줘"       # Lower change rate ceiling
"2차 윤문해줘"              # Polish current result further
```

## Do-NOT Touch List

The skill will never modify:
- Numbers, units, dates
- Proper nouns, person names, product names, model names
- Text inside `"직접 인용"` (direct quotes)
- Legal/regulatory article text
- Academic terms (when unavoidable)

## Configuration Reference

### Workspace Structure

```
im-not-ai/
├── .claude/
│   ├── skills/humanize-korean/
│   │   ├── agents/           # 7 agent definitions
│   │   └── references/
│   │       ├── ai-tell-taxonomy.md    # Full 40+ pattern SSOT
│   │       ├── rewriting-playbook.md  # Prescription per pattern
│   │       ├── quick-rules.md         # Slim rulebook for monolith (~150 lines)
│   │       └── web-service-spec.md    # Web expansion spec
│   └── commands/
│       ├── humanize.md        # /humanize command definition
│       └── humanize-redo.md   # /humanize-redo command definition
└── _workspace/               # Auto-created output directory
    └── {date-number}/
        ├── 01_input.txt
        ├── final.md
        └── summary.md
```

### Genre/Domain Options

Pass as natural language after the text:

```
/humanize ./essay.md 장르: 칼럼
/humanize ./report.md 장르: 학술 강도: 보수
/humanize ./blog.md 장르: 블로그 최소심각도: S2
```

## Common Patterns for Developers

### Programmatic Integration (Python wrapper example)

```python
import subprocess
import json
from pathlib import Path

def humanize_korean(text: str, strict: bool = False) -> dict:
    """
    Run im-not-ai humanization on Korean text.
    
    Args:
        text: Korean AI-generated text to humanize
        strict: Use 5-agent pipeline instead of monolith
    
    Returns:
        dict with 'final' (rewritten text) and 'summary' (metrics)
    """
    # Write input to temp file
    input_path = Path("_workspace/temp_input.txt")
    input_path.parent.mkdir(exist_ok=True)
    input_path.write_text(text, encoding="utf-8")
    
    # Build command
    cmd_parts = ["/humanize", str(input_path)]
    if strict:
        cmd_parts.append("--strict")
    
    prompt = " ".join(cmd_parts)
    
    result = subprocess.run(
        ["claude", "-p", prompt],
        cwd="/path/to/im-not-ai",  # Must run from project root
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Humanization failed: {result.stderr}")
    
    # Find latest workspace output
    workspace = Path("_workspace")
    latest = sorted(workspace.iterdir())[-1]
    
    return {
        "final": (latest / "final.md").read_text(encoding="utf-8"),
        "summary": (latest / "summary.md").read_text(encoding="utf-8"),
        "input": text,
    }


# Usage
result = humanize_korean("""
AI 기술을 통해 다양한 분야에서 혁신적인 변화가 일어나고 있다.
첫째, 의료 분야에서는 진단 정확도를 높일 수 있다.
둘째, 교육 분야에서는 맞춤형 학습이 가능해진다.
결론적으로, 이는 우리 사회에 시사하는 바가 크다.
""")

print(result["final"])
# AI로 여러 분야가 바뀌고 있다.
# 의료에서는 진단이 더 정확해졌고,
# 교육에서는 각자 수준에 맞는 수업이 가능해졌다.
```

### Batch Processing Multiple Files

```python
import subprocess
from pathlib import Path

def batch_humanize(input_dir: str, output_dir: str):
    """Process all .txt and .md files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
    
    for file in files:
        print(f"Processing: {file.name}")
        
        text = file.read_text(encoding="utf-8")
        char_count = len(text)
        
        # Auto-select mode based on length
        strict_flag = "--strict" if char_count > 8000 else ""
        prompt = f"/humanize {file} {strict_flag}".strip()
        
        result = subprocess.run(
            ["claude", "-p", prompt],
            cwd="/path/to/im-not-ai",
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        if result.returncode == 0:
            # Copy final.md to output dir with original filename
            workspace = sorted(Path("_workspace").iterdir())[-1]
            final = (workspace / "final.md").read_text(encoding="utf-8")
            (output_path / file.name).write_text(final, encoding="utf-8")
            print(f"  ✓ Saved to {output_path / file.name}")
        else:
            print(f"  ✗ Failed: {result.stderr[:100]}")


batch_humanize("./ai_drafts", "./humanized_output")
```

### Parse Detection JSON (Strict Mode)

```python
import json
from pathlib import Path

def analyze_ai_tells(workspace_dir: str) -> dict:
    """Parse the detection report from a strict-mode run."""
    detection_file = Path(workspace_dir) / "02_detection.json"
    
    with open(detection_file, encoding="utf-8") as f:
        detection = json.load(f)
    
    # Group by severity
    by_severity = {"S1": [], "S2": [], "S3": []}
    by_category = {}
    
    for span in detection.get("spans", []):
        sev = span.get("severity", "S3")
        by_severity[sev].append(span)
        
        cat = span.get("category", "unknown")
        by_category.setdefault(cat, []).append(span)
    
    return {
        "total_findings": len(detection.get("spans", [])),
        "by_severity": {k: len(v) for k, v in by_severity.items()},
        "by_category": {k: len(v) for k, v in by_category.items()},
        "critical_spans": by_severity["S1"],  # Must-fix items
    }


# After a strict mode run:
analysis = analyze_ai_tells("_workspace/2026-04-26-001")
print(f"Critical (S1): {analysis['by_severity']['S1']} patterns found")
print(f"Category breakdown: {analysis['by_category']}")
```

### Adding New Patterns to Taxonomy

```
# Invoke the taxonomist agent directly
claude -p "새 패턴 심사 요청: '~와 같은 맥락에서' — 문두에서 불필요한 연결어로 사용됨. 심각도 S2 제안"
```

Or as a slash command inside Claude Code:
```
/taxonomy-review "~와 같은 맥락에서" 심각도: S2 카테고리: H
```

## Troubleshooting

### Skill Not Loading

```bash
# Wrong — skill won't load
cd ~/Documents
claude  # ← opens generic Claude Code

# Right — must be in project root
cd /path/to/im-not-ai
claude  # ← loads humanize-korean skill
```

### Output Taking Too Long

v1.5 fixed the 25-minute wall-clock regression. If still slow:
- Fast mode (≤5,000 chars): should complete in 2-3 minutes
- Strict mode: longer by design (5-agent pipeline)
- Check you're not accidentally triggering strict mode on short text

### Over-Rewrite Warning

If you see a >30% change rate warning:
```
"윤문 강도 낮춰줘"          # Reduce intensity
"S1 패턴만 제거해줘"         # Only critical patterns
"원문 유지 비율 높여줘"       # Preserve more original text
```

### Grade C or D Results

```
# Grade C — triggers auto round-2, or manually:
"2차 윤문해줘"

# Grade D — requires human review, but you can try:
"S1 패턴만 먼저 제거하고 결과 보여줘"
```

### Content Changed Incorrectly

The `content-fidelity-auditor` (strict mode) runs 13-point checks. In fast mode, self-verification catches most issues. If meaning was altered:
```
"원문의 [구체적 내용] 부분이 바뀌었어. 롤백해줘"
"이 수치/고유명사가 바뀌면 안 돼. 다시 윤문해줘"
```

### Multiple Runs Stay Separate

Each run creates a new `_workspace/{date-number}/` folder — previous results are never overwritten.

## Reference Files

| File | Purpose |
|------|---------|
| `.claude/skills/humanize-korean/references/ai-tell-taxonomy.md` | Full SSOT: 40+ patterns, examples, prescriptions |
| `.claude/skills/humanize-korean/references/rewriting-playbook.md` | Pattern-by-pattern rewrite instructions |
| `.claude/skills/humanize-korean/references/quick-rules.md` | Slim rulebook for monolith fast path (~150 lines) |
| `.claude/skills/humanize-korean/references/web-service-spec.md` | Next.js 15 web service expansion spec |
| `.claude/commands/humanize.md` | `/humanize` command definition |
| `.claude/commands/humanize-redo.md` | `/humanize-redo` command definition |

## Version History

- **v1.5** (2026-04-26) — Rolled back v1.2-v1.4; added `humanize-monolith` fast path; 5,000-char target 2-3 min wall-clock
- **v1.1** — Base 5-agent pipeline (still used as strict mode)
- **v1.2-v1.4** — Voice profile, candidate pool, model distribution experiments (abandoned — caused 25-min regressions)
```
