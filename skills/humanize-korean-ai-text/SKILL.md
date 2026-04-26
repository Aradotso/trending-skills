```markdown
---
name: humanize-korean-ai-text
description: Remove AI-tell patterns from Korean text using the im-not-ai Claude Code skill harness, preserving meaning while naturalizing style, rhythm, and phrasing.
triggers:
  - "AI 티 없애줘"
  - "GPT 문체 제거해줘"
  - "사람이 쓴 것처럼 윤문해줘"
  - "번역투 제거해줘"
  - "한글 AI 윤문해줘"
  - "remove AI tell from Korean text"
  - "humanize this Korean AI-generated text"
  - "ChatGPT 글 자연스럽게 고쳐줘"
---

# Humanize KR — 한글 AI 티 제거기 Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code skill harness that removes AI-tell patterns from Korean text (ChatGPT, Claude, Gemini output) without changing meaning, facts, or intent. Targets 10 categories × 40+ sub-patterns including translation-isms, mechanical parallelism, AI clichés, passive-voice overuse, and visual decoration abuse.

---

## Installation

```bash
git clone https://github.com/epoko77-ai/im-not-ai.git
cd im-not-ai
claude
```

> **Critical:** Always launch `claude` from inside the `im-not-ai` directory. Skills are loaded from `.claude/` relative to the working directory — running Claude Code from another location loads no skills.

---

## Quick Start

Paste AI-generated Korean text and use any natural trigger phrase:

```
이 AI 글 자연스럽게 윤문해줘:

[붙여넣기]
```

Or use the slash command:

```
/humanize [텍스트 또는 파일 경로]
```

Results are saved to `_workspace/{date-number}/`.

---

## Two Operating Modes

### Fast Mode (default, ≤5,000 chars, ~2–3 min)

Single-agent `humanize-monolith` handles detect → rewrite → self-verify in one call (4–5 tool calls capped).

**Output files:**

| File | Contents |
|------|----------|
| `01_input.txt` | Original text verbatim |
| `final.md` | Humanized output |
| `summary.md` | Metrics, category detections before/after, self-verify 6-point checklist, grade, change highlights |

### Strict Mode (`--strict` flag or auto-upgrade at 8,000+ chars)

Five-agent pipeline with separate outputs per stage.

**Trigger explicitly:**
```
/humanize --strict [텍스트]
```

**Output files:**

| File | Contents |
|------|----------|
| `01_input.txt` | Original verbatim |
| `02_detection.json` | Span-level detection report (position, category, severity) |
| `03_rewrite.md` | Rewritten draft |
| `04_fidelity_audit.json` | 13-point meaning-equivalence audit |
| `05_naturalness_review.json` | Residual AI-tell re-scan + over-humanization check |
| `final.md` | Accepted final output |
| `summary.md` | Score, grade, key changes |

---

## Slash Commands

### `/humanize`

```
/humanize 텍스트 또는 파일 경로 [옵션]
```

**Inline options (natural language appended):**

```
/humanize ./draft.md 장르: 칼럼
/humanize ./draft.md 강도: 적극
/humanize ./draft.md 최소심각도: S1
/humanize ./draft.md --strict
```

### `/humanize-redo`

Re-run with targeted instructions on the previous result:

```
/humanize-redo "번역투만 다시"
/humanize-redo "관용구 카테고리만 재처리"
/humanize-redo "윤문 강도 낮춰줘"
```

---

## Agents Reference

| Agent | Mode | Role |
|-------|------|------|
| `humanize-monolith` | Fast (default) | Single-call: detect + rewrite + self-verify |
| `ai-tell-detector` | Strict | Span-level JSON detection report |
| `korean-style-rewriter` | Strict | Surgical rewrite based on findings, change-rate monitoring |
| `content-fidelity-auditor` | Strict | 13-point meaning-equivalence audit, triggers rollback |
| `naturalness-reviewer` | Strict | Residual AI-tell + over-humanization check, grades A–D |
| `korean-ai-tell-taxonomist` | On-demand | Manages SSOT taxonomy, promotes new patterns |
| `humanize-web-architect` | Optional | Next.js 15 + Vercel web service design |

---

## AI-Tell Taxonomy (10 Categories)

| ID | Category | Key Patterns |
|----|----------|--------------|
| A | 번역투 | `~를 통해`, `~에 대해`, `~에 있어서`, `~되어진다`, `가지고 있다` |
| B | 영어 인용 과다 | Unnecessary parenthetical English, translatable terms kept in English |
| C | 구조적 AI 패턴 | Mechanical `첫째/둘째/셋째`, excessive bullets, headings, emoji |
| D | AI 특유 관용구 | `결론적으로`, `시사하는 바가 크다`, `주목할 만하다`, `혁신적인` |
| E | 리듬 균일성 | Low sentence-length variance, repeated ending patterns |
| F | 수식·중복 | `매우`, `정말`, synonym double-modification, `~적/~성/~화` overuse |
| G | Hedging 남용 | `~할 수 있을 것으로 보인다` multi-layer hedges |
| H | 접속사 남발 | Sentence-initial `또한/따라서/즉/나아가` chains |
| I | 형식명사 과다 | `것이다`, `점`, `수`, `바`, `~할 필요가 있다` |
| J | 시각 장식 남용 | Excessive **bold**, `"quotes"`, em-dash `—` |

Full taxonomy + prescriptions: `.claude/skills/humanize-korean/references/ai-tell-taxonomy.md`  
Rewriting playbook: `.claude/skills/humanize-korean/references/rewriting-playbook.md`

---

## Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **S1** 결정적 | One occurrence = AI confirmed | Always remove |
| **S2** 강함 | 1–2 OK, 3+ = problem | Remove on repeat |
| **S3** 약함 | Only problematic when stacked | Remove in clusters |

---

## Quality Grades (post-rewrite)

| Grade | Criteria |
|-------|----------|
| **A** | S1: 0, S2: ≤2, improvement ≥70% |
| **B** | S1: 0, S2: ≤4, improvement ≥50% |
| **C** | S1: 1–2 or 2+ over-humanization signals → trigger round 2 |
| **D** | S1: 3+ or severe over-humanization → escalate to human review |

---

## Four Core Rules (철칙)

1. **의미 불변** — Facts, claims, numbers, proper nouns, direct quotes: 100% preserved.
2. **근거 기반** — Only detected spans are touched; undetected text is left alone.
3. **장르 유지** — A column stays a column; a report stays a report.
4. **과윤문 금지** — Change rate >30%: warning. >50%: forced stop.

---

## Do-NOT List (탐지·윤문 제외 대상)

These are never modified regardless of patterns found:

- Numbers, units, dates
- Proper nouns, names, product names, model names
- Direct quotations inside `"큰따옴표"`
- Legal / regulatory text
- Unavoidable academic terminology

---

## Common Usage Patterns

### Pattern 1 — Direct paste, Fast Mode

```
이 ChatGPT 초안 사람이 쓴 것처럼 윤문해줘:

AI 기술을 통해 업무 효율을 높일 수 있다는 점은 매우 주목할 만하다.
첫째, 자동화를 통해 반복 작업을 줄일 수 있다. 둘째, 데이터 분석에
있어서 정확성을 높일 수 있다. 결론적으로, 이는 시사하는 바가 크다.
```

Expected `final.md` output (illustrative):

```
AI로 업무 효율이 높아졌다. 반복 작업이 줄고 데이터 분석이 정확해진다.
```

### Pattern 2 — File input with genre hint

```
/humanize ./article-draft.md 장르: 신문 칼럼 강도: 보수적
```

### Pattern 3 — Strict mode for long documents

```
/humanize ./long-report.md --strict
```

Or let it auto-upgrade (8,000+ chars triggers Strict automatically).

### Pattern 4 — Category-targeted redo

After receiving results:

```
/humanize-redo "번역투(A카테고리)만 더 손봐줘. 나머지는 그대로 둬."
```

### Pattern 5 — Reduce intensity

```
/humanize-redo "윤문 강도 낮춰줘. S1 패턴만 제거하고 나머지 건드리지 마."
```

### Pattern 6 — Second-pass refinement

```
2차 윤문해줘
```

Automatically runs Strict mode on `final.md` from the previous run.

---

## Python Integration (Programmatic Use)

The skill is Claude Code-native, but you can drive it programmatically via the Claude Code SDK or subprocess:

```python
import subprocess
import json
from pathlib import Path

def humanize_korean(text: str, strict: bool = False, workspace: str = ".") -> dict:
    """
    Run im-not-ai humanizer on Korean text via Claude Code CLI.
    
    Args:
        text: Korean AI-generated text to humanize
        strict: Use strict 5-agent pipeline (auto-enabled for 8000+ chars)
        workspace: Path to the im-not-ai repo directory
        
    Returns:
        dict with 'final' (humanized text) and 'summary' (metrics)
    """
    input_file = Path(workspace) / "_tmp_input.txt"
    input_file.write_text(text, encoding="utf-8")
    
    command = f"/humanize {input_file}"
    if strict or len(text) >= 8000:
        command += " --strict"
    
    result = subprocess.run(
        ["claude", "--print", command],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Claude Code error: {result.stderr}")
    
    # Find most recent workspace output
    workspaces = sorted(Path(workspace, "_workspace").glob("*/final.md"))
    if not workspaces:
        raise FileNotFoundError("No final.md found in _workspace/")
    
    latest = workspaces[-1].parent
    return {
        "final": (latest / "final.md").read_text(encoding="utf-8"),
        "summary": (latest / "summary.md").read_text(encoding="utf-8"),
        "detection": json.loads((latest / "02_detection.json").read_text(encoding="utf-8"))
                     if (latest / "02_detection.json").exists() else None,
    }


# Usage
if __name__ == "__main__":
    ai_text = """
    AI 기술을 통해 효율을 높일 수 있다는 점은 주목할 만하다.
    결론적으로, 이는 시사하는 바가 크다.
    """
    
    output = humanize_korean(ai_text, workspace="/path/to/im-not-ai")
    print(output["final"])
```

### Batch Processing

```python
from pathlib import Path
import time

def batch_humanize(input_dir: str, output_dir: str, workspace: str) -> None:
    """Humanize all .txt and .md files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file in input_path.glob("*.{txt,md}"):
        print(f"Processing {file.name}...")
        text = file.read_text(encoding="utf-8")
        
        try:
            result = humanize_korean(text, workspace=workspace)
            out_file = output_path / f"{file.stem}_humanized.md"
            out_file.write_text(result["final"], encoding="utf-8")
            print(f"  ✓ Saved to {out_file}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Respect rate limits between calls
        time.sleep(5)
```

### Parse Detection Report

```python
import json
from pathlib import Path
from typing import List, Dict

def parse_detection_report(workspace_run_dir: str) -> List[Dict]:
    """
    Parse the span-level detection JSON from a Strict mode run.
    
    Returns list of findings: [{span, category, severity, suggestion}, ...]
    """
    detection_file = Path(workspace_run_dir) / "02_detection.json"
    
    if not detection_file.exists():
        raise FileNotFoundError("Run in --strict mode to get detection.json")
    
    data = json.loads(detection_file.read_text(encoding="utf-8"))
    findings = data.get("findings", [])
    
    # Group by severity
    s1 = [f for f in findings if f.get("severity") == "S1"]
    s2 = [f for f in findings if f.get("severity") == "S2"]
    s3 = [f for f in findings if f.get("severity") == "S3"]
    
    print(f"S1 (critical): {len(s1)}")
    print(f"S2 (strong):   {len(s2)}")
    print(f"S3 (weak):     {len(s3)}")
    
    return findings


# Example: filter only S1 findings for review
findings = parse_detection_report("./_workspace/2026-04-26-001")
critical = [f for f in findings if f["severity"] == "S1"]
for finding in critical:
    print(f"[{finding['category']}] '{finding['span']}' → {finding['suggestion']}")
```

---

## Workspace Directory Structure

```
im-not-ai/
├── .claude/
│   ├── commands/
│   │   ├── humanize.md          # /humanize slash command definition
│   │   └── humanize-redo.md     # /humanize-redo slash command definition
│   └── skills/
│       └── humanize-korean/
│           ├── agents/
│           │   ├── humanize-monolith.md
│           │   ├── ai-tell-detector.md
│           │   ├── korean-style-rewriter.md
│           │   ├── content-fidelity-auditor.md
│           │   ├── naturalness-reviewer.md
│           │   ├── korean-ai-tell-taxonomist.md
│           │   └── humanize-web-architect.md
│           └── references/
│               ├── ai-tell-taxonomy.md      # Full 40+ pattern SSOT
│               ├── rewriting-playbook.md    # Per-pattern prescriptions
│               ├── quick-rules.md           # ~150-line slim rulebook (monolith)
│               └── web-service-spec.md      # Next.js web expansion spec
├── _workspace/                  # Auto-created per run
│   └── {date}-{number}/
│       ├── 01_input.txt
│       ├── 02_detection.json    # Strict only
│       ├── 03_rewrite.md        # Strict only
│       ├── 04_fidelity_audit.json  # Strict only
│       ├── 05_naturalness_review.json  # Strict only
│       ├── final.md
│       └── summary.md
└── assets/
    └── social-preview.png
```

---

## Troubleshooting

### Skill not activating

**Symptom:** Claude Code responds as a generic assistant, no humanization pipeline runs.

**Fix:** Ensure you launched `claude` from inside the `im-not-ai` directory:
```bash
cd /path/to/im-not-ai
claude
```

### Wall-clock time too long

**Symptom:** Fast mode taking >10 minutes.

**Cause:** Input exceeded 5,000 chars silently triggering Strict mode, or Claude Code is re-loading large context files.

**Fix:**
- Check character count: `wc -m your-file.txt`
- For ≤5,000 chars, use Fast (no flag)
- For large files, explicitly pass `--strict` so the pipeline is intentional

### Grade C or D output

**Symptom:** `summary.md` shows grade C/D.

**Fix:** Request second pass:
```
2차 윤문해줘
```
Or target specific remaining issues:
```
/humanize-redo "S1 패턴인 '결론적으로', '시사하는 바가 크다' 제거해줘"
```

### Change rate warning (>30%)

**Symptom:** `summary.md` shows change-rate warning.

**Cause:** Over-humanization — the rewriter modified too much.

**Fix:**
```
/humanize-redo "윤문 강도 낮춰줘. S1만 제거하고 나머지는 원문 유지."
```

### `final.md` not found

**Symptom:** `_workspace/` is empty or missing expected files.

**Cause:** Run failed mid-way (API timeout, context overflow).

**Fix:**
- Check Claude Code session logs
- For very long texts (>8,000 chars), ensure `--strict` is explicit
- Re-run: `/humanize [파일경로]`

### Meaning changed unexpectedly

**Symptom:** `04_fidelity_audit.json` shows failed items, or you notice factual changes.

**Fix:** The `content-fidelity-auditor` should have caught this and triggered rollback. If it slipped through:
```
이 수정은 의미가 바뀌었어. 원문의 [특정 내용]을 그대로 유지하면서 다시 윤문해줘.
```

---

## Key Rewriting Rules (Quick Reference)

The most important S1 patterns and their corrections:

```
# Category A — 번역투

"~를 통해"        → 구체적 수단으로 대체하거나 삭제
"~에 대해"        → "~을", "~를", 또는 재구성
"~에 있어서"      → 삭제 또는 "~에서"
"~되어진다"       → "~된다" (이중 피동 제거)
"가지고 있다"     → "있다", "~이다"
"~에 의해"        → "~가", "~로"

# Category D — AI 관용구

"결론적으로"      → 삭제 또는 내용으로 대체
"시사하는 바가 크다" → 삭제
"주목할 만하다"   → 삭제 또는 구체적 이유로 대체
"혁신적인"        → 삭제 또는 구체적 기술로 대체

# Category C — 구조적 패턴

"첫째, 둘째, 셋째" → 자연스러운 연결어 또는 단락 분리
과도한 불릿 리스트 → 산문으로 통합
```

Full prescriptions: `.claude/skills/humanize-korean/references/rewriting-playbook.md`

---

## Web Service Extension (Optional)

`humanize-web-architect` agent designs a full web product:

- **Stack:** Next.js 15 App Router + Vercel Fluid Compute + AI Gateway
- **UX:** 4 screens — Input → Detection highlights → Left/right diff → Copy output
- **Roadmap:** v0 MVP (anonymous) → v1 (auth + history) → v2 (Pro/Team + API + webhooks) → v3 (Chrome Extension) → v4 (Japanese/Chinese expansion)

Spec: `.claude/skills/humanize-korean/references/web-service-spec.md`

Invoke:
```
humanize-web-architect로 웹 서비스 설계해줘
```

---

## Taxonomy Management

To propose or promote a new AI-tell pattern:

```
korean-ai-tell-taxonomist로 새 패턴 심사해줘: "~한 바 있다"
```

The taxonomist agent evaluates the pattern against existing taxonomy, assigns category/severity, and updates `ai-tell-taxonomy.md` if approved.
```
