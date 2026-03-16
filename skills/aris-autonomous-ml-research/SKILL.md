---
name: aris-autonomous-ml-research
description: Orchestrate autonomous ML research workflows with cross-model review loops, idea discovery, and experiment automation using Claude Code and Codex MCP
triggers:
  - autonomous research pipeline
  - auto review loop
  - idea discovery research
  - paper writing automation
  - cross-model review
  - overnight ML experiments
  - research while sleeping
  - ARIS research workflow
---

# ARIS — Autonomous ML Research In Sleep

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection

ARIS (Auto-Research-In-Sleep) turns Claude Code into an autonomous ML research engine. It chains **idea discovery → cross-model review loops → paper writing → compiled PDF** into hands-off overnight pipelines. Claude Code drives execution while an external model (Codex/GPT-5.4, GLM, DeepSeek, Kimi, etc.) acts as adversarial reviewer — breaking self-play blind spots that single-model review cannot escape.

## What It Does

| Workflow | Trigger | What Runs |
|---|---|---|
| Idea Discovery | `/idea-discovery` | Literature survey → 8–12 ideas → novelty check → pilot GPU runs → ranked report |
| Auto Review Loop | `/auto-review-loop` | 4-round review/fix cycle, score tracked per round (e.g. 5/10 → 7.5/10) |
| Paper Writing | `/paper-writing` | Narrative → outline → figures → LaTeX → PDF → 2-round auto-improvement |
| Full Pipeline | `/research-pipeline` | Chains all three end-to-end from a single prompt |

## Installation

```bash
# 1. Clone and install skills
git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git
cp -r Auto-claude-code-research-in-sleep/skills/* ~/.claude/skills/

# 2. Install Codex MCP (cross-model reviewer)
npm install -g @openai/codex
codex setup          # set model to gpt-5.4 when prompted
claude mcp add codex -s user -- codex mcp-server

# 3. Verify MCP is connected
claude mcp list      # should show "codex" in the list
```

### Codex Model Configuration

The reviewer model is read from `~/.codex/config.toml`, **not** from skill files. Edit directly if needed:

```toml
# ~/.codex/config.toml
model = "gpt-5.4"        # recommended — most rigorous reviewer
# model = "gpt-5.3-codex"
# model = "gpt-5.2-codex"
# model = "o3"
```

## Core Workflows

### Workflow 1 — Idea Discovery

```
/idea-discovery "factorized gap in discrete diffusion language models"
```

Be specific — "NLP" produces weak ideas; "factorized gap in discrete diffusion LMs" targets a real research gap.

**What runs:**
1. Multi-source literature search (arXiv, Scholar, Zotero, Obsidian, local PDFs)
2. Claude brainstorms 8–12 candidate ideas
3. Codex reviewer cross-checks novelty against literature
4. Pilot GPU experiments on top candidates
5. Ranked idea report saved to `idea_discovery_report.md`

### Workflow 2 — Auto Review Loop

```
/auto-review-loop
```

Run from a directory containing your paper draft or experiment results.

**What runs:**
1. Claude submits current work to Codex reviewer
2. Codex returns structured critique with score `/10`
3. Claude implements fixes (experiments, writing, ablations)
4. Repeat up to 4 rounds or until score threshold met
5. Score curve saved to `docs/auto_review_score_curve.png`

### Workflow 3 — Paper Writing

```
/paper-writing "NARRATIVE_REPORT.md"
```

Point at a narrative markdown file describing your findings.

**What runs:**
1. Outline generation (sections, figures, tables)
2. Figure generation from experiment results
3. LaTeX source assembly
4. `pdflatex` compilation
5. 2-round auto-review-and-improve cycle
6. Final PDF + anti-hallucination BibTeX (fetched from DBLP/CrossRef)

### Full Pipeline

```
/research-pipeline "your research direction"
```

Chains Workflows 1 → 2 → 3 from a single prompt. Wake up to a scored, compiled paper.

## Inline Configuration Overrides

Append `— key: value` to any command:

```
/research-pipeline "topic" — AUTO_PROCEED: false
/research-pipeline "topic" — human checkpoint: true
/research-pipeline "topic" — arxiv download: true
/research-pipeline "topic" — AUTO_PROCEED: false, human checkpoint: true
```

| Parameter | Default | Effect |
|---|---|---|
| `AUTO_PROCEED` | `true` | `false` = pause at idea selection gate before committing GPU time |
| `human checkpoint` | `false` | `true` = pause after each review round for manual feedback |
| `arxiv download` | `false` | `true` = download full PDFs during literature survey (vs metadata only) |
| `DBLP_BIBTEX` | `true` | `false` = use LLM-generated BibTeX (not recommended — hallucination risk) |

## Alternative Model Combinations

No Claude or OpenAI API required — swap any OpenAI-compatible endpoint via the `llm-chat` MCP server:

```bash
# Install the bundled llm-chat MCP server
cd Auto-claude-code-research-in-sleep/mcp-servers/llm-chat
pip install -r requirements.txt

# Configure your provider
export LLM_CHAT_BASE_URL="https://open.bigmodel.cn/api/paas/v4"   # GLM-4
export LLM_CHAT_API_KEY="your-key"
export LLM_CHAT_MODEL="glm-4-plus"

# Add to Claude Code
claude mcp add llm-chat -s user -- python server.py
```

**Tested reviewer models:**

| Provider | Model | Notes |
|---|---|---|
| OpenAI | `gpt-5.4` | Recommended — most rigorous |
| Zhipu AI | `glm-4-plus` | Strong Chinese-language papers |
| MiniMax | `abab6.5s-chat` | Fast, cost-effective |
| Moonshot | `moonshot-v1-128k` | Kimi — long-context papers |
| DeepSeek | `deepseek-chat` | Code-heavy experiments |
| 01.ai | `yi-large` | LongCat — long context |

## Anti-Hallucination Citations

BibTeX is fetched from real databases by default — no manual flag needed:

```python
# skills/paper-writing/citation_fetcher.py pattern used internally
import requests

def fetch_bibtex_dblp(title: str) -> str | None:
    """Fetch real BibTeX from DBLP by paper title."""
    resp = requests.get(
        "https://dblp.org/search/publ/api",
        params={"q": title, "format": "json", "h": 1}
    )
    hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
    if not hits:
        return None
    key = hits[0]["info"].get("key", "")
    bib_resp = requests.get(f"https://dblp.org/rec/{key}.bib")
    return bib_resp.text if bib_resp.ok else None

def fetch_bibtex_crossref(doi: str) -> str | None:
    """Fallback: fetch BibTeX from CrossRef by DOI."""
    resp = requests.get(
        f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    )
    return resp.text if resp.ok else None
```

Disable with `— DBLP_BIBTEX: false` if working fully offline.

## Optional Integrations

### Zotero

```bash
# Install Zotero Better BibTeX plugin, then:
export ZOTERO_API_KEY="your-zotero-web-api-key"
export ZOTERO_LIBRARY_ID="your-library-id"
export ZOTERO_LIBRARY_TYPE="user"   # or "group"
```

Literature search will query your Zotero library before hitting arXiv.

### Obsidian

```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"
```

Skill will search markdown notes in the vault for related work before external queries.

### Feishu / Lark Notifications

```bash
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/your-token"
export FEISHU_MODE="push"   # off | push | interactive
```

| Mode | Behaviour |
|---|---|
| `off` | No notifications |
| `push` | One-way alerts: review scores, experiment completions, checkpoints |
| `interactive` | Mobile approval buttons at `AUTO_PROCEED: false` gates |

## Directory Layout After a Pipeline Run

```
your-project/
├── idea_discovery_report.md       # ranked ideas with novelty scores
├── NARRATIVE_REPORT.md            # auto-generated findings narrative
├── paper/
│   ├── main.tex                   # assembled LaTeX
│   ├── main.pdf                   # compiled output
│   ├── figures/                   # auto-generated plots
│   └── references.bib             # real BibTeX from DBLP/CrossRef
├── experiments/
│   ├── pilot_runs/                # idea-discovery GPU pilots
│   └── review_round_*/            # per-round experiment results
└── docs/
    └── auto_review_score_curve.png
```

## Python Integration Pattern

Trigger ARIS workflows programmatically from a Python script (e.g. a cron job or CI step):

```python
import subprocess
import json
from pathlib import Path

def run_aris_pipeline(
    research_direction: str,
    output_dir: str = ".",
    auto_proceed: bool = True,
    human_checkpoint: bool = False,
    arxiv_download: bool = False,
) -> dict:
    """
    Launch ARIS full pipeline via Claude Code CLI.
    Returns parsed score progression from the review curve JSON.
    """
    overrides = ", ".join([
        f"AUTO_PROCEED: {str(auto_proceed).lower()}",
        f"human checkpoint: {str(human_checkpoint).lower()}",
        f"arxiv download: {str(arxiv_download).lower()}",
    ])

    command = f'/research-pipeline "{research_direction}" — {overrides}'

    result = subprocess.run(
        ["claude", "--print", command],
        cwd=output_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ARIS pipeline failed:\n{result.stderr}")

    # Parse score progression if available
    score_json = Path(output_dir) / "docs" / "review_scores.json"
    if score_json.exists():
        return json.loads(score_json.read_text())

    return {"stdout": result.stdout}


# Example: nightly research job
if __name__ == "__main__":
    scores = run_aris_pipeline(
        research_direction="token-level uncertainty calibration in autoregressive LMs",
        output_dir="./nightly_research",
        auto_proceed=True,
        human_checkpoint=False,
    )
    print(f"Final review score: {scores.get('rounds', [{}])[-1].get('score')}/10")
```

## Skill Composition

ARIS ships 20 composable sub-skills. Chain them manually for custom workflows:

```
# Literature only
/literature-survey "topic"

# Brainstorm without pilot experiments
/idea-brainstorm "topic" — pilot experiments: false

# Single review round (no loop)
/single-review "path/to/draft.md"

# Proof-writing (community skill)
/proof-writer "theorem statement"

# Write paper from existing narrative, skip review
/paper-writing "NARRATIVE.md" — auto-review: false
```

## Troubleshooting

**Codex MCP not found**
```bash
claude mcp list                          # verify "codex" appears
codex setup                              # re-run setup if missing
claude mcp remove codex && \
  claude mcp add codex -s user -- codex mcp-server   # re-add
```

**Skills not loading in Claude Code**
```bash
ls ~/.claude/skills/                     # verify files copied
# Each skill must be a directory with SKILL.md inside
ls ~/.claude/skills/auto-review-loop/SKILL.md
```

**`pdflatex` not found during paper writing**
```bash
# macOS
brew install --cask mactex-no-gui
# Ubuntu/Debian
sudo apt install texlive-full
# Then retry — skill auto-detects pdflatex on PATH
```

**Reviewer returns empty critique**
Check `~/.codex/config.toml` — ensure `model` is set and your API key is valid:
```bash
codex "say hello"    # quick smoke test outside Claude Code
```

**GLM/DeepSeek reviewer not triggering**
Verify `llm-chat` MCP server is listed:
```bash
claude mcp list      # should show "llm-chat"
echo $LLM_CHAT_BASE_URL   # must be set in the shell that launches claude
```

**Score not improving after 4 rounds**
- Add `— human checkpoint: true` and inspect each round's critique file in `experiments/review_round_*/`
- Consider switching reviewer model — a different architecture surfaces different weaknesses
- Lower-level issues (bad data, flawed baseline) need manual intervention before another loop

## Community Skills

| Skill | Description |
|---|---|
| `proof-writer` | Rigorous theorem proof drafting with anti-hallucination citations |

Add your own skill: create `skills/your-skill-name/SKILL.md` and open a PR.

## Cross-Model Review — Why It Works

```
Claude Code (executor)          Codex / external LLM (reviewer)
─────────────────────          ───────────────────────────────
Fast, fluid code execution  ←→  Deliberate, rigorous critique
Broad context retention         Adversarial probing of blind spots
Narrative generation            Structural weakness detection
```

Single-model self-review falls into local minima — the same pattern-matching that generated the work also evaluates it. Cross-model review is adversarial: the reviewer actively probes weaknesses the executor didn't anticipate. The 1→2 model jump produces the largest quality gain; adding more reviewers yields diminishing returns.
