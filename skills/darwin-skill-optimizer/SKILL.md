```markdown
---
name: darwin-skill-optimizer
description: Autonomous skill optimization loop for Claude Code — evaluates, improves, tests, and ratchets SKILL.md files using an autoresearch-inspired evolutionary cycle.
triggers:
  - optimize my skills
  - improve all skills
  - run darwin skill optimizer
  - evaluate and improve skill files
  - optimize a specific skill
  - run skill optimization loop
  - apply darwin evolution to skills
  - ratchet my agent skills
---

# darwin-skill — Autonomous SKILL.md Optimizer

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

darwin-skill brings Andrej Karpathy's `autoresearch` loop to Agent Skill optimization. It evaluates every SKILL.md file across 8 weighted dimensions (100 pts total), proposes targeted improvements, tests them, and keeps only changes that measurably improve the score — a ratchet that never goes backwards.

---

## Installation

```bash
# via npx (recommended)
npx skills add alchaincyf/darwin-skill

# manual (no GitHub access)
# Download: https://pub-161ae4b5ed0644c4a43b5c6412287e03.r2.dev/skills/darwin-skill.zip
# Unzip → place SKILL.md at: ~/.claude/skills/darwin-skill/SKILL.md
```

Compatible with: **Claude Code**, Cursor, Codex, Trae, CodeBuddy, OpenClaw — any agent that reads `~/.claude/skills/` or equivalent.

---

## What It Does

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Phase 1    │────▶│   Phase 2    │────▶│   Phase 3     │
│  Inventory  │     │  Optimize    │     │  Report       │
│  + Score    │     │  (ratchet)   │     │  + Confirm    │
└─────────────┘     └──────────────┘     └───────────────┘
        │                  │
        │            ┌─────▼──────┐
        │            │ score(new) │
        │            │ > score(old│
        │            │ keep / rev │
        │            └────────────┘
```

**Key guarantee**: Every skill's score can only increase. Any change that doesn't improve the score is automatically `git revert`ed.

---

## Triggering Optimization

Once installed, speak naturally to your agent:

```
"optimize all skills"
"optimize the darwin-skill skill"
"run the darwin optimization loop on my nuwa skill"
"evaluate all my skill files and improve the weakest ones"
```

The agent will:
1. Discover all SKILL.md files in `~/.claude/skills/`
2. Score each across 8 dimensions
3. Propose + apply improvements (one at a time)
4. Keep or revert based on score delta
5. Pause after each skill for your confirmation before continuing

---

## The 8-Dimension Scoring Rubric (100 pts)

| Dimension | Max | Method |
|---|---|---|
| YAML frontmatter completeness | 10 | Static analysis |
| Trigger phrase quality | 10 | Static analysis |
| Structure & headings | 10 | Static analysis |
| Code example quality | 15 | Static analysis |
| Clarity & conciseness | 15 | Static analysis |
| Real-world task coverage | 10 | **Live test** |
| Output correctness | 15 | **Live test** |
| Agent usability | 15 | **Live test** |

> Static analysis = 60 pts. Live testing = 40 pts. A beautiful skill with poor runtime output scores low.

---

## The Ratchet Mechanism

```
Round 1: baseline = 65
Round 2: proposal scores 75 → KEEP   (baseline = 75)
Round 3: proposal scores 71 → REVERT (baseline stays 75)
Round 4: proposal scores 82 → KEEP   (baseline = 82)
```

Implementation (what the agent does internally):

```bash
# Before each improvement attempt
git add skills/<name>/SKILL.md
git commit -m "darwin: pre-improvement snapshot (<name>)"

# Apply the targeted edit to SKILL.md...

# Re-score with an isolated sub-agent
NEW_SCORE=$(run_scoring_agent skills/<name>/SKILL.md)

if [ "$NEW_SCORE" -gt "$BASELINE_SCORE" ]; then
  git add skills/<name>/SKILL.md
  git commit -m "darwin: improve <name> (+$DELTA pts → $NEW_SCORE)"
  echo "✅ Kept: $BASELINE_SCORE → $NEW_SCORE"
else
  git revert HEAD --no-edit
  echo "⏪ Reverted: $NEW_SCORE < $BASELINE_SCORE"
fi
```

---

## Optimization Phases

### Phase 1 — Inventory & Score

The agent scans all skills and produces a ranked table:

```
Skill                 Score   Weakest Dimension
──────────────────────────────────────────────
nuwa-skill            88/100  Code examples (11/15)
darwin-skill          76/100  Live test coverage (8/15)
my-custom-skill       54/100  Trigger phrases (4/10)
```

### Phase 2 — Targeted Improvement Loop

For each skill (lowest score first):
1. Identify the single lowest-scoring dimension
2. Generate exactly one improvement (scoped to that dimension)
3. Edit the SKILL.md
4. Sub-agent re-scores independently (no self-grading bias)
5. Keep or revert
6. **Pause** — show diff + score change — wait for user `y/n`

### Phase 3 — Report

```markdown
## Darwin Optimization Report

| Skill           | Before | After | Delta |
|-----------------|--------|-------|-------|
| nuwa-skill      | 88     | 92    | +4    |
| darwin-skill    | 76     | 83    | +7    |
| my-custom-skill | 54     | 61    | +7    |

Total improvement: +18 pts across 3 skills
Reverted attempts: 2
```

---

## test-prompts.json Format

Darwin uses a test prompt file to validate live behavior. Place it alongside your skill:

```json
// ~/.claude/skills/<name>/test-prompts.json
{
  "skill": "my-custom-skill",
  "prompts": [
    {
      "id": "basic-usage",
      "input": "show me how to initialize this project",
      "expect_contains": ["npm install", "import", "config"],
      "expect_not_contains": ["TODO", "placeholder"]
    },
    {
      "id": "error-handling",
      "input": "how do I handle auth errors in this library",
      "expect_contains": ["try", "catch", "401"],
      "weight": 1.5
    }
  ]
}
```

---

## Directory Structure

```
~/.claude/skills/
├── darwin-skill/
│   └── SKILL.md              ← this skill
├── nuwa-skill/
│   ├── SKILL.md              ← skill to optimize
│   └── test-prompts.json     ← optional live tests
└── my-other-skill/
    ├── SKILL.md
    └── test-prompts.json
```

---

## What Gets Optimized (Examples)

**Weak trigger phrases → improved:**
```yaml
# Before
triggers:
  - use the tool
  - help me

# After
triggers:
  - initialize a new project with this library
  - configure authentication for my app
  - show me how to handle errors
  - debug connection timeout issues
```

**Missing code examples → added:**
```markdown
# Before
Use the `connect()` method to establish a connection.

# After
Use `connect()` with your credentials:

```typescript
import { Client } from 'my-lib';

const client = new Client({
  url: process.env.SERVICE_URL,
  token: process.env.SERVICE_TOKEN,
});

await client.connect();
```
```

**Vague troubleshooting → made actionable:**
```markdown
# Before
If something goes wrong, check your config.

# After
**"Connection refused" errors**
- Verify `SERVICE_URL` is set: `echo $SERVICE_URL`
- Check firewall allows port 443
- Test with: `curl -I $SERVICE_URL/health`
```

---

## Design Principles

| Principle | What it means |
|---|---|
| Single editable asset | Only one SKILL.md changes per round — improvements are attributable |
| Dual evaluation | Static analysis (structure) + live testing (behavior) |
| Ratchet | Score can only increase; regressions auto-revert |
| Independent scoring | Sub-agent scores, not the same agent that wrote the change |
| Human in the loop | Pauses between skills; you confirm or skip |

---

## Constraints & Limitations

- **One skill at a time** — parallel edits break attribution
- **Git required** — the ratchet depends on `git revert`
- **Human confirmation** — by design, won't auto-batch all skills without pauses
- **Live tests are optional** — skills without `test-prompts.json` score 0 on live dimensions unless the agent can infer test cases from the skill content

---

## Relationship to autoresearch

| autoresearch | darwin-skill |
|---|---|
| `program.md` (defines goal) | This SKILL.md |
| `train.py` (optimized asset) | Each target SKILL.md |
| `val_bpb` loss metric | 8-dimension weighted score (100 pts) |
| `git ratchet` | keep / revert per round |
| Test set | `test-prompts.json` |
| Fully autonomous | **Human-in-loop** (skill quality is subtler than loss) |

---

## Companion: nuwa-skill

darwin-skill optimizes skills. [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) *creates* them from scratch.

```bash
# Create new skills with nuwa, then evolve them with darwin
npx skills add alchaincyf/nuwa-skill
npx skills add alchaincyf/darwin-skill
```

Workflow:
1. `nuwa` → generates initial SKILL.md from a repo URL or description
2. `darwin` → runs the optimization loop, ratchets quality upward

---

## Troubleshooting

**"No skills found"**
- Confirm skills are in `~/.claude/skills/<name>/SKILL.md`
- Or specify the path: `"optimize the skill at ./my-project/SKILL.md"`

**Score not improving after many rounds**
- The skill may be locally optimal — try: `"regenerate this skill from scratch with nuwa, then re-optimize"`
- Check if `test-prompts.json` exists; live test scores (40 pts) are the largest lever

**Git revert fails**
- Ensure the skills directory is a git repo: `cd ~/.claude/skills && git init`
- Darwin needs a clean working tree before each run

**Sub-agent scoring seems inconsistent**
- This is expected for borderline changes (±1–2 pts)
- Darwin only keeps changes with a strict `>` improvement, so ties revert safely
```
