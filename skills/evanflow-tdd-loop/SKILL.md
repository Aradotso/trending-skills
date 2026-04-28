```markdown
---
name: evanflow-tdd-loop
description: Use EvanFlow's 16-skill TDD-driven iterative feedback loop for Claude Code to walk ideas from brainstorm through implementation with checkpoints and guardrails.
triggers:
  - "let's evanflow this"
  - "set up evanflow for my project"
  - "use evanflow to build this feature"
  - "run the evanflow loop"
  - "install evanflow skills"
  - "evanflow brainstorm plan execute"
  - "tdd feedback loop with claude code"
  - "evanflow coder overseer parallel"
---

# EvanFlow TDD Loop

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

EvanFlow is a TDD-driven iterative feedback loop for software development with Claude Code. 16 cohesive skills + 2 custom subagents walk an idea from brainstorm → plan → execute → iterate, with real checkpoints where you stay in control. The agent never auto-commits, never auto-stages, never proposes a PR without your explicit direction.

```
brainstorm → plan → execute (vertical-slice TDD per task) → iterate → STOP
                    └─ sequential, or parallel coder/overseer
```

---

## Installation

### Path 1 — Plugin Marketplace (recommended)

Auto-wires all skills, subagents, and the guardrail hook:

```
/plugin marketplace add evanklem/evanflow
/plugin install evanflow@evanflow
```

Restart Claude Code (or `/reload-plugins`). Skills appear as `/evanflow:evanflow-go`, `/evanflow:evanflow-tdd`, etc.

To uninstall: `/plugin uninstall evanflow@evanflow`

### Path 2 — npx CLI (skills only)

```bash
# All 16 skills at once
npx skills@latest add evanklem/evanflow -s '*' -y

# Individual skills
npx skills@latest add evanklem/evanflow/evanflow-go
npx skills@latest add evanklem/evanflow/evanflow-tdd
npx skills@latest add evanklem/evanflow/evanflow-iterate
```

> ⚠️ This path does NOT install the guardrail hook or custom subagents. Add those manually if needed.

### Path 3 — Manual Copy

```bash
git clone https://github.com/evanklem/evanflow.git
cd evanflow

# Skills (project-level)
mkdir -p .claude/skills
cp -r skills/* .claude/skills/

# Custom subagents
mkdir -p .claude/agents
cp agents/*.md .claude/agents/

# Git guardrail hook
mkdir -p .claude/hooks
cp hooks/block-dangerous-git.sh .claude/hooks/
chmod +x .claude/hooks/block-dangerous-git.sh
```

Register the hook in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/block-dangerous-git.sh"
          }
        ]
      }
    ]
  }
}
```

### Requirements

- **Claude Code** (any recent version)
- **Bash** (Linux, macOS, or Windows + WSL)
- **`jq`** — used by the guardrail hook. Install via:
  ```bash
  # macOS
  brew install jq
  # Ubuntu/Debian
  apt install jq
  ```
  > ⚠️ If `jq` is missing, the hook fails silently and dangerous git ops are NOT blocked.
- **`chromium` or `google-chrome`** (optional) — for UI screenshot verification in `evanflow-iterate`. Falls back gracefully if missing.

### Verify Install

Restart Claude Code, then say:

> "Let's evanflow this — I want to add a small feature that does X."

`evanflow-go` should fire and walk you through the loop.

---

## The 16 Skills at a Glance

### Default Loop (invoke in order, or via `evanflow-go`)

| Skill | What it does |
|---|---|
| `evanflow-brainstorming` | Clarifies intent, proposes 2–3 approaches with embedded stress-test. Mockup quick-mode for visual-only requests. |
| `evanflow-writing-plans` | Maps file structure first, creates bite-sized tasks, offers parallel path if 3+ independent units. |
| `evanflow-executing-plans` | Runs tasks sequentially with inline verification. Surfaces blockers. Hands off to iterate, then STOPS. |
| `evanflow-tdd` | Vertical-slice TDD inside each code task: one failing test → minimal impl → refactor → next test. |
| `evanflow-iterate` | Self-review after implementation. Re-reads diff, runs quality checks, screenshots UI, Five Failure Modes checklist. Hard cap: 5 iterations. |

### Single Entry Point

```
evanflow-go   →   Say "let's evanflow this" — orchestrates the full loop.
```

### Special-Purpose Skills

| Skill | When to use it |
|---|---|
| `evanflow-glossary` | Extract canonical domain terms into `CONTEXT.md` |
| `evanflow-improve-architecture` | Surface refactor opportunities (deletion test + deep-modules) |
| `evanflow-design-interface` | "Design it twice" — 3+ parallel sub-agents with different constraints |
| `evanflow-debug` | Root-cause first, hypothesis stated explicitly, failing test before fix |
| `evanflow-review` | Both sides of code review; don't capitulate to feedback you can't justify |
| `evanflow-prd` | Synthesize a PRD from existing context |
| `evanflow-qa` | Conversational bug discovery → issue draft (asks before filing) |
| `evanflow-compact` | Long-session context management; triggers at clean phase boundaries |

### Custom Subagents (in `agents/`)

| Subagent | Tool access | Role |
|---|---|---|
| `evanflow-coder` | Read, Edit, Write, Glob, Grep, Bash, TodoWrite | Implementation — no git ops, no out-of-scope edits |
| `evanflow-overseer` | Read, Grep, Glob only | Read-only review — physically cannot modify code |

---

## Key Workflows

### 1. Full Loop via Single Entry Point

```
User: "Let's evanflow this — add rate limiting to our API"
```

`evanflow-go` fires and sequences:
1. **Brainstorm** — proposes 2–3 approaches, stress-tests each, awaits your design approval
2. **Plan** — maps files first, creates tasks, offers parallel path → awaits your plan approval
3. **Execute** — runs tasks with TDD inside each code task → surfaces any blockers → hands to iterate
4. **Iterate** — self-review loop (max 5), Five Failure Modes check, quality checks
5. **STOP** — reports outcome, awaits your direction. No commits.

### 2. TDD Discipline Inside Execute

`evanflow-tdd` is the discipline inside every code-writing task — not a separate phase:

```
Per task:
  RED:    Write one failing test that specifies behavior via public interface
  GREEN:  Write minimal implementation to pass it
  REFACTOR: Clean up while the test is still your safety net
  → next test
```

Assertion-correctness check: would a one-character bug in the impl still pass this assertion? If yes, the assertion is wrong.

### 3. Parallel Coder/Overseer (3+ independent units)

When `evanflow-writing-plans` detects 3+ truly independent units, it offers the parallel path:

```
Plan with 3 independent modules
→ fork: 3 evanflow-coder subagents (one per module)
       + 3 evanflow-overseer subagents (one per coder, read-only)
       + 1 integration overseer (runs named integration tests at every touchpoint)
```

Integration tests are the executable contract — interfaces can't drift if both sides must satisfy the same passing test.

### 4. Architecture Review

```
User: "evanflow improve architecture on the auth module"
```

`evanflow-improve-architecture` runs the **deletion test** (if you deleted this, what breaks?) and **deep-modules vocabulary** (wide interface, narrow implementation) to surface refactor opportunities.

### 5. Debug with Root-Cause Discipline

```
User: "evanflow debug — users are getting 500s on /api/checkout"
```

`evanflow-debug` sequence:
1. State explicit hypothesis before touching code
2. Embedded stress-test of the hypothesis
3. Write a failing test that reproduces the bug
4. Fix → verify test passes → run full suite

### 6. Context Management for Long Sessions

```
User: "evanflow compact — we've been going for a while"
```

`evanflow-compact` triggers proactive summarization. Drift symptoms it watches for:
- Re-asking questions that were already settled
- Contradicting earlier decisions
- Losing track of which tasks are done vs. pending

---

## Hard Rules (Every Skill Enforces These)

1. **Never auto-commit, never auto-stage, never auto-finish.** Every git write op requires your explicit ask in the current turn.
2. **Never invent values.** File paths, env vars, IDs, function names, library APIs — if unsure, the agent stops and asks.
3. **No skill tax.** Ad-hoc questions don't require a skill invocation.
4. **No forced spec/plan paths.** Files live where you want them.
5. **Verify before claiming done.** Typecheck, lint, and tests run before any "done" report.

---

## Guardrail Hook — What It Blocks

`hooks/block-dangerous-git.sh` (PreToolUse on Bash) blocks:

```bash
git push            # blocked
git reset --hard    # blocked
git clean -f        # blocked
git branch -D       # blocked
git checkout .      # blocked
git restore .       # blocked
```

Safe git ops (status, diff, log, add, commit with your explicit instruction) pass through.

Test it: try `git reset --hard HEAD` via the Bash tool — it should be blocked with an explanation.

---

## Skill Namespace After Plugin Install

After `/plugin install evanflow@evanflow`, skills are namespaced:

```
/evanflow:evanflow-go
/evanflow:evanflow-brainstorming
/evanflow:evanflow-writing-plans
/evanflow:evanflow-executing-plans
/evanflow:evanflow-tdd
/evanflow:evanflow-iterate
/evanflow:evanflow-debug
/evanflow:evanflow-review
/evanflow:evanflow-compact
/evanflow:evanflow-glossary
/evanflow:evanflow-improve-architecture
/evanflow:evanflow-design-interface
/evanflow:evanflow-prd
/evanflow:evanflow-qa
/evanflow:evanflow
```

Auto-invocation via natural language ("let's evanflow this") still works regardless of namespace.

---

## Adding EvanFlow Context to Your Project

Paste the snippet from `examples/CLAUDE.md.snippet` into your project's `CLAUDE.md` to brief Claude about EvanFlow conventions from session start:

```bash
cat examples/CLAUDE.md.snippet >> CLAUDE.md
```

This ensures EvanFlow's vocabulary (deletion test, vertical-slice TDD, Five Failure Modes, etc.) is in context without requiring you to explain it each session.

---

## Common Patterns

### Starting a Feature from Scratch

```
"Let's evanflow this — I want to add webhook support to our API"
```

### Jumping to a Specific Phase

```
"evanflow-plan: we've already brainstormed, here's the design we settled on: [...]"
"evanflow-tdd: implement the UserRepository.findByEmail method"
"evanflow-debug: the payment webhook is firing twice"
```

### Using Design Interface for Hard API Decisions

```
"evanflow design interface for the plugin system — I want 3+ radically different approaches compared"
```

### Running QA Conversationally

```
"evanflow qa — let's find bugs in the checkout flow before the release"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Hook not blocking dangerous git ops | Check `jq` is installed (`jq --version`). Check hook is executable (`chmod +x .claude/hooks/block-dangerous-git.sh`). Check `settings.json` registration (Path 3). |
| Skills not appearing after install | Restart Claude Code or run `/reload-plugins`. For manual install, verify files landed in `.claude/skills/`. |
| `evanflow-go` not triggering on "let's evanflow this" | Check skill is installed. Try explicit invocation: `/evanflow:evanflow-go` or `/evanflow-go`. |
| Chromium screenshot step skipped | Install `chromium` or `google-chrome`. `evanflow-iterate` falls back gracefully — it will flag UI changes need manual verification. |
| Parallel path not offered | Plan needs 3+ truly independent units. If tasks share state or ordering constraints, the sequential path is correct. |
| Agent inventing file paths / env var names | This violates Hard Rule 2. Tell the agent to stop and ask. If it persists, report as a bug at `evanklem/evanflow`. |

---

## Research Grounding

EvanFlow's hard rules cite their sources:

- **Never invent values** — action-hallucination is the #1 failure mode per [DAPLab/Columbia "9 Critical Failure Patterns of Coding Agents"](https://daplab.cs.columbia.edu/general/2026/01/08/9-critical-failure-patterns-of-coding-agents.html)
- **Assertion-correctness warning** — 62%+ of LLM-generated test assertions were incorrect per ["Test-Driven Development for Code Generation" (arXiv 2402.13521)](https://arxiv.org/pdf/2402.13521) §3.2
- **Context drift watch** — ~65% of enterprise AI failures in 2025 attributed to context drift, not raw context exhaustion per [Alex Merced, "Context Management Strategies for OpenCode" (March 2026)](https://datalakehousehub.com/blog/2026-03-context-management-opencode/)

Rules without citations (no auto-commit, no skill tax) are labeled as opinion from running the loop on real projects.
```
