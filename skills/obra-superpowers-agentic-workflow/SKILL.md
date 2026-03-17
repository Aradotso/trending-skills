```markdown
---
name: obra-superpowers-agentic-workflow
description: Agentic skills framework for coding agents providing structured workflows for TDD, planning, debugging, and subagent-driven development
triggers:
  - "set up superpowers for my coding agent"
  - "install superpowers skills framework"
  - "use superpowers workflow"
  - "help me plan this feature with superpowers"
  - "set up agentic development workflow"
  - "configure coding agent skills"
  - "add superpowers to claude code"
  - "use subagent driven development"
---

# obra/superpowers Agentic Skills Framework

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Superpowers is a composable skills framework that gives coding agents (Claude Code, Cursor, Codex, OpenCode, Gemini CLI) structured workflows for software development. It enforces TDD, systematic debugging, design-first planning, and subagent-driven development — automatically, without manual prompting.

## What It Does

- **Auto-triggers skills** based on what you're doing (planning, debugging, implementing)
- **Enforces RED-GREEN-REFACTOR TDD** — writes failing tests first, always
- **Runs subagent-driven development** — spawns fresh subagents per task with two-stage review
- **Structures design before code** — Socratic brainstorming → spec → implementation plan → execution
- **Manages git worktrees** for parallel isolated development branches

## Installation

### Claude Code (Official Marketplace)

```bash
/plugin install superpowers@claude-plugins-official
```

### Claude Code (via Custom Marketplace)

```bash
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```

### Cursor

In Cursor Agent chat:

```text
/add-plugin superpowers
```

Or search "superpowers" in the plugin marketplace UI.

### Codex

Tell Codex in a session:

```
Fetch and follow instructions from https://raw.githubusercontent.com/obra/superpowers/refs/heads/main/.codex/INSTALL.md
```

### OpenCode

Tell OpenCode in a session:

```
Fetch and follow instructions from https://raw.githubusercontent.com/obra/superpowers/refs/heads/main/.opencode/INSTALL.md
```

### Gemini CLI

```bash
gemini extensions install https://github.com/obra/superpowers
```

Update later with:

```bash
gemini extensions update superpowers
```

### Update Any Platform

```bash
/plugin update superpowers
```

### Verify Installation

Start a new session and say:

```
help me plan this feature
```

The agent should automatically invoke the `brainstorming` skill without being asked.

---

## Core Workflow (in order)

### 1. Brainstorming — Design Before Code

**Triggers automatically** when you describe something to build.

The agent will NOT write code immediately. Instead it:
1. Asks clarifying questions to extract a real spec
2. Presents the design in readable chunks for your approval
3. Only proceeds after you sign off

**Example prompt:**

```
I want to add rate limiting to my API
```

The agent will ask: What kind of rate limiting? Per user or per IP? What limits? What happens on exceed? etc.

---

### 2. using-git-worktrees — Isolated Workspace

**Triggers after design approval.**

Creates a new git branch + worktree so implementation is isolated:

```bash
# What the agent does internally:
git worktree add ../my-project-feature-branch -b feature/rate-limiting
cd ../my-project-feature-branch
# runs project setup (npm install, bundle install, etc.)
# verifies clean test baseline
```

---

### 3. writing-plans — Bite-Sized Task Plans

**Triggers with approved design.**

Produces a plan where every task includes:
- Exact file paths
- Complete code to write
- Verification steps (tests to run)
- 2–5 minute estimated scope

**Example plan output structure:**

```markdown
## Task 1: Add RateLimiter class
- File: `src/middleware/rate_limiter.rb`
- Write failing test first: `spec/middleware/rate_limiter_spec.rb`
- Test: `bundle exec rspec spec/middleware/rate_limiter_spec.rb`
- Expected: RED (test fails, class doesn't exist)

## Task 2: Implement token bucket algorithm
- File: `src/middleware/rate_limiter.rb`
- Write minimal code to pass test
- Test: `bundle exec rspec spec/middleware/rate_limiter_spec.rb`
- Expected: GREEN
```

---

### 4. subagent-driven-development — Automated Execution

**Triggers when you say "go" on the plan.**

Dispatches a fresh subagent per task. Each task goes through two-stage review:
1. **Spec compliance** — did it follow the plan?
2. **Code quality** — is it clean, minimal, tested?

Critical review issues block forward progress.

Alternative: `executing-plans` runs tasks in batches with human checkpoints instead of subagents.

---

### 5. test-driven-development — RED-GREEN-REFACTOR

**Triggers during any implementation task.**

The strict cycle:

```
1. Write a failing test
2. Run it — confirm RED
3. Write MINIMAL code to pass
4. Run it — confirm GREEN
5. Refactor if needed
6. Commit
7. Repeat
```

**Anti-pattern the agent will refuse:**
- Writing code before tests
- Writing more code than needed to pass the test
- Skipping the "watch it fail" step

If code was written before tests, the agent **deletes it** and restarts with tests.

---

### 6. requesting-code-review

**Triggers between tasks.**

The agent reviews work against:
- The original plan spec
- Code quality standards
- Test coverage

Severity levels: `critical` (blocks) / `major` (should fix) / `minor` (optional)

---

### 7. finishing-a-development-branch

**Triggers when all tasks complete.**

Presents options:
- Merge to main
- Open a PR
- Keep branch for more work
- Discard branch

Then cleans up the worktree:

```bash
git worktree remove ../my-project-feature-branch
```

---

## Skills Reference

### Testing
| Skill | What it does |
|---|---|
| `test-driven-development` | Enforces RED-GREEN-REFACTOR with anti-pattern detection |

### Debugging
| Skill | What it does |
|---|---|
| `systematic-debugging` | 4-phase root cause process |
| `verification-before-completion` | Confirms issue is actually fixed |

### Collaboration & Planning
| Skill | What it does |
|---|---|
| `brainstorming` | Socratic design refinement before any code |
| `writing-plans` | Detailed task-by-task implementation plans |
| `executing-plans` | Batch execution with human checkpoints |
| `dispatching-parallel-agents` | Concurrent subagent workflows |
| `requesting-code-review` | Pre-merge review checklist |
| `receiving-code-review` | Structured response to review feedback |
| `using-git-worktrees` | Parallel isolated development branches |
| `finishing-a-development-branch` | Merge/PR/discard decision + cleanup |
| `subagent-driven-development` | Per-task subagents with two-stage review |

### Meta
| Skill | What it does |
|---|---|
| `writing-skills` | Create new skills following best practices |
| `using-superpowers` | Introduction to the skills system |

---

## Writing a Custom Skill

Skills live in `skills/<skill-name>/SKILL.md`. To add one:

```bash
# Fork the repo, create a branch
git checkout -b skill/my-custom-skill

# Create skill directory
mkdir -p skills/my-custom-skill
touch skills/my-custom-skill/SKILL.md
```

Follow the `writing-skills` skill format. Frontmatter structure:

```yaml
---
name: my-custom-skill
description: One line description
triggers:
  - "phrase that activates this skill"
  - "another trigger phrase"
---
```

Then submit a PR to `obra/superpowers`.

---

## Common Patterns

### Starting a new feature from scratch

```
You: I want to build a CSV export feature for my Rails app

Agent: [brainstorming skill activates]
Agent: Let me ask a few questions...
  1. What data should be exportable?
  2. Should exports be synchronous or background jobs?
  3. What format — standard CSV or with custom headers?
```

### Resuming an existing plan

```
You: Let's continue the implementation plan

Agent: [executing-plans or subagent-driven-development activates]
Agent: Picking up at Task 3: Implement background job...
```

### Debugging a failing test

```
You: This test keeps failing and I don't know why

Agent: [systematic-debugging skill activates]
Agent: Let's work through this systematically.
Phase 1: Reproduce reliably...
```

### Parallel work on independent features

```
You: Can we work on the auth refactor and the new dashboard in parallel?

Agent: [dispatching-parallel-agents skill activates]
Agent: I'll set up two worktrees and dispatch subagents for each...
```

---

## Philosophy & Guardrails

**YAGNI** — The agent will push back on scope creep. If it's not in the spec, it won't build it.

**DRY** — Duplication spotted in review will be flagged as `major` severity.

**Complexity reduction** — Simpler solution always preferred over clever solution.

**Evidence over claims** — The agent runs tests and shows output rather than claiming something works.

**Tests first, always** — No exceptions. Code written before tests gets deleted.

---

## Troubleshooting

### Skills not triggering automatically

- Start a **new session** after installation (skills load at session start)
- Verify install: ask "what superpowers skills do you have?"
- Re-run the install command for your platform

### Agent skipping the brainstorming step

Say explicitly:

```
Before writing any code, use the brainstorming skill
```

### Subagent tasks getting stuck in review loop

Critical review issues block progress by design. Tell the agent:

```
Show me the critical issues blocking Task N
```

Then decide: fix them or explicitly override with:

```
Accept the current implementation and continue to Task N+1
```

### Worktree conflicts

```bash
# List active worktrees
git worktree list

# Remove a stuck worktree manually
git worktree remove --force ../project-branch-name
git branch -d feature/branch-name
```

### Plugin update not applying

Restart your agent session after updating — skills are loaded at session initialization.

---

## Resources

- **Repository**: https://github.com/obra/superpowers
- **Blog post**: https://blog.fsck.com/2025/10/09/superpowers-for-claude-code/
- **Discord**: https://discord.gg/Jd8Vphy9jq
- **Issues**: https://github.com/obra/superpowers/issues
- **Marketplace**: https://github.com/obra/superpowers-marketplace
- **Sponsor**: https://github.com/sponsors/obra
```
