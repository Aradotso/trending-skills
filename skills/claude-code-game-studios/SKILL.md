```markdown
---
name: claude-code-game-studios
description: Turn Claude Code into a full 49-agent game dev studio with 72 workflow skills, automated hooks, and a real studio hierarchy for Godot, Unity, and Unreal projects.
triggers:
  - "set up claude code game studios"
  - "use ai agents for game development"
  - "set up game dev studio with claude"
  - "add game studio agents to my project"
  - "how do I use claude code for game dev"
  - "set up godot unity unreal ai workflow"
  - "49 agents game studio claude"
  - "game development ai coordination system"
---

# Claude Code Game Studios

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Claude Code Game Studios transforms a single Claude Code session into a structured game development studio with 49 specialized AI agents, 72 workflow slash commands, 12 automated safety hooks, and 11 path-scoped coding rules — mirroring a real studio hierarchy across design, programming, art, audio, narrative, QA, and production.

---

## Installation

### New Project

```bash
git clone https://github.com/Donchitos/Claude-Code-Game-Studios.git my-game
cd my-game
claude
```

Then type `/start` inside Claude Code.

### Add to an Existing Project

```bash
# Copy only the studio infrastructure (no src/ or assets/ overwrite)
git clone https://github.com/Donchitos/Claude-Code-Game-Studios.git _studio_tmp
cp -r _studio_tmp/.claude ./
cp _studio_tmp/CLAUDE.md ./CLAUDE.md
rm -rf _studio_tmp

claude
# Then run:
/adopt        # Detects existing code and maps it into the studio structure
```

### Prerequisites

```bash
# Required
npm install -g @anthropic-ai/claude-code

# Recommended (hooks fail gracefully without these)
brew install jq          # macOS
sudo apt install jq      # Linux
# Python 3 — usually pre-installed
```

---

## Studio Hierarchy

Agents are tiered by responsibility and model:

```
Tier 1 — Directors (Opus)
  creative-director · technical-director · producer

Tier 2 — Department Leads (Sonnet)
  game-designer · lead-programmer · art-director
  audio-director · narrative-director · qa-lead
  release-manager · localization-lead

Tier 3 — Specialists (Sonnet/Haiku)
  gameplay-programmer · engine-programmer · ai-programmer
  network-programmer · tools-programmer · ui-programmer
  systems-designer · level-designer · economy-designer
  technical-artist · sound-designer · writer
  world-builder · ux-designer · prototyper
  performance-analyst · devops-engineer · analytics-engineer
  security-engineer · qa-tester · accessibility-specialist
  live-ops-designer · community-manager
```

### Engine Specialist Sets

| Engine | Lead Agent | Sub-Specialists |
|---|---|---|
| Godot 4 | `godot-specialist` | GDScript, Shaders, GDExtension |
| Unity | `unity-specialist` | DOTS/ECS, Shaders/VFX, Addressables, UI Toolkit |
| Unreal Engine 5 | `unreal-specialist` | GAS, Blueprints, Replication, UMG/CommonUI |

---

## Key Slash Commands

Type `/` in Claude Code to access all 72 skills. The most important ones:

### Onboarding

```
/start                    # Guided entry — asks where you are, routes you correctly
/help                     # Shows 7-phase pipeline and available skills
/project-stage-detect     # Analyzes an existing project and suggests next steps
/setup-engine godot 4.6   # Configures engine-specific agents and rules
/adopt                    # Integrates existing code/assets into the studio structure
```

### Game Design

```
/brainstorm               # Explore game ideas from scratch with creative-director
/map-systems              # Visualize system dependencies across the game
/design-system <name>     # Deep-dive design for a specific system (combat, economy, etc.)
/quick-design             # Rapid design doc for a small feature
/review-all-gdds          # Audit consistency across all Game Design Documents
/propagate-design-change  # Coordinate a design change across affected departments
```

### Stories & Sprints

```
/create-epics             # Break down the GDD into epics
/create-stories           # Decompose an epic into user stories
/dev-story <id>           # Start development on a story (branches, scaffolding, context)
/story-done <id>          # Close a story (tests, docs, PR checklist)
/sprint-plan              # Build a sprint from the backlog
/sprint-status            # Show current sprint health
/estimate <story-id>      # Get effort estimates with rationale
```

### Code & Architecture

```
/create-architecture      # Generate technical architecture for the project
/architecture-decision    # Record an ADR (Architecture Decision Record)
/architecture-review      # Audit current architecture against the design
/code-review              # Review a file or diff with lead-programmer
/tech-debt                # Identify and prioritize technical debt
/perf-profile             # Analyze performance bottlenecks
```

### QA & Testing

```
/qa-plan                  # Generate a QA plan for a feature or milestone
/smoke-check              # Fast sanity check — critical paths only
/regression-suite         # Full regression test generation
/test-setup               # Scaffold test infrastructure for the project
/test-evidence-review     # Review test results and flag gaps
/skill-test               # Test a skill after modifying it
/skill-improve            # Improve a skill based on test results
```

### Team Orchestration

These coordinate multiple agents simultaneously on a single feature:

```
/team-combat              # Spawns: gameplay-programmer + systems-designer + qa-tester
/team-narrative           # Spawns: writer + world-builder + narrative-director
/team-ui                  # Spawns: ui-programmer + ux-designer + accessibility-specialist
/team-release             # Spawns: release-manager + devops-engineer + qa-lead
/team-polish              # Spawns: performance-analyst + technical-artist + qa-tester
/team-audio               # Spawns: audio-director + sound-designer
/team-level               # Spawns: level-designer + world-builder + technical-artist
/team-live-ops            # Spawns: live-ops-designer + analytics-engineer + community-manager
/team-qa                  # Spawns: qa-lead + qa-tester + performance-analyst
```

### Release

```
/release-checklist        # Full pre-release gate check
/launch-checklist         # Day-of-launch verification
/changelog                # Generate changelog from git history
/patch-notes              # Player-facing patch notes from commits
/hotfix <issue>           # Scoped emergency fix workflow
```

---

## Project Structure

```
CLAUDE.md                         # Master studio configuration
.claude/
  settings.json                   # Hooks, permissions, safety rules
  agents/                         # 49 agent definitions (.md + YAML frontmatter)
  skills/                         # 72 slash commands (one subdirectory per skill)
  hooks/                          # 12 hook scripts (bash)
  rules/                          # 11 path-scoped coding standards
  statusline.sh                   # Shows context%, model, stage, epic breadcrumb
  docs/
    workflow-catalog.yaml         # 7-phase pipeline (read by /help)
    templates/                    # 39 document templates (GDDs, ADRs, specs, etc.)
src/                              # Game source code
assets/                           # Art, audio, VFX, shaders, data
design/                           # GDDs, narrative docs, level designs
docs/                             # Technical docs and ADRs
tests/                            # Unit, integration, performance, playtest suites
tools/                            # Build and pipeline tools
prototypes/                       # Isolated throwaway prototypes
production/                       # Sprint plans, milestones, release tracking
```

---

## How Agents Are Defined

Each agent is a Markdown file in `.claude/agents/` with YAML frontmatter:

```markdown
---
name: gameplay-programmer
tier: 3
model: claude-sonnet
lead: lead-programmer
domain: src/gameplay/
---

# Gameplay Programmer

Owns all gameplay mechanics code. Consults systems-designer for balance,
escalates architectural decisions to lead-programmer.

## Responsibilities
- Implement player controller, abilities, and interactions
- Write unit tests for all gameplay systems
- Profile and optimize hot paths in gameplay code

## Escalation
- Cross-domain changes → lead-programmer
- Balance implications → systems-designer
- Vision conflicts → creative-director (via lead-programmer)
```

### Calling a Specific Agent

Inside Claude Code, reference agents by name:

```
@godot-specialist implement a state machine for the player character
@qa-lead create a test plan for the inventory system
@technical-director review the multiplayer architecture
```

---

## Agent Coordination Model

```
Vertical delegation:    directors → leads → specialists
Horizontal consultation: same-tier agents can consult, not decide
Conflict resolution:    escalate to shared parent director
Change propagation:     cross-department changes go through producer
Domain boundaries:      agents only modify files in their domain/
```

### Collaboration Protocol (All Agents)

Every agent follows this sequence — nothing is written without your approval:

1. **Ask** clarifying questions before proposing
2. **Present** 2–4 options with pros/cons
3. **You decide** — user always makes the call
4. **Draft** — agent shows work before finalizing
5. **Approve** — explicit sign-off before any file changes

---

## Automated Hooks

Hooks fire automatically during your session:

| Hook | Trigger | Purpose |
|---|---|---|
| `validate-commit.sh` | PreToolUse (Bash: `git commit`) | Checks hardcoded values, TODO format, JSON validity, GDD sections |
| `validate-push.sh` | PreToolUse (Bash: `git push`) | Warns on pushes to protected branches |
| `validate-assets.sh` | PostToolUse (Write/Edit in `assets/`) | Validates naming conventions and JSON structure |
| `detect-gaps.sh` | Session open | Detects missing design docs when code or prototypes exist |
| `session-start.sh` | Session open | Shows branch and recent commits for orientation |
| `session-stop.sh` | Session close | Archives `active.md` to session log, records git activity |
| `pre-compact.sh` | Before compaction | Preserves session progress notes |
| `post-compact.sh` | After compaction | Reminds Claude to restore state from `active.md` |
| `log-agent.sh` | Agent spawned | Audit trail — logs subagent invocation |
| `log-agent-stop.sh` | Agent stops | Audit trail — completes subagent record |
| `notify.sh` | Notification event | Windows toast via PowerShell |
| `validate-skill-change.sh` | PostToolUse (Write/Edit in `.claude/skills/`) | Advises running `/skill-test` after skill edits |

> Hooks that don't match their trigger condition exit 0 immediately — no performance impact.

---

## Path-Scoped Coding Rules

Rules in `.claude/rules/` apply automatically when editing files in their scope:

```
gameplay.md     → src/gameplay/**
engine.md       → src/engine/**
ai-npc.md       → src/ai/**
ui.md           → src/ui/**
network.md      → src/network/**
shaders.md      → assets/shaders/**
data.md         → assets/data/**
tests.md        → tests/**
tools.md        → tools/**
prototypes.md   → prototypes/**
docs.md         → design/**, docs/**
```

Example rule entry in `gameplay.md`:

```markdown
# Gameplay Code Standards

- No magic numbers — all constants in `src/gameplay/constants/`
- State machines must use the project's `StateMachine` base class
- Every public method needs a docstring
- Physics interactions must be in `_physics_process`, not `_process`
- New systems require a corresponding test file in `tests/gameplay/`
```

---

## Common Workflows

### Starting a New Game from Scratch

```bash
cd my-game && claude
```
```
/start
# → Guided: "I have no idea yet"
# → Routes to /brainstorm with creative-director

/brainstorm
# → Collaborative ideation session
# → Produces: design/concepts/initial-brainstorm.md

/map-systems
# → Diagrams all game systems and dependencies
# → Produces: design/systems-map.md

/create-architecture
# → technical-director generates tech architecture
# → Produces: docs/architecture/overview.md

/setup-engine godot 4.6
# → Activates Godot-specific agents and rules

/create-epics
# → Breaks GDD into epics
# → Populates: production/epics/

/sprint-plan
# → First sprint from epics
# → Produces: production/sprints/sprint-01.md

/dev-story EPIC-01-S01
# → Scaffolds branch, context, acceptance criteria
# → You build the feature, agent assists

/story-done EPIC-01-S01
# → Tests, docs, PR checklist, closes story
```

### Adopting an Existing Godot Project

```bash
cp -r Claude-Code-Game-Studios/.claude ./
cp Claude-Code-Game-Studios/CLAUDE.md ./
claude
```
```
/adopt
# → Scans src/, assets/, detects engine, maps existing code
# → Produces: design/reverse-engineered-gdd.md (partial)

/project-stage-detect
# → Identifies current stage and recommends next actions

/reverse-document
# → Generates missing design docs from existing code
```

### Running a QA Pass Before a Milestone

```
/milestone-review
# → Checks all stories done, tests passing, docs current

/team-qa
# → Spawns qa-lead + qa-tester + performance-analyst
# → Coordinated: test plan → execution → evidence review

/gate-check
# → Final quality gate: coverage, perf, accessibility, loc

/release-checklist
# → Full pre-release verification list
```

---

## Customizing Agents

### Add a New Specialist

Create `.claude/agents/vfx-artist.md`:

```markdown
---
name: vfx-artist
tier: 3
model: claude-sonnet
lead: art-director
domain: assets/vfx/
---

# VFX Artist

Owns all real-time visual effects. Works with technical-artist on
shader-based VFX, consults gameplay-programmer for effect triggers.

## Responsibilities
- Design and implement particle systems
- Create shader-based VFX (dissolves, impacts, ambient)
- Optimize effect performance on target platforms

## Escalation
- Shader architecture → technical-artist
- Performance budget → performance-analyst
- Art vision → art-director
```

### Add a New Skill

Create `.claude/skills/vfx-design/skill.md`:

```markdown
---
name: vfx-design
description: Design and spec a VFX system for a gameplay moment
---

# /vfx-design

Invokes `vfx-artist` to design a complete VFX specification.

## Steps
1. Ask: what gameplay moment needs VFX?
2. Consult `technical-artist` for shader feasibility
3. Present 3 visual directions with performance estimates
4. Draft: `assets/vfx/specs/<name>-vfx-spec.md`
5. Await approval before any asset creation
```

---

## Troubleshooting

### Hooks Not Running

```bash
# Check hook permissions
chmod +x .claude/hooks/*.sh

# Verify settings.json references hooks correctly
cat .claude/settings.json | jq '.hooks'
```

### Agent Not Responding to Domain

```bash
# Confirm the agent file exists and has correct frontmatter
cat .claude/agents/gameplay-programmer.md | head -10

# Check domain path matches actual project structure
ls src/gameplay/
```

### `/start` Loops or Gives Unexpected Route

```bash
# Manually detect stage
/project-stage-detect

# Or jump directly to the phase you're in:
/setup-engine unreal 5.4    # if engine is known
/create-epics               # if GDD is done
/dev-story <id>             # if stories exist
```

### Missing `jq` — Hook Validation Skipped

```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# Windows (chocolatey)
choco install jq
```

All hooks exit gracefully (exit 0) if `jq` is absent — you lose validation feedback but nothing breaks.

### Skill Changes Not Taking Effect

```bash
# After editing any .claude/skills/ file, run:
/skill-test <skill-name>
/skill-improve <skill-name>   # if test reveals issues
```

---

## Upgrading from an Older Version

```bash
# See UPGRADING.md for full migration guide
cat UPGRADING.md

# Safe to overwrite: .claude/hooks/, .claude/rules/, .claude/docs/templates/
# Manual merge needed: CLAUDE.md, .claude/settings.json, .claude/agents/ (custom agents)
# Never overwrite: src/, assets/, design/, production/ (your game content)
```

---

## References

- [Claude Code Docs](https://docs.anthropic.com/en/docs/claude-code)
- [Repository](https://github.com/Donchitos/Claude-Code-Game-Studios)
- [UPGRADING.md](https://github.com/Donchitos/Claude-Code-Game-Studios/blob/main/UPGRADING.md)
- [Godot 4 Docs](https://docs.godotengine.org/en/stable/)
- [Unity Docs](https://docs.unity3d.com/)
- [Unreal Engine 5 Docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/)
```
