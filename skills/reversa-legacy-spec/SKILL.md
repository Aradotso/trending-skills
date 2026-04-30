```markdown
---
name: reversa-legacy-spec
description: Use Reversa to reverse-engineer legacy codebases into executable AI-agent specifications
triggers:
  - "analyze my legacy codebase"
  - "generate specs from existing code"
  - "reverse engineer this project for AI agents"
  - "install reversa in my project"
  - "extract business rules from old code"
  - "create specifications from legacy system"
  - "help me document this codebase with AI agents"
  - "turn my legacy code into agent-ready specs"
---

# Reversa Legacy Spec Engineering

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Reversa reverse-engineers legacy codebases into executable, traceable specifications ready for AI coding agents. It coordinates a pipeline of specialized agents (Scout, Archaeologist, Detective, Architect, Writer, Reviewer, and more) to extract business rules, architectural decisions, data flows, and implicit knowledge buried in existing code — then outputs structured operational contracts in `_reversa_sdd/`.

---

## Installation

Run inside the root of any legacy project. Node.js 18+ required.

```bash
npx reversa install
```

The installer will:
1. Detect AI engines present (Claude Code, Cursor, Codex, Windsurf, etc.)
2. Prompt which agents to install (all selected by default)
3. Collect project name, language, and preferences
4. Copy agent skills to `.agents/skills/` (and `.claude/skills/` for Claude Code)
5. Create the engine entry file (`CLAUDE.md`, `AGENTS.md`, `.cursorrules`, etc.)
6. Create `.reversa/` state/config structure
7. Generate SHA-256 manifest for safe future updates

> Reversa **never modifies or deletes** any existing file in your project. All writes go to `.reversa/` and `_reversa_sdd/`.

---

## CLI Commands

```bash
# Install Reversa into the current project
npx reversa install

# Show current analysis state and progress
npx reversa status

# Update agents to the latest version (respects your customizations via SHA-256)
npx reversa update

# Add a single agent to an existing installation
npx reversa add-agent

# Add support for an additional AI engine
npx reversa add-engine

# Remove all Reversa-created files (legacy project untouched)
npx reversa uninstall
```

---

## Activating the Analysis

After installation, open your project in the AI agent and run:

```
/reversa
```

For engines without slash command support (Codex, Aider, Opencode):

```
reversa
```

Reversa introduces itself, generates a personalized exploration plan, and saves progress to `.reversa/state.json` at every checkpoint. If the session is interrupted, just type `/reversa` again to resume exactly where you left off.

---

## The 5-Phase Pipeline

```
Phase 1: Reconnaissance  →  Scout
Phase 2: Excavation      →  Archaeologist
Phase 3: Interpretation  →  Detective + Architect
Phase 4: Generation      →  Writer
Phase 5: Review          →  Reviewer
```

Independent agents that can run at any phase: **Visor**, **Data Master**, **Design System**, **Tracer**

### Agent Roles

| Agent | What it does |
|-------|-------------|
| **Scout** | Maps folder structure, languages, frameworks, dependencies, entry points |
| **Archaeologist** | Deep module analysis: algorithms, control flows, data structures |
| **Detective** | Extracts implicit business rules, retroactive ADRs, state machines, permissions |
| **Architect** | Synthesizes C4 diagrams, full ERD, integration map, technical debt register |
| **Writer** | Generates specs as operational contracts with code traceability links |
| **Reviewer** | Finds inconsistencies and validates gaps with the user |
| **Tracer** | Resolves gaps via logs, tracing, real data (read-only) |
| **Visor** | Documents UI from screenshots — no running system needed |
| **Data Master** | Full DB analysis: DDL, migrations, ORM, ERD, triggers, procedures |
| **Design System** | Extracts design tokens: colors, typography, spacing, themes |
| **Chronicler** | Documents code changes during active development sessions |

---

## Output Structure

All generated specifications land in `_reversa_sdd/`:

```
_reversa_sdd/
├── inventory.md              # Full project inventory
├── dependencies.md           # Dependencies with versions
├── code-analysis.md          # Technical analysis per module
├── data-dictionary.md        # Data dictionary
├── domain.md                 # Glossary and business rules
├── state-machines.md         # State machines (Mermaid diagrams)
├── permissions.md            # Permission/role matrix
├── architecture.md           # Architectural overview
├── c4-context.md             # C4 Context diagram
├── c4-containers.md          # C4 Containers diagram
├── c4-components.md          # C4 Components diagram
├── erd-complete.md           # Full ERD in Mermaid
├── confidence-report.md      # 🟢🟡🔴 confidence per statement
├── gaps.md                   # Identified knowledge gaps
├── questions.md              # Questions for human validation
├── dynamic.md                # Dynamic analysis findings (Tracer)
├── sdd/                      # Specs per component
│   └── [component].md
├── openapi/                  # API specs (if applicable)
├── user-stories/             # User stories
├── adrs/                     # Retroactive architectural decisions
├── flowcharts/               # Flowcharts in Mermaid
├── sequences/                # Sequence diagrams
├── ui/                       # Interface specs (Visor)
├── database/                 # Database specs (Data Master)
├── design-system/            # Design tokens (Design System)
└── traceability/
    ├── spec-impact-matrix.md # Which spec impacts which
    └── code-spec-matrix.md   # Code file → corresponding spec
```

### Confidence Scale

Every statement in generated specs is tagged:

| Mark | Meaning |
|------|---------|
| 🟢 CONFIRMED | Extracted directly from code — cited with file + line |
| 🟡 INFERRED | Deduced from patterns — may need validation |
| 🔴 GAP | Not determinable from code — requires human input |

---

## Internal State Structure

```
.reversa/
├── state.json              # Analysis progress between sessions
├── config.toml             # Project configuration
├── config.user.toml        # Personal preferences (add to .gitignore)
├── plan.md                 # Exploration plan (user-editable)
├── version                 # Installed Reversa version
├── context/
│   ├── surface.json        # Scout output
│   └── modules.json        # Archaeologist output
└── _config/
    ├── manifest.yaml           # Installation metadata
    └── files-manifest.json     # SHA-256 hashes for safe updates

.agents/skills/             # Agent skills (all compatible engines)
.claude/skills/             # Mirror for Claude Code specifically
```

---

## Supported AI Engines

| Engine | Entry file created | Activation command |
|--------|-------------------|-------------------|
| Claude Code | `CLAUDE.md` | `/reversa` |
| Codex | `AGENTS.md` | `reversa` |
| Cursor | `.cursorrules` | `/reversa` |
| Gemini CLI | `GEMINI.md` | `/reversa` |
| Windsurf | `.windsurfrules` | `/reversa` |
| Kiro | `.kiro/steering/reversa.md` | `/reversa` |
| Cline | `.clinerules` | `/reversa` |
| Roo Code | `.roorules` | `/reversa` |
| GitHub Copilot | `.github/copilot-instructions.md` | `/reversa` |
| Aider | `CONVENTIONS.md` | `reversa` |
| Amazon Q Developer | `.amazonq/rules/reversa.md` | `/reversa` |

---

## Common Workflows

### Full analysis of a legacy project

```bash
# 1. Navigate to your legacy project root
cd /path/to/legacy-project

# 2. Commit everything first (safety)
git add -A && git commit -m "chore: snapshot before reversa analysis"

# 3. Install Reversa
npx reversa install

# 4. Open project in your AI agent (Claude Code, Cursor, etc.)
# 5. Activate Reversa
/reversa
```

### Resuming an interrupted session

```bash
# Check what phase you're on
npx reversa status

# Then in your AI agent, just re-activate
/reversa
# Reversa reads .reversa/state.json and continues from the last checkpoint
```

### Running only the database agent

```
/reversa
```

Then tell the agent: *"Run only Data Master on the database layer"*

### Updating agents after a new Reversa release

```bash
npx reversa update
# Files you've manually customized are detected via SHA-256 and preserved
```

### Adding a new engine to an existing installation

```bash
npx reversa add-engine
# Follow prompts to select the new engine (e.g. Kiro, Opencode)
```

---

## Configuration

### `.reversa/config.toml` (project-level, commit this)

```toml
[project]
name = "my-legacy-app"
language = "JavaScript"
output_dir = "_reversa_sdd"

[analysis]
include_ui = true
include_database = true
include_design_system = false
```

### `.reversa/config.user.toml` (personal preferences, do NOT commit)

```toml
[preferences]
verbose_checkpoints = true
auto_resume = true
```

### `.reversa/plan.md`

Editable markdown file where you can guide the exploration order, flag modules to skip, or add context the agents should know before starting. Edit this before activating Reversa for best results.

---

## Troubleshooting

### Agent stops mid-analysis

```
/reversa
```

State is persisted in `.reversa/state.json`. The orchestrator will pick up from the last checkpoint automatically.

### Output folder is wrong or missing

Check `.reversa/config.toml` — ensure `output_dir` matches where you expect output. Default is `_reversa_sdd`.

### Reversa modified my existing files

It shouldn't — this is a bug. Restore with:

```bash
git restore .
```

Then open an issue at https://github.com/sandeco/reversa with your OS, Node version, and engine.

### Update overwrote my customized agent skill

```bash
# SHA-256 mismatch means the file was detected as modified and should be preserved.
# If it wasn't, restore your version from git:
git restore .agents/skills/reversa-detective/SKILL.md
```

### Installation fails — Node version

```bash
node --version   # Must be 18+
nvm use 20       # Or install via nvm
npx reversa install
```

### Adding Reversa to a project already using Claude Code slash commands

The installer creates `CLAUDE.md` only if it doesn't exist. If you already have one, the installer appends the Reversa activation block. Check `CLAUDE.md` after install to confirm the `/reversa` entry is present.

---

## Best Practices

- **Always commit before running** — `git add -A && git commit -m "pre-reversa snapshot"`
- **Edit `.reversa/plan.md`** before activation to give agents context about known sensitive modules or areas to prioritize
- **Use `confidence-report.md`** as your review starting point — focus human validation time on 🔴 GAP items first
- **Don't commit `config.user.toml`** — add it to `.gitignore`
- **Run Tracer** when static analysis leaves too many 🔴 GAPs — it resolves ambiguity through dynamic observation
- **Specs in `_reversa_sdd/` are inputs to other agents** — commit them and reference them when asking a coding agent to modify the legacy system

---

## Contributing

```bash
git clone https://github.com/sandeco/reversa.git
cd reversa
npm install
```

Open an issue before submitting a PR to discuss the change.

**License:** MIT
```
