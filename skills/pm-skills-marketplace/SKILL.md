```markdown
---
name: pm-skills-marketplace
description: AI coding agent skill for using the PM Skills Marketplace — 65+ agentic PM skills and 36 chained workflows across 8 plugins for Claude Code, Cowork, and other AI assistants.
triggers:
  - "install pm skills marketplace"
  - "set up product management skills for claude"
  - "use pm skills plugin"
  - "run /discover command for product discovery"
  - "write a PRD using pm skills"
  - "install phuryn/pm-skills"
  - "add pm skills to my project"
  - "use product management workflows in claude code"
---

# PM Skills Marketplace

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

PM Skills Marketplace (`phuryn/pm-skills`) is an open-source collection of 65 PM skills and 36 chained workflows across 8 plugins, designed to give AI coding agents and product managers structured, framework-driven workflows for every PM domain — from discovery and strategy to execution, launch, and growth.

Skills encode proven PM frameworks (Teresa Torres, Marty Cagan, Alberto Savoia) and chain together into commands like `/discover`, `/strategy`, `/write-prd`, and `/plan-launch`.

---

## Installation

### Claude Code (CLI) — Recommended for Developers

```bash
# Add the marketplace registry
claude plugin marketplace add phuryn/pm-skills

# Install all 8 plugins individually
claude plugin install pm-toolkit@pm-skills
claude plugin install pm-product-strategy@pm-skills
claude plugin install pm-product-discovery@pm-skills
claude plugin install pm-market-research@pm-skills
claude plugin install pm-data-analytics@pm-skills
claude plugin install pm-marketing-growth@pm-skills
claude plugin install pm-go-to-market@pm-skills
claude plugin install pm-execution@pm-skills
```

### Claude Cowork (GUI — recommended for non-developers)

1. Open **Customize** (bottom-left corner)
2. Go to **Browse plugins → Personal → +**
3. Select **Add marketplace from GitHub**
4. Enter: `phuryn/pm-skills`

All 8 plugins install automatically.

### Other AI Assistants (skills only — no slash commands)

Skills live in `skills/*/SKILL.md` and follow a universal format readable by any tool.

```bash
# Copy all skills for OpenCode (project-level)
for plugin in pm-*/; do
  mkdir -p .opencode/skills/
  cp -r "$plugin/skills/"* .opencode/skills/ 2>/dev/null
done

# Copy all skills for Gemini CLI (global)
for plugin in pm-*/; do
  cp -r "$plugin/skills/"* ~/.gemini/skills/ 2>/dev/null
done

# Copy all skills for Cursor (project-level)
for plugin in pm-*/; do
  mkdir -p .cursor/skills/
  cp -r "$plugin/skills/"* .cursor/skills/ 2>/dev/null
done

# Copy all skills for Codex CLI (project-level)
for plugin in pm-*/; do
  mkdir -p .codex/skills/
  cp -r "$plugin/skills/"* .codex/skills/ 2>/dev/null
done
```

---

## Plugin Overview

| Plugin | Domain | Skills | Commands |
|--------|--------|--------|----------|
| `pm-product-discovery` | Ideation, assumptions, experiments, OSTs, interviews | 13 | 5 |
| `pm-product-strategy` | Vision, business models, pricing, competitive landscape | 12 | 5 |
| `pm-execution` | PRDs, OKRs, roadmaps, sprints, retros, release notes | 15 | 10 |
| `pm-go-to-market` | Launch planning, positioning, GTM strategy | — | — |
| `pm-market-research` | Market sizing, competitive analysis, customer research | — | — |
| `pm-data-analytics` | Metrics, dashboards, analytics frameworks | — | — |
| `pm-marketing-growth` | Growth loops, acquisition, retention | — | — |
| `pm-toolkit` | Core PM utilities and shared foundations | — | — |

---

## Key Commands and Usage

### Product Discovery

```
# Full discovery cycle: ideation → assumption mapping → prioritization → experiments
/discover AI-powered meeting summarizer for remote teams

# Brainstorm ideas or experiments for existing or new products
/brainstorm ideas existing — We need to reduce churn in our onboarding flow
/brainstorm experiments new — Marketplace for freelance designers

# Triage and prioritize a batch of feature requests
/triage-requests [attach CSV of requests]

# Prepare an interview script or summarize a transcript
/interview prep — We're interviewing enterprise buyers about procurement workflow
/interview summarize — [paste transcript]

# Design a product metrics dashboard
/setup-metrics
```

### Product Strategy

```
# Create a complete 9-section Product Strategy Canvas
/strategy B2B project management tool for agencies

# Explore business models
/business-model startup — AI writing tool for non-native English speakers
/business-model lean — Marketplace for freelance designers
/business-model full — Enterprise SaaS for HR teams
/business-model all — Compare all models side by side

# Design a value proposition using 6-part JTBD template
/value-proposition SaaS onboarding tool for enterprise customers

# Macro environment scan: SWOT + PESTLE + Porter's Five Forces + Ansoff
/market-scan Project management SaaS market

# Design a pricing strategy with competitive analysis
/pricing AI writing assistant — B2C, targeting students and professionals
```

### Execution

```
# Create a PRD from a feature idea or problem statement
/write-prd Dark mode for our mobile app

# Brainstorm team-level OKRs
/plan-okrs Q3 — Growth team, goal: improve activation

# Convert a feature-based roadmap into outcome-focused
/transform-roadmap [paste feature list]

# Sprint lifecycle
/sprint plan — 2-week sprint, 4 engineers, 1 designer
/sprint retro — [paste sprint summary]
/sprint release — [paste merged tickets]

# Pre-mortem risk analysis
/pre-mortem [paste PRD or launch plan]

# Summarize a meeting transcript
/meeting-notes [paste transcript]

# Map stakeholders and create a communication plan
/stakeholder-map Major platform redesign — 6 stakeholders involved

# Write user stories
/write-stories User authentication feature
```

---

## Directly Invoking Skills

Skills load automatically when contextually relevant. To force-load a specific skill:

```
# Using plugin prefix
/pm-product-discovery:opportunity-solution-tree

# Without prefix (Claude infers plugin)
/opportunity-solution-tree

# Direct skill invocation examples
/prioritization-frameworks          # Reference guide to 9 frameworks (ICE, RICE, Kano, MoSCoW, etc.)
/opportunity-solution-tree          # Build an OST: outcome → opportunities → solutions → experiments
/lean-canvas                        # Lean Canvas for a new product idea
/pre-mortem                         # Risk analysis using Tigers/Paper Tigers/Elephants
/stakeholder-map                    # Power × Interest grid with communication plan
```

---

## Skill Chaining: How Commands Work

Commands chain multiple skills into end-to-end workflows. Example — `/discover` chains:

```
/discover
  └── brainstorm-ideas         (multi-perspective ideation)
  └── identify-assumptions     (Value, Usability, Viability, Feasibility risks)
  └── prioritize-assumptions   (Impact × Risk matrix)
  └── brainstorm-experiments   (lean startup pretotypes)
```

After each command completes, it suggests the next relevant command — follow the prompts to move through the full PM workflow naturally.

---

## Practical Workflow Examples

### New Product Idea — End-to-End

```
# 1. Start with discovery
/discover AI-powered expense tracking for freelancers

# 2. Build out strategy
/strategy AI-powered expense tracking for freelancers

# 3. Explore business models
/business-model startup — AI expense tracking, B2C freemium

# 4. Define value proposition
/value-proposition AI expense tracker for freelancers who hate admin

# 5. Scan the market
/market-scan Expense management SaaS for SMBs

# 6. Write the PRD
/write-prd Automated receipt scanning and categorization feature

# 7. Plan the launch
/plan-launch [attach PRD]
```

### Feature Development — Execution Flow

```
# 1. Write the PRD
/write-prd User-facing API key management for developer accounts

# 2. Run a pre-mortem
/pre-mortem [paste PRD]

# 3. Break into user stories
/write-stories API key management feature

# 4. Plan the sprint
/sprint plan — 2-week sprint, 3 engineers, estimated 24 story points

# 5. Write release notes after shipping
/sprint release — [paste merged PRs or ticket list]
```

### Customer Research Flow

```
# 1. Prepare interview script
/interview prep — Interviewing SaaS founders about their hiring workflow

# 2. After interviews, summarize transcripts
/interview summarize — [paste transcript 1]
/interview summarize — [paste transcript 2]

# 3. Triage resulting feature requests
/triage-requests [attach CSV of requests from interviews]

# 4. Map to Opportunity Solution Tree
/opportunity-solution-tree — Outcome: improve time-to-hire for scaling startups
```

---

## Skill File Structure

Each skill follows the universal `SKILL.md` format:

```
pm-product-discovery/
├── skills/
│   ├── brainstorm-ideas-existing/
│   │   └── SKILL.md
│   ├── identify-assumptions-existing/
│   │   └── SKILL.md
│   ├── opportunity-solution-tree/
│   │   └── SKILL.md
│   └── ...
├── commands/
│   ├── discover.md
│   ├── brainstorm.md
│   └── ...
└── plugin.yaml
```

---

## Configuration

No environment variables or secrets required. Skills are pure Markdown files with structured frontmatter.

To override or customize a skill, copy its `SKILL.md` file to your project's local skills directory and modify it:

```bash
# Copy a skill locally to customize it
cp pm-product-discovery/skills/opportunity-solution-tree/SKILL.md \
   .claude/skills/opportunity-solution-tree/SKILL.md
```

Local skills take precedence over marketplace skills in Claude Code.

---

## Troubleshooting

**Skills not loading automatically**
Force-load with explicit invocation: `/pm-product-discovery:brainstorm-ideas-existing` or just `/brainstorm-ideas-existing`.

**Command not found after installation**
Verify the plugin installed correctly:
```bash
claude plugin list
# Should show all 8 pm-* plugins
```

Re-install if missing:
```bash
claude plugin install pm-product-discovery@pm-skills
```

**Using skills in non-Claude tools**
Slash commands (`/discover`, `/strategy`, etc.) are Claude-specific. For Cursor, Gemini CLI, Codex, and others, copy the `SKILL.md` files manually — the skills work, but you invoke them conversationally rather than with `/` commands.

**Skills conflicting with other plugins**
Use the full namespaced form `/plugin-name:skill-name` to be explicit:
```
/pm-execution:create-prd
/pm-product-strategy:lean-canvas
```

**Contributing a new skill**
```bash
git clone https://github.com/phuryn/pm-skills
cd pm-skills
# Add your skill in the appropriate plugin folder
# Follow the SKILL.md format from existing skills
# Submit a PR — see CONTRIBUTING.md
```

---

## Quick Reference Card

| Goal | Command |
|------|---------|
| New idea → full discovery | `/discover [idea]` |
| Strategic clarity | `/strategy [product]` |
| Write a PRD | `/write-prd [feature]` |
| Plan a launch | `/plan-launch` |
| Define metrics | `/north-star` or `/setup-metrics` |
| OKR planning | `/plan-okrs [team + goal]` |
| Sprint planning | `/sprint plan` |
| Customer interviews | `/interview prep` |
| Market analysis | `/market-scan [market]` |
| Pricing strategy | `/pricing [product]` |
| Risk analysis | `/pre-mortem` |
| Stakeholder comms | `/stakeholder-map` |

---

## Resources

- **GitHub**: https://github.com/phuryn/pm-skills
- **Homepage / Full Guide**: https://www.productcompass.pm/p/pm-skills-marketplace-claude
- **License**: MIT
- **Contributing**: See `CONTRIBUTING.md` in the repo
```
