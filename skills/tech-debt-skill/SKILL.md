```markdown
---
name: tech-debt-skill
description: Claude Code skill that produces a thorough, file-cited tech debt audit of an entire codebase across nine dimensions with severity ratings, effort estimates, and a ranked action list
triggers:
  - run a tech debt audit on this codebase
  - audit the technical debt in this project
  - find tech debt and code quality issues
  - generate a TECH_DEBT_AUDIT.md file
  - analyze codebase for architectural decay and code rot
  - what are the biggest tech debt problems in this repo
  - do a thorough code quality review with file citations
  - help me understand what needs refactoring in this project
---

# tech-debt-audit

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Claude Code skill that runs a structured, three-phase tech debt audit across an entire codebase and writes `TECH_DEBT_AUDIT.md` — a persistent, file-cited, severity-ranked artifact you can commit and track over time.

## What it does

The skill sweeps nine debt dimensions:

1. **Architectural decay** — god classes, spaghetti dependencies, layer violations
2. **Consistency rot** — mixed patterns, style drift, naming inconsistencies
3. **Type & contract debt** — `any`, untyped interfaces, implicit nulls, broken contracts
4. **Test debt** — coverage gaps, brittle tests, untested critical paths
5. **Dependency & config debt** — outdated deps, unused deps, config duplication
6. **Performance & resource hygiene** — N+1 queries, unbounded growth, leaked resources
7. **Error handling & observability** — swallowed errors, missing logging, silent failures
8. **Security hygiene** — hardcoded secrets, injection patterns, weak crypto
9. **Documentation drift** — stale comments, missing docstrings, lying READMEs

Every finding is cited as `path/to/file.ext:LINE`. Every finding has severity (`Critical / High / Medium / Low`) and effort (`XS / S / M / L / XL`). The report includes a "things that look bad but are actually fine" section — the single biggest quality signal in any audit.

## Installation

**Personal install** (available in all your projects):

```bash
mkdir -p ~/.claude/skills/tech-debt-audit
curl -o ~/.claude/skills/tech-debt-audit/SKILL.md \
  https://raw.githubusercontent.com/ksimback/tech-debt-skill/main/SKILL.md
```

**Project-only install** (just this repo):

```bash
mkdir -p .claude/skills/tech-debt-audit
curl -o .claude/skills/tech-debt-audit/SKILL.md \
  https://raw.githubusercontent.com/ksimback/tech-debt-skill/main/SKILL.md
```

**Verify installation:**

```bash
claude --print "/skills" | grep tech-debt-audit
```

## Usage

### Basic audit

In Claude Code, in the repo root:

```
/tech-debt-audit
```

Output: `TECH_DEBT_AUDIT.md` in the repo root. First run: 5–20 minutes depending on repo size.

### Audit a subtree (monorepos)

```
/tech-debt-audit src/payments
```

Scopes the entire three-phase protocol to `src/payments/` only. Useful for large monorepos or targeted investigations.

### Mid-audit course correction

Interrupt after Phase 1 completes with:

```
Before Phase 2, tell me what surprised you in Phase 1 and what you
want to investigate that isn't in the dimensions list.
```

The best findings often come from things the prompt didn't anticipate.

### Repeat runs (living document mode)

On subsequent runs the skill reads the existing `TECH_DEBT_AUDIT.md` as a baseline:

- Resolved findings → marked `RESOLVED`
- Stale findings → updated in place
- New findings → tagged `NEW`

Just run `/tech-debt-audit` again. No flags needed.

## How it works (three phases)

### Phase 1 — Orient

The skill reads the manifest (`package.json`, `Cargo.toml`, `go.mod`, `pyproject.toml`, etc.), maps the directory structure, analyzes `git log` for churn, identifies the largest and most-modified files (their intersection is where debt usually hides), then writes an internal mental model of the architecture before forming any opinions.

**Phase 1 is not optional.** Findings without context are vibes.

### Phase 2 — Audit

Sweeps all nine dimensions using `rg`, `ast-grep`, and language-native tooling:

| Stack | Tools run |
|-------|-----------|
| TypeScript / JavaScript | `npm audit`, `knip`, `madge`, `depcheck` |
| Python | `pip-audit`, `ruff`, `vulture`, `pydeps` |
| Rust | `cargo audit`, `cargo udeps`, `cargo machete` |
| Go | `govulncheck`, `staticcheck`, `golangci-lint` |

Tool output is folded into the findings table with citations. For repos over ~50k LOC, subagents are dispatched per module to avoid context exhaustion.

### Phase 3 — Deliverable

Writes `TECH_DEBT_AUDIT.md` with:

- Executive summary (finding counts by severity, debt concentration by path)
- Mental model of the architecture
- Full findings table with `file:line`, severity, effort, description, recommendation
- Top 5 priorities with rationale
- Quick wins checklist
- "Things that look bad but are actually fine" section
- Open questions for the maintainer

## Output format

```markdown
## Executive summary
- 3 Critical findings, 12 High, 31 Medium, 18 Low
- Largest debt concentration: src/payments/* (3 of 3 Critical findings)
- Estimated remediation effort for Critical+High: ~6 engineer-weeks

## Mental model
src/payments/ is the core revenue path. processor.ts has grown
to own routing, validation, retry, and reconciliation — functions
that arrived incrementally and were never extracted...

## Findings

| ID   | Category            | File:Line                      | Severity | Effort | Description                                              | Recommendation                                          |
|------|---------------------|-------------------------------|----------|--------|----------------------------------------------------------|---------------------------------------------------------|
| F001 | Architectural decay | src/payments/processor.ts:1240 | Critical | L      | 1,400-line god class: routing, validation, retry, recon | Extract retry + reconciliation into separate services   |
| F002 | Type & contract     | src/api/handlers.ts:88         | High     | S      | `req.body` typed as `any`, propagates to 6 call sites   | Add zod schema at boundary; infer type downstream       |
| ...  |                     |                               |          |        |                                                          |                                                         |

## Top 5

1. **F001** — Decompose `payments/processor.ts`: the retry and reconciliation
   logic has different failure modes and deployment cadence than routing.
   Extracting them reduces blast radius for the most-changed file in the repo.

2. ...

## Quick wins

- [ ] F042: Remove unused dep `lodash.merge` (replaced by native `structuredClone` at src/utils/clone.ts:14)
- [ ] F039: Delete commented-out feature flag block at src/flags/beta.ts:201–247 (flag shipped in v2.3)
- [ ] ...

## Things that look bad but are actually fine

- The deeply nested callback chain in `src/legacy/webhooks.ts:88–140` looks
  like a refactor target, but it preserves strict ordering guarantees the
  queue-based replacement would silently break. Leave it.
- `src/db/migrations/` has zero test coverage. This is intentional — the
  migration runner validates checksums before applying; tests would duplicate
  that guarantee without adding signal.

## Open questions for the maintainer

- Is `src/experiments/` intentionally untested, or did it fall through coverage?
- `config/staging.yml` and `config/production.yml` are 94% identical. Is there
  a plan to merge them, or is the divergence load-bearing?
```

## Customization

### Add domain-specific dimensions

Fork `SKILL.md` and add dimensions to the Phase 2 sweep. Examples:

- **Frontend repos** → accessibility debt (missing ARIA, contrast violations)
- **ML repos** → eval drift (model versions pinned to stale benchmarks)
- **LLM apps** → prompt versioning debt, tool-call cost regressions
- **Infra repos** → IaC drift (resources that exist in cloud but not in Terraform)

### Tune severity thresholds

The default god-file threshold is 500 LOC. If your codebase has a higher baseline, edit the dimension definition in `SKILL.md` directly:

```markdown
<!-- in SKILL.md, Phase 2, Dimension 1 -->
God file threshold: >800 LOC (adjusted for this codebase's baseline)
```

### Per-project override

A `.claude/skills/tech-debt-audit/SKILL.md` in a specific repo overrides the global `~/.claude/skills/tech-debt-audit/SKILL.md`. Useful when one project needs custom dimensions others don't.

### Split into supporting files

As `SKILL.md` grows, extract sections into sibling files and reference them. Claude Code lazy-loads supporting files:

```
.claude/skills/tech-debt-audit/
  SKILL.md               # main protocol
  severity-rubric.md     # extracted severity definitions
  stack-tooling.md       # per-language tool configs
```

## Committing and tracking the audit

```bash
# Commit the initial audit
git add TECH_DEBT_AUDIT.md
git commit -m "chore: initial tech debt audit"

# After fixing F001 and F042, run again and commit the updated audit
/tech-debt-audit
git add TECH_DEBT_AUDIT.md
git commit -m "chore: mark F001 resolved, F042 resolved in audit"
```

The file format is stable across runs, so diffs are readable in PRs.

## Limitations

- **Not a security audit.** Catches obvious hygiene issues (hardcoded secrets, SQL injection patterns, weak crypto) but won't replace a pen test or threat model.
- **Not a business-logic bug finder.** Requires domain knowledge the model doesn't have.
- **Very large repos (>200k LOC).** Even subagent dispatch can produce shallow results. Scope to a module or run section-by-section.
- **Intentional simplicity vs. accidental simplicity.** The model can't always tell the difference. The "open questions" section is where it asks rather than asserts.

## Troubleshooting

**The skill isn't listed in `/skills`**

```bash
# Confirm the file exists at the right path
ls ~/.claude/skills/tech-debt-audit/SKILL.md

# Reinstall
curl -o ~/.claude/skills/tech-debt-audit/SKILL.md \
  https://raw.githubusercontent.com/ksimback/tech-debt-skill/main/SKILL.md
```

**Phase 2 tools are missing (e.g., `knip: command not found`)**

The skill runs available tools and skips unavailable ones with a note in the report. To get full coverage, install the tools for your stack:

```bash
# TypeScript / JavaScript
npm install -g knip

# Python
pip install pip-audit ruff vulture

# Rust
cargo install cargo-audit cargo-udeps cargo-machete

# Go
go install golang.org/x/vuln/cmd/govulncheck@latest
go install honnef.co/go/tools/cmd/staticcheck@latest
```

**Audit ran out of context on a large repo**

Scope to a subtree:

```
/tech-debt-audit src/core
```

Or interrupt after Phase 1 and ask:

```
Given the mental model you just built, which three modules should
I audit first? Let's run Phase 2 on just those.
```

**The "looks bad but is actually fine" section is empty**

This means the audit didn't look hard enough. Interrupt and prompt:

```
The false-positive section is empty. Walk me through three things
you considered flagging but chose not to, and why.
```

If the model can't produce any, the Phase 2 sweep was too shallow.

**TECH_DEBT_AUDIT.md wasn't written**

The skill writes the file at the end of Phase 3. If Phase 3 was cut short, ask:

```
Please write TECH_DEBT_AUDIT.md now using the findings from Phase 2.
```
```
