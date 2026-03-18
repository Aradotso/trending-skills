```markdown
---
name: ralph-autonomous-agent-loop
description: Ralph is an autonomous AI agent loop that runs AI coding tools (Amp or Claude Code) repeatedly until all PRD items are complete, with memory persisting via git history, progress.txt, and prd.json.
triggers:
  - "set up ralph for my project"
  - "run ralph agent loop"
  - "automate my PRD with ralph"
  - "create a prd.json for ralph"
  - "use ralph to implement features autonomously"
  - "configure ralph with claude code"
  - "convert my PRD to ralph format"
  - "run autonomous coding agent on my tasks"
---

# Ralph Autonomous Agent Loop

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Ralph is an autonomous AI agent loop that spawns fresh instances of AI coding tools (Amp or Claude Code) repeatedly until all items in a `prd.json` task list are marked complete. Memory between iterations persists via git history, `progress.txt`, and `prd.json`. Based on [Geoffrey Huntley's Ralph pattern](https://ghuntley.com/ralph/).

---

## How Ralph Works

Each iteration of the loop:
1. Reads `prd.json` and picks the highest-priority story where `passes: false`
2. Spawns a **fresh AI instance** (Amp or Claude Code) with a prompt describing that story
3. The AI implements the story, runs quality checks (typecheck, tests), and commits
4. Updates `prd.json` to mark the story `passes: true`
5. Appends learnings to `progress.txt`
6. Repeats until all stories pass or max iterations are reached

When all stories are complete, the AI outputs `<promise>COMPLETE</promise>` and the loop exits.

---

## Prerequisites

- **Amp CLI** (default): Install from [ampcode.com](https://ampcode.com)
- **OR Claude Code**: `npm install -g @anthropic-ai/claude-code`
- `jq` installed: `brew install jq` (macOS) or `apt install jq`
- A git repository for your project

---

## Installation

### Option 1: Copy scripts into your project

```bash
# From your project root
mkdir -p scripts/ralph

# Copy the shell loop
cp /path/to/ralph/ralph.sh scripts/ralph/

# Copy prompt template for your tool:
cp /path/to/ralph/prompt.md scripts/ralph/prompt.md     # For Amp
# OR
cp /path/to/ralph/CLAUDE.md scripts/ralph/CLAUDE.md     # For Claude Code

chmod +x scripts/ralph/ralph.sh
```

### Option 2: Install Amp skills globally

```bash
cp -r skills/prd ~/.config/amp/skills/
cp -r skills/ralph ~/.config/amp/skills/
```

### Option 3: Claude Code Marketplace

```bash
/plugin marketplace add snarktank/ralph
/plugin install ralph-skills@ralph-marketplace
```

Available skills after install:
- `/prd` — Generate a Product Requirements Document
- `/ralph` — Convert a PRD markdown file to `prd.json`

---

## Workflow: Step by Step

### Step 1 — Generate a PRD

Using the installed skill in your AI coding tool:

```
Load the prd skill and create a PRD for [your feature description]
```

The skill asks clarifying questions and saves output to `tasks/prd-[feature-name].md`.

### Step 2 — Convert PRD to `prd.json`

```
Load the ralph skill and convert tasks/prd-my-feature.md to prd.json
```

This creates `prd.json` structured for autonomous execution.

### Step 3 — Run Ralph

```bash
# Default: Amp, 10 iterations
./scripts/ralph/ralph.sh

# With explicit iteration count
./scripts/ralph/ralph.sh 20

# Using Claude Code
./scripts/ralph/ralph.sh --tool claude

# Using Claude Code with custom iteration count
./scripts/ralph/ralph.sh --tool claude 15

# Using Amp explicitly
./scripts/ralph/ralph.sh --tool amp 10
```

---

## `prd.json` Format

This is the core task list Ralph reads and updates.

```json
{
  "branchName": "feature/user-dashboard",
  "projectContext": "A Next.js SaaS app using Prisma, tRPC, and Tailwind.",
  "userStories": [
    {
      "id": "story-1",
      "title": "Add avatar column to users table",
      "priority": 1,
      "passes": false,
      "description": "Add an optional avatarUrl string column to the User model via Prisma migration.",
      "acceptanceCriteria": [
        "Migration file created and applied",
        "User model updated in schema.prisma",
        "TypeScript types regenerated",
        "Existing tests still pass"
      ]
    },
    {
      "id": "story-2",
      "title": "Display avatar in profile header",
      "priority": 2,
      "passes": false,
      "description": "Show the user's avatar image in the top-right profile header component.",
      "acceptanceCriteria": [
        "Avatar renders when avatarUrl is set",
        "Fallback initials shown when avatarUrl is null",
        "Verify in browser using dev-browser skill"
      ]
    }
  ]
}
```

**Key fields:**
| Field | Type | Purpose |
|---|---|---|
| `branchName` | string | Git branch Ralph creates/works on |
| `projectContext` | string | Shared context injected into every iteration prompt |
| `userStories[].id` | string | Unique identifier |
| `userStories[].priority` | number | Lower = higher priority (1 is first) |
| `userStories[].passes` | boolean | `false` = not done, `true` = complete |
| `userStories[].acceptanceCriteria` | string[] | What the AI must verify before marking done |

---

## Key Files Reference

| File | Purpose |
|---|---|
| `ralph.sh` | Bash loop spawning fresh AI instances |
| `prompt.md` | Prompt template injected into each Amp iteration |
| `CLAUDE.md` | Prompt template for Claude Code iterations |
| `prd.json` | Live task list with `passes` status |
| `prd.json.example` | Reference example of the PRD format |
| `progress.txt` | Append-only learnings across iterations |
| `skills/prd/` | Skill: generate PRD markdown |
| `skills/ralph/` | Skill: convert PRD to `prd.json` |
| `.claude-plugin/` | Plugin manifest for Claude Code marketplace |

---

## Inspecting State

```bash
# Which stories are done vs pending?
cat prd.json | jq '.userStories[] | {id, title, passes}'

# What has Ralph learned so far?
cat progress.txt

# Recent commits from Ralph iterations
git log --oneline -10

# Count remaining stories
cat prd.json | jq '[.userStories[] | select(.passes == false)] | length'
```

---

## Customizing the Prompt Templates

After copying `prompt.md` or `CLAUDE.md` to your project, edit it to add:

```markdown
## Project-Specific Quality Checks
- Run `npm run typecheck` — must pass with 0 errors
- Run `npm test -- --passWithNoTests` — must not regress
- Run `npm run lint` — must pass

## Stack Conventions
- This project uses tRPC for API routes — do not create REST endpoints
- Database access only through Prisma — do not use raw SQL
- All new components go in `src/components/` and must be typed

## Common Gotchas
- Always run `npx prisma generate` after schema changes
- Environment variables must be added to both `.env.example` and Zod schema in `src/env.js`
```

---

## Configuring Amp Auto-Handoff (Recommended)

Add to `~/.config/amp/settings.json` to handle stories that exceed one context window:

```json
{
  "amp.experimental.autoHandoff": { "context": 90 }
}
```

When context reaches 90%, Amp automatically hands off to a fresh instance, which Ralph's loop structure handles gracefully.

---

## Writing Good User Stories for Ralph

### ✅ Right-sized stories (one context window)

```json
{ "title": "Add `stripeCustomerId` column to users table via Prisma migration" }
{ "title": "Add a loading skeleton to the /dashboard page" }
{ "title": "Filter the orders list by `status` query param" }
{ "title": "Send welcome email via Resend when user signs up" }
```

### ❌ Too large (split these up)

```json
{ "title": "Build the entire authentication system" }
{ "title": "Refactor the API layer" }
{ "title": "Add the full onboarding flow" }
```

**Rule of thumb:** If a human developer couldn't comfortably implement it in 1–2 hours with full context, split it.

---

## `progress.txt` Pattern

Ralph appends learnings after each iteration. Future iterations read this file for context. Example content:

```
[2026-01-15 story-1] Prisma migration applied successfully. Note: must run
`npx prisma generate` AND restart the dev server for types to update in the
IDE. The User model is in prisma/schema.prisma.

[2026-01-15 story-2] Avatar component is at src/components/ui/Avatar.tsx.
Discovered that the profile header reads from the tRPC `user.me` query —
updating that query's return type was required to expose avatarUrl.
```

---

## AGENTS.md Updates

After each iteration, Ralph updates relevant `AGENTS.md` files with discovered patterns. This propagates knowledge to all future iterations and human developers. Example:

```markdown
# AGENTS.md — src/components/

## Patterns
- All interactive components use `"use client"` directive
- Icon imports come from `lucide-react`, not heroicons
- Tailwind class merging uses `cn()` from `src/lib/utils`

## Gotchas
- Do not forget to export new components from `src/components/index.ts`
- The `Button` component requires explicit `type="button"` to avoid form submission
```

---

## Archiving Previous Runs

When you start a new feature with a different `branchName` in `prd.json`, Ralph automatically archives the previous run:

```
archive/
  2026-01-10-user-dashboard/
    prd.json
    progress.txt
  2026-01-15-billing-flow/
    prd.json
    progress.txt
```

---

## Troubleshooting

### Ralph loops without making progress
- Check `progress.txt` — the AI may be hitting a repeated blocker
- Inspect the last commit: `git show HEAD`
- The story may be too large — split it into smaller stories in `prd.json`
- Verify quality check commands in your prompt template are correct for your project

### `jq` not found
```bash
brew install jq          # macOS
sudo apt install jq      # Ubuntu/Debian
sudo dnf install jq      # Fedora
```

### Stories stuck at `passes: false` after implementation
- The AI coding tool may have failed quality checks — check git log for commits
- Run quality checks manually: `npm run typecheck && npm test`
- If checks fail, fix the issue and manually set `passes: true` for that story to unblock the loop

### Context window exhausted mid-story
- Enable Amp auto-handoff (see above)
- For Claude Code, split the story into smaller pieces in `prd.json`

### Branch already exists
```bash
git branch -D feature/your-branch-name
# Then re-run ralph.sh
```

### Loop exits before all stories complete
- Default max is 10 iterations — increase it: `./scripts/ralph/ralph.sh 25`
- Check if a story is blocking: `cat prd.json | jq '.userStories[] | select(.passes == false)'`

---

## Full Example: End-to-End Usage

```bash
# 1. Clone/navigate to your project
cd my-nextjs-app

# 2. Set up Ralph
mkdir -p scripts/ralph
cp ~/tools/ralph/ralph.sh scripts/ralph/
cp ~/tools/ralph/prompt.md scripts/ralph/
chmod +x scripts/ralph/ralph.sh

# 3. Create prd.json manually or via skill
cat > prd.json << 'EOF'
{
  "branchName": "feature/notifications",
  "projectContext": "Next.js 14 app with Prisma, tRPC, Tailwind, and Resend for email.",
  "userStories": [
    {
      "id": "notif-1",
      "title": "Add notifications table to database",
      "priority": 1,
      "passes": false,
      "description": "Create Notification model in Prisma with fields: id, userId, message, read (bool), createdAt.",
      "acceptanceCriteria": [
        "Migration created and applied",
        "Prisma client regenerated",
        "TypeScript compiles with 0 errors"
      ]
    },
    {
      "id": "notif-2",
      "title": "Add tRPC endpoint to fetch unread notifications",
      "priority": 2,
      "passes": false,
      "description": "Create notifications.getUnread tRPC query returning unread notifications for the current user.",
      "acceptanceCriteria": [
        "Endpoint returns correct data",
        "Protected by auth middleware",
        "TypeScript compiles with 0 errors",
        "Unit test added"
      ]
    }
  ]
}
EOF

# 4. Run Ralph
./scripts/ralph/ralph.sh --tool amp 10

# 5. Monitor progress
watch -n 5 'cat prd.json | jq ".userStories[] | {id, title, passes}"'
```

---

## References

- [Ralph GitHub Repository](https://github.com/snarktank/ralph)
- [Geoffrey Huntley's Ralph Pattern](https://ghuntley.com/ralph/)
- [Ryan Carson's usage article](https://x.com/ryancarson/status/2008548371712135632)
- [Amp documentation](https://ampcode.com/manual)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Interactive Ralph Flowchart](https://snarktank.github.io/ralph/)
```
