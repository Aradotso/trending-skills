```markdown
---
name: clawsweeper-issue-triage
description: ClawSweeper is a conservative GitHub maintainer bot that scans all open issues and PRs, writes per-item markdown review records, and closes only when evidence meets strict criteria.
triggers:
  - set up clawsweeper for my repo
  - automate issue triage with clawsweeper
  - how do I run clawsweeper locally
  - configure clawsweeper to close stale issues
  - apply clawsweeper closures to my repository
  - review open PRs with clawsweeper
  - clawsweeper dashboard not updating
  - how does clawsweeper decide what to close
---

# ClawSweeper Issue Triage Bot

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

ClawSweeper is a conservative AI-powered GitHub maintainer bot that reviews every open issue and PR in a target repository, writes a regenerated markdown record per item, and closes only when evidence is strong. It runs on a weekly cadence per item and uses Codex (`gpt-5.5`) to evaluate each item against a strict set of allowed close reasons.

## What ClawSweeper Does

- Scans all open issues and PRs in a target repo (e.g. `openclaw/openclaw`)
- Writes one markdown file per item under `items/<number>.md`
- Proposes closes only for items that match allowed reasons
- Actually closes items only when `apply_closures=true` is explicitly set
- Archives closed item records to `closed/<number>.md`
- Updates a live dashboard in the README after each run
- Never auto-closes items authored by `OWNER`, `MEMBER`, or `COLLABORATOR`

### Allowed Close Reasons

1. Already implemented on `main`
2. Cannot reproduce on current `main`
3. Belongs on a plugin/skill hub, not in core
4. Too incoherent to be actionable
5. Stale issue older than 60 days with insufficient data to verify the bug

## Installation

Requires **Node 24**.

```bash
git clone https://github.com/openclaw/clawsweeper.git
cd clawsweeper
npm install
npm run build
```

Set required environment variables:

```bash
export GITHUB_TOKEN=$GITHUB_TOKEN          # read access to target repo
export OPENAI_API_KEY=$OPENAI_API_KEY      # Codex access
export CLAWSWEEPER_REPO="owner/repo"       # target repo to scan
```

## Key CLI Commands

### Plan a sweep

Scans open items and produces shard batches for review jobs.

```bash
npm run plan -- \
  --batch-size 5 \
  --shard-count 40 \
  --max-pages 250 \
  --codex-model gpt-5.5 \
  --codex-reasoning-effort medium \
  --codex-service-tier fast
```

### Review items

Runs Codex on each planned item inside a local checkout of the target repo.

```bash
npm run review -- \
  --openclaw-dir ../openclaw \
  --batch-size 5 \
  --max-pages 250 \
  --artifact-dir artifacts/reviews \
  --codex-model gpt-5.5 \
  --codex-reasoning-effort medium \
  --codex-service-tier fast \
  --codex-timeout-ms 600000
```

### Apply review artifacts to the repo

Merges shard artifacts and updates dashboard. Does **not** close anything yet.

```bash
npm run apply-artifacts -- --artifact-dir artifacts/reviews
```

### Apply existing proposals (close issues/PRs)

Re-fetches issue context, recomputes snapshot hash, and closes if nothing changed since the proposal.

```bash
npm run apply-decisions -- --limit 20
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--limit` | unlimited | Max fresh closes in this run |
| `--apply-kind` | `issue` | `issue`, `pr`, or `all` |
| `--apply-min-age-days` | `0` | Minimum age of proposal before applying |
| `--close-delay-ms` | `5000` | Delay between close API calls |

### Reconcile items folder

Moves externally closed items from `items/` to `closed/` and reopened items back to `items/` as stale.

```bash
npm run reconcile -- --dry-run   # preview only
npm run reconcile               # apply changes
```

## GitHub Actions Workflow

ClawSweeper is designed to run as a scheduled GitHub Actions workflow on the clawsweeper repo itself. Key workflow inputs:

```yaml
# Trigger a manual run with apply enabled
workflow_dispatch:
  inputs:
    apply_closures:
      description: "Set to true to actually close issues/PRs"
      default: "false"
    apply_existing:
      description: "Apply existing proposals without re-running Codex"
      default: "false"
    apply_kind:
      description: "issue | pr | all"
      default: "issue"
    apply_min_age_days:
      description: "Min age in days for a proposal to be applied"
      default: "0"
```

The normal scheduled run is **proposal-only** — it never comments or closes without explicit `apply_closures=true`.

## Item Record Format

Each item is stored as `items/<number>.md`. Example structure:

```markdown
# Issue #58150

**Title:** [Bug]: RISC-V64: OpenClaw fails with LLM request failed

**Outcome:** keep_open

**Reason:** Bug is reproducible and not fixed on main. Insufficient evidence
to close as cannot-reproduce.

**Snapshot hash:** abc123def456

**Reviewed:** Apr 25, 2026, 16:55 UTC

**Proposed close comment:** N/A
```

When outcome is `proposed_close`, the record includes the exact comment that will be posted before closing.

## Review Cadence

| Item type | Cadence |
|---|---|
| All PRs | Daily |
| Issues < 30 days old | Daily |
| Issues ≥ 30 days old with no recent activity | Weekly |
| Items with new activity since last snapshot | Daily (promoted) |

The planner prioritizes: **active items → PRs → new issues → older weekly issues** when more items are due than fit in a run.

## Configuration Reference

ClawSweeper reads configuration from environment variables and CLI flags. There is no separate config file — all settings are passed at invocation time.

| Env Var | Purpose |
|---|---|
| `GITHUB_TOKEN` | GitHub API access (read for review, write for apply) |
| `OPENAI_API_KEY` | Codex API key |
| `CLAWSWEEPER_REPO` | Target repo in `owner/repo` format |

| CLI Flag | Command | Description |
|---|---|---|
| `--batch-size` | plan, review | Items per shard batch |
| `--shard-count` | plan | Number of parallel review shards |
| `--max-pages` | plan, review | Max GitHub API pages to scan |
| `--codex-model` | plan, review | Model to use (default: `gpt-5.5`) |
| `--codex-reasoning-effort` | plan, review | `low`, `medium`, `high` |
| `--codex-service-tier` | plan, review | `default`, `fast` |
| `--codex-timeout-ms` | review | Per-item timeout (default: 600000) |
| `--artifact-dir` | review, apply-artifacts | Directory for shard output |
| `--dry-run` | reconcile | Preview without writing |
| `--apply-closures` | review | Actually close items (use carefully) |

## Code Examples

### Reading an item record programmatically

```javascript
import { readFileSync } from 'fs';
import { join } from 'path';

function loadItemRecord(itemNumber) {
  const filePath = join('items', `${itemNumber}.md`);
  const content = readFileSync(filePath, 'utf-8');
  
  const outcomeMatch = content.match(/\*\*Outcome:\*\*\s+(\S+)/);
  const snapshotMatch = content.match(/\*\*Snapshot hash:\*\*\s+(\S+)/);
  
  return {
    number: itemNumber,
    outcome: outcomeMatch?.[1] ?? 'unknown',
    snapshotHash: snapshotMatch?.[1] ?? null,
    raw: content,
  };
}

const record = loadItemRecord(58150);
console.log(record.outcome); // "keep_open" or "proposed_close"
```

### Listing all proposed closes

```javascript
import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';

function getProposedCloses(itemsDir = 'items') {
  const files = readdirSync(itemsDir).filter(f => f.endsWith('.md'));
  
  return files
    .map(file => {
      const content = readFileSync(join(itemsDir, file), 'utf-8');
      const number = parseInt(file.replace('.md', ''), 10);
      const outcomeMatch = content.match(/\*\*Outcome:\*\*\s+(\S+)/);
      return { number, outcome: outcomeMatch?.[1] };
    })
    .filter(item => item.outcome === 'proposed_close')
    .map(item => item.number);
}

const toClose = getProposedCloses();
console.log(`Proposed closes: ${toClose.length}`);
console.log(toClose);
```

### Checking dashboard metrics from the README

```javascript
import { readFileSync } from 'fs';

function parseDashboard(readmePath = 'README.md') {
  const content = readFileSync(readmePath, 'utf-8');
  
  const stateMatch = content.match(/State:\s+(.+)/);
  const checkpointMatch = content.match(/Checkpoint (\d+) finished/);
  const totalClosesMatch = content.match(/Total fresh closes in this run:\s+(\d+)\/(\d+)/);
  
  return {
    state: stateMatch?.[1]?.trim(),
    checkpoint: checkpointMatch ? parseInt(checkpointMatch[1], 10) : null,
    closesApplied: totalClosesMatch ? parseInt(totalClosesMatch[1], 10) : null,
    closesLimit: totalClosesMatch ? parseInt(totalClosesMatch[2], 10) : null,
  };
}

const dashboard = parseDashboard();
console.log(dashboard);
// { state: 'Apply in progress', checkpoint: 7, closesApplied: 350, closesLimit: 500 }
```

### Triggering a workflow dispatch via GitHub API

```javascript
import { Octokit } from '@octokit/rest';

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

async function triggerApplyRun({ limit = 50, kind = 'issue', minAgeDays = 0 } = {}) {
  await octokit.actions.createWorkflowDispatch({
    owner: 'openclaw',
    repo: 'clawsweeper',
    workflow_id: 'sweep.yml',
    ref: 'main',
    inputs: {
      apply_existing: 'true',
      apply_closures: 'true',
      apply_kind: kind,
      apply_min_age_days: String(minAgeDays),
    },
  });
  console.log(`Apply run triggered: limit=${limit}, kind=${kind}`);
}

await triggerApplyRun({ limit: 100, kind: 'issue' });
```

## Common Patterns

### Safe incremental apply

Apply a small batch first to verify the bot is behaving correctly:

```bash
# Step 1: review only (default, no closes)
npm run plan -- --batch-size 5 --shard-count 10
npm run review -- --openclaw-dir ../myrepo --artifact-dir artifacts/reviews
npm run apply-artifacts -- --artifact-dir artifacts/reviews

# Step 2: inspect proposals
ls items/ | wc -l
grep -l "proposed_close" items/*.md

# Step 3: apply a small batch
npm run apply-decisions -- --limit 5 --apply-kind issue
```

### Excluding PRs from apply runs

The default `apply_kind=issue` already excludes PRs. To explicitly target only issues:

```bash
npm run apply-decisions -- --limit 50 --apply-kind issue
```

### Reviewing a single item manually

```bash
# Run plan with a targeted batch override (if supported by your build)
npm run review -- \
  --openclaw-dir ../myrepo \
  --batch-size 1 \
  --artifact-dir artifacts/manual \
  --codex-timeout-ms 600000
```

Then check `artifacts/manual/` and `items/<number>.md` for the result.

## Troubleshooting

### Dashboard not updating

The dashboard is updated by the final publish job in the workflow. If it's stale:
- Check the workflow run linked in the README status block
- Look for a timed-out shard (shards time out at 75 minutes)
- Re-run the publish job manually if shards completed but publish failed

### Codex leaves changes in the checkout

ClawSweeper makes the target repo checkout read-only in CI and verifies it before and after each review. If Codex writes anything, the item is marked as failed. Locally, ensure the `--openclaw-dir` path is not the same repo you are actively editing.

### GitHub secondary rate limiting during apply

Apply mode implements long retry backoff automatically. When throttled, it posts a heartbeat to the dashboard:

```
Throttle heartbeat: checkpoint 3, processed 120, retry in 65s
```

If you need to stop and resume, re-dispatch with `apply_existing=true` — already-closed items are skipped automatically.

### Items reappearing after close

If an issue is reopened externally, reconcile moves it back from `closed/` to `items/` as stale so the planner re-reviews it:

```bash
npm run reconcile -- --dry-run  # check what would move
npm run reconcile               # apply
```

### Maintainer items being proposed for close

ClawSweeper reads `author_association` from the GitHub API. Items authored by `OWNER`, `MEMBER`, or `COLLABORATOR` are excluded from automated close actions at both plan and apply time. If a maintainer item appears as `proposed_close`, verify that the GitHub token has sufficient scope to read association data, and re-run the review for that item.
```
