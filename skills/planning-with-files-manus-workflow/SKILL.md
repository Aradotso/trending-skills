```markdown
---
name: planning-with-files-manus-workflow
description: Persistent markdown planning workflow for AI coding agents — the Manus-style 3-file pattern for context-aware, goal-tracking task execution.
triggers:
  - set up persistent planning files for my project
  - use manus style planning workflow
  - create task plan progress tracking files
  - help me plan complex tasks with markdown files
  - install planning with files skill
  - set up agent planning workflow like manus
  - use planning files to track my coding session
  - keep track of progress across context resets
---

# Planning with Files

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Persistent markdown planning for AI coding agents — the exact workflow pattern that powered Manus AI's $2B acquisition. Instead of relying on volatile context, this skill externalises memory to the filesystem using three structured markdown files that persist across sessions, context resets, and tool call chains.

## Core Concept

```
Context Window = RAM  (volatile, limited)
Filesystem     = Disk (persistent, unlimited)

→ Anything important gets written to disk immediately.
```

## Installation

### Quickest (works with Claude Code, Cursor, Codex, Gemini CLI, 40+ agents)

```bash
npx skills add OthmanAdi/planning-with-files --skill planning-with-files -g
```

### Claude Code Plugin (adds `/plan` autocomplete)

```
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files
```

### Copy skill locally (removes prefix requirement)

**macOS/Linux:**
```bash
cp -r ~/.claude/plugins/cache/planning-with-files/planning-with-files/*/skills/planning-with-files ~/.claude/skills/
```

**Windows (PowerShell):**
```powershell
Copy-Item -Recurse `
  -Path "$env:USERPROFILE\.claude\plugins\cache\planning-with-files\planning-with-files\*\skills\planning-with-files" `
  -Destination "$env:USERPROFILE\.claude\skills\"
```

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `/planning-with-files:plan` | `/plan` | Start a new planning session |
| `/planning-with-files:status` | `/plan:status` | Show current planning progress |
| `/planning-with-files:start` | `/planning` | Original start command |

## The 3-File Pattern

For every complex task, create exactly **three files** in the project root:

```
task_plan.md    → Phases, goals, checkboxes, error log
findings.md     → Research results, discoveries, references
progress.md     → Session log, test results, current state
```

### `task_plan.md` — Master Plan

```markdown
# Task Plan: [Feature/Task Name]

## Goal
[One-paragraph description of what done looks like]

## Phases

### Phase 1: Research & Setup
- [ ] Understand existing codebase structure
- [ ] Identify affected modules
- [ ] Check for related tests

### Phase 2: Implementation
- [ ] Create new module `src/feature.py`
- [ ] Update `src/main.py` to integrate feature
- [ ] Add error handling

### Phase 3: Testing & Validation
- [ ] Write unit tests
- [ ] Run full test suite
- [ ] Verify edge cases

## Current Status
**Active Phase:** Phase 1
**Last Updated:** 2026-03-18

## Errors & Blockers
<!-- Log failures here so they aren't repeated -->
- [2026-03-18] Import error in `utils.py` — missing `requests` dep, fixed by adding to requirements.txt

## Notes
- API rate limit: 100 req/min
- Auth token stored in $API_TOKEN env var
```

### `findings.md` — Research Storage

```markdown
# Findings: [Task Name]

## Architecture Notes
- Entry point: `src/main.py:run()`
- Config loaded from `config/settings.yaml`
- Database: PostgreSQL via SQLAlchemy ORM

## API Behaviour
- POST /api/v1/items returns 201 with `{"id": "uuid", "status": "created"}`
- Rate limit headers: `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- Auth: Bearer token in Authorization header

## Key Files
| File | Purpose |
|------|---------|
| `src/models.py` | SQLAlchemy models |
| `src/api/client.py` | HTTP client wrapper |
| `tests/conftest.py` | Pytest fixtures |

## Gotchas
- `session.commit()` must be called explicitly — no autocommit
- Timezone: all datetimes stored as UTC, displayed as local
```

### `progress.md` — Session Log

```markdown
# Progress Log: [Task Name]

## Session 2026-03-18

### Completed
- [x] Cloned repo, confirmed Python 3.11 environment
- [x] Read through `src/api/client.py` — documented in findings.md
- [x] Created skeleton for `src/feature.py`

### In Progress
- [ ] Implementing `FeatureProcessor.process()` method

### Test Results
```
pytest tests/test_feature.py -v
PASSED tests/test_feature.py::test_basic_flow
FAILED tests/test_feature.py::test_edge_case_empty_input
  AssertionError: expected ValueError, got None
```

### Next Session Start
1. Fix empty input edge case in `FeatureProcessor.process()`
2. Add integration test for API round-trip
```

## Python Implementation Pattern

When writing the actual agent logic or hooks, follow this pattern:

```python
from pathlib import Path
import datetime

PLAN_DIR = Path(".")  # or a subfolder like Path(".planning")

def get_plan_files():
    return {
        "plan":     PLAN_DIR / "task_plan.md",
        "findings": PLAN_DIR / "findings.md",
        "progress": PLAN_DIR / "progress.md",
    }

def initialize_planning(task_description: str) -> None:
    """Create the 3-file planning scaffold for a new task."""
    files = get_plan_files()
    today = datetime.date.today().isoformat()

    if not files["plan"].exists():
        files["plan"].write_text(f"""# Task Plan: {task_description}

## Goal
{task_description}

## Phases

### Phase 1: Research & Setup
- [ ] Understand existing codebase

### Phase 2: Implementation
- [ ] Implement solution

### Phase 3: Validation
- [ ] Test and verify

## Current Status
**Active Phase:** Phase 1
**Last Updated:** {today}

## Errors & Blockers

## Notes
""")

    if not files["findings"].exists():
        files["findings"].write_text(f"# Findings: {task_description}\n\n")

    if not files["progress"].exists():
        files["progress"].write_text(
            f"# Progress Log: {task_description}\n\n## Session {today}\n\n"
        )

    print(f"Planning files created in {PLAN_DIR.resolve()}")


def reread_plan() -> str:
    """Load the current plan — call this before every major decision."""
    plan_file = PLAN_DIR / "task_plan.md"
    if plan_file.exists():
        return plan_file.read_text()
    return "No task_plan.md found. Run initialize_planning() first."


def log_error(error_message: str) -> None:
    """Persist an error so it won't be repeated next session."""
    plan_file = PLAN_DIR / "task_plan.md"
    if not plan_file.exists():
        return
    today = datetime.date.today().isoformat()
    content = plan_file.read_text()
    entry = f"- [{today}] {error_message}\n"
    content = content.replace("## Errors & Blockers\n", f"## Errors & Blockers\n{entry}")
    plan_file.write_text(content)


def log_progress(entry: str) -> None:
    """Append a progress entry to progress.md."""
    progress_file = PLAN_DIR / "progress.md"
    today = datetime.date.today().isoformat()
    with progress_file.open("a") as f:
        f.write(f"\n- [{today}] {entry}\n")


def store_finding(section: str, content: str) -> None:
    """Append a finding to findings.md under a named section."""
    findings_file = PLAN_DIR / "findings.md"
    with findings_file.open("a") as f:
        f.write(f"\n## {section}\n{content}\n")


def check_completion() -> dict:
    """
    Parse task_plan.md and return completion stats.
    Returns: {"total": int, "done": int, "pending": list[str]}
    """
    plan_file = PLAN_DIR / "task_plan.md"
    if not plan_file.exists():
        return {"total": 0, "done": 0, "pending": []}

    lines = plan_file.read_text().splitlines()
    total, done, pending = 0, 0, []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ["):
            total += 1
            if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                done += 1
            else:
                pending.append(stripped[6:])  # strip "- [ ] "

    return {"total": total, "done": done, "pending": pending}


def mark_complete(task_substring: str) -> bool:
    """Mark a task checkbox as done by matching a substring."""
    plan_file = PLAN_DIR / "task_plan.md"
    if not plan_file.exists():
        return False
    content = plan_file.read_text()
    # Find first unchecked item containing the substring
    import re
    pattern = re.compile(r"- \[ \] (.*?" + re.escape(task_substring) + r".*?)", re.IGNORECASE)
    match = pattern.search(content)
    if match:
        old = f"- [ ] {match.group(1)}"
        new = f"- [x] {match.group(1)}"
        plan_file.write_text(content.replace(old, new, 1))
        return True
    return False
```

## Hook Integration (Claude Code)

Hooks automatically re-read the plan before tool calls and verify completion on stop:

```json
// .claude/hooks.json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"from pathlib import Path; p=Path('task_plan.md'); print(p.read_text()) if p.exists() else None\""
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"exec(open(str(__import__('pathlib').Path.home()/'.claude/skills/planning-with-files/hooks/stop_check.py')).read())\""
          }
        ]
      }
    ]
  }
}
```

## Session Recovery

When context fills up and you run `/clear`, the skill auto-recovers:

1. Scans `~/.claude/projects/` for previous session data
2. Finds when planning files were last modified
3. Extracts conversation after that point
4. Presents a catchup summary

**Disable auto-compact to maximise context before clearing:**

```json
// ~/.claude/settings.json
{
  "autoCompact": false
}
```

## Workflow: Step by Step

```python
# 1. Start every complex task by initialising planning
initialize_planning("Add OAuth2 login to the API")

# 2. Before any major decision, re-read the plan
current_plan = reread_plan()
# → use `current_plan` content to guide next action

# 3. Store discoveries in findings (not in context)
store_finding(
    "OAuth Library Evaluation",
    "Using `authlib` v1.3 — supports PKCE, JWT, and refresh tokens.\n"
    "Install: pip install authlib\n"
)

# 4. Log progress after completing steps
log_progress("Implemented OAuth callback endpoint at /auth/callback")
mark_complete("Add error handling")

# 5. Log errors immediately when they occur
try:
    result = call_oauth_provider()
except TokenExpiredError as e:
    log_error(f"OAuth token expired during flow — {e}. Fixed by adding token refresh in client.py")

# 6. Check completion before stopping
stats = check_completion()
print(f"Progress: {stats['done']}/{stats['total']} tasks complete")
if stats["pending"]:
    print("Still pending:")
    for task in stats["pending"]:
        print(f"  - {task}")
```

## Manus Principles Cheatsheet

| Principle | What to do |
|-----------|-----------|
| **Filesystem as memory** | Write findings/progress to files, not into prompts |
| **Attention manipulation** | Re-read `task_plan.md` before every major decision |
| **Error persistence** | Log ALL failures in `task_plan.md` → Errors section |
| **Goal tracking** | Use `- [ ]` / `- [x]` checkboxes for every step |
| **Completion verification** | Check all boxes before declaring task done |
| **Session continuity** | Start each session by reading all three files |

## Troubleshooting

### Planning files not created
- Confirm you ran `/planning-with-files:plan` or called `initialize_planning()`
- Check write permissions in the project directory
- On Windows, ensure PowerShell execution policy allows scripts: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Hooks not firing in Claude Code
- Verify `.claude/hooks.json` is valid JSON: `python3 -m json.tool .claude/hooks.json`
- Check the plugin is installed: `/plugin list`
- For Mastra Code, use `hooks.json` at workspace root, not `.claude/`

### Session recovery shows false positives
- Upgrade to v2.15.1+: `npx skills add OthmanAdi/planning-with-files --skill planning-with-files -g`
- The fix addresses catchup triggering when no real context was lost

### Garbled characters in GitHub Copilot / Copilot terminal
- Upgrade to v2.18.1+ which includes the PS1 UTF-8 encoding fix
- Set terminal encoding: `$OutputEncoding = [System.Text.Encoding]::UTF8`

### `/plan` autocomplete not working
- Ensure you installed via the Claude Code plugin method (not just `npx skills add`)
- Plugin path: `~/.claude/plugins/cache/planning-with-files/`
- Reinstall: `/plugin install planning-with-files@planning-with-files`

### OpenCode partial support warning
- Session catchup is limited in OpenCode — manually read `progress.md` at session start
- All other features (3-file pattern, hooks) work normally

## Platform-Specific Notes

| Platform | Guide | Key Difference |
|----------|-------|---------------|
| Gemini CLI | `docs/gemini.md` | Uses Agent Skills spec |
| Cursor | `docs/cursor.md` | Skills + Hooks config |
| Kiro | `docs/kiro.md` | Steering files format |
| Codex | `docs/codex.md` | Personal Skill path |
| GitHub Copilot | `docs/copilot.md` | Hooks-only (no skill runner) |
| BoxLite | `docs/boxlite.md` | Load via ClaudeBox inside micro-VM |

## Links

- **Repo:** https://github.com/OthmanAdi/planning-with-files
- **Homepage:** https://www.aikux.ai
- **Changelog:** https://github.com/OthmanAdi/planning-with-files/blob/main/CHANGELOG.md
- **Evals:** https://github.com/OthmanAdi/planning-with-files/blob/main/docs/evals.md
- **Agent Skills spec:** https://agentskills.io
```
