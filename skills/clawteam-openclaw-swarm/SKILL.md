```markdown
---
name: clawteam-openclaw-swarm
description: Multi-agent swarm coordination with OpenClaw as default agent — spawn, coordinate, and monitor CLI agent teams using git worktrees and tmux
triggers:
  - use clawteam to coordinate agents
  - spawn multiple agents to work in parallel
  - set up a multi-agent swarm
  - coordinate AI agents across tasks
  - use openclaw with clawteam
  - split work across multiple coding agents
  - launch an agent team for this project
  - orchestrate agents with clawteam
---

# ClawTeam-OpenClaw — Multi-Agent Swarm Coordination

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

ClawTeam-OpenClaw is a fork of [HKUDS/ClawTeam](https://github.com/HKUDS/ClawTeam) that turns CLI coding agents (OpenClaw, Claude Code, Codex, nanobot) into self-organizing swarms. A leader agent spawns workers, assigns tasks with dependency chains, coordinates via inbox messaging, and merges results — all through a single `clawteam` CLI. Each worker gets an isolated git worktree and tmux window.

---

## Installation

### Prerequisites

```bash
python3 --version   # Need 3.10+
tmux -V             # Any version
openclaw --version  # Or: claude --version / codex --version
```

Install missing tools:

```bash
# macOS
brew install python@3.12 tmux
pip install openclaw

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip tmux
pip install openclaw
```

### Install ClawTeam-OpenClaw

> Do NOT use `pip install clawteam` — that installs the upstream version without OpenClaw support.

```bash
git clone https://github.com/win4r/ClawTeam-OpenClaw.git
cd ClawTeam-OpenClaw
pip install -e .

# Optional: ZeroMQ P2P transport
pip install -e ".[p2p]"
```

### Symlink for spawned agents

Spawned agents run in fresh shells without pip's bin in PATH. Create a symlink:

```bash
mkdir -p ~/bin
ln -sf "$(which clawteam)" ~/bin/clawteam

# Add ~/bin to PATH if not already there (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/bin:$PATH"
```

### OpenClaw skill and exec approvals (OpenClaw users only)

```bash
# Install the ClawTeam skill so OpenClaw agents understand coordination commands
mkdir -p ~/.openclaw/workspace/skills/clawteam
cp skills/openclaw/SKILL.md ~/.openclaw/workspace/skills/clawteam/SKILL.md

# Allow spawned agents to run clawteam without interactive prompts
python3 -c "
import json, pathlib
p = pathlib.Path.home() / '.openclaw' / 'exec-approvals.json'
if p.exists():
    d = json.loads(p.read_text())
    d.setdefault('defaults', {})['security'] = 'allowlist'
    p.write_text(json.dumps(d, indent=2))
    print('Updated: security = allowlist')
else:
    print('Run openclaw once first, then re-run this step')
"
openclaw approvals allowlist add --agent "*" "*/clawteam"
```

### Automated installer (Steps 2–5 in one shot)

```bash
git clone https://github.com/win4r/ClawTeam-OpenClaw.git
cd ClawTeam-OpenClaw
bash scripts/install-openclaw.sh
```

### Verify

```bash
clawteam --version
clawteam config health        # All green = ready
openclaw skills list | grep clawteam  # OpenClaw users only
```

---

## Core Concepts

| Concept | What it is |
|---|---|
| **Team** | Named group of agents sharing a task board and inbox |
| **Leader** | The coordinating agent that spawns and monitors workers |
| **Worker** | A spawned agent with its own git worktree and tmux window |
| **Worktree** | Isolated git branch (`clawteam/{team}/{agent}`) — no merge conflicts |
| **Task** | Unit of work with status: `pending → in_progress → completed / blocked` |
| **Inbox** | Point-to-point or broadcast message queue between agents |
| **Transport** | File-based (default) or ZeroMQ P2P |

---

## Key CLI Commands

### Team management

```bash
# Create a team (leader registers itself)
clawteam team spawn-team my-team \
  --description "Build the auth module" \
  --agent-name leader

# List all teams
clawteam team list

# Show team members and status
clawteam team show my-team
```

### Spawning workers

```bash
# Spawn an OpenClaw worker (default)
clawteam spawn --team my-team \
  --agent-name alice \
  --task "Implement OAuth2 flow"

# Spawn with explicit agent type
clawteam spawn tmux openclaw --team my-team \
  --agent-name bob \
  --task "Write unit tests for auth"

# Spawn a Claude Code worker
clawteam spawn tmux claude --team my-team \
  --agent-name carol \
  --task "Set up CI pipeline"

# Spawn a Codex worker
clawteam spawn tmux codex --team my-team \
  --agent-name dave \
  --task "Optimize database queries"

# Subprocess backend (non-tmux, e.g. Cursor)
clawteam spawn subprocess cursor --team my-team \
  --agent-name eve \
  --task "Review and document APIs"
```

### Task management

```bash
# List all tasks for a team
clawteam task list my-team

# List tasks owned by a specific agent
clawteam task list my-team --owner alice

# List tasks by status
clawteam task list my-team --status pending
clawteam task list my-team --status in_progress
clawteam task list my-team --status completed

# Create a task manually
clawteam task create my-team \
  --title "Implement login endpoint" \
  --description "POST /auth/login, returns JWT" \
  --owner alice

# Create a task with dependency (bob's task unblocks after alice completes hers)
clawteam task create my-team \
  --title "Write auth tests" \
  --owner bob \
  --blocked-by <alice-task-id>

# Update task status
clawteam task update my-team <task-id> --status completed

# Block until all tasks in a team are done (useful in scripts)
clawteam task wait my-team
```

### Inter-agent messaging

```bash
# Send a message from leader to alice
clawteam inbox send my-team alice \
  "Please use the OpenAPI spec at docs/api.yaml for all endpoint contracts."

# Read your inbox (as the agent named 'alice')
clawteam inbox receive my-team alice

# Peek without consuming messages
clawteam inbox peek my-team alice

# Broadcast to all team members
clawteam inbox broadcast my-team \
  "API schema finalized. All teams unblocked."
```

### Monitoring and dashboards

```bash
# Terminal kanban board (snapshot)
clawteam board show my-team

# Auto-refreshing live dashboard
clawteam board live my-team

# Tiled tmux view — all agent windows side by side
clawteam board attach my-team

# Web dashboard (real-time updates)
clawteam board serve --port 8080
clawteam board serve my-team --port 8080  # Filter to one team
```

### Worktree and merge operations

```bash
# Checkpoint current worktree state (commit in worker branch)
clawteam worktree checkpoint my-team alice \
  --message "OAuth2 flow complete"

# Merge a worker's worktree back into main
clawteam worktree merge my-team alice --into main

# Merge all workers
clawteam worktree merge-all my-team --into main

# Clean up worktrees after merging
clawteam worktree cleanup my-team
```

### Template-based launches

```bash
# Launch a predefined team from a TOML template
clawteam launch hedge-fund \
  --team fund1 \
  --goal "Analyze AAPL, MSFT, NVDA for Q2 2026"

# List available templates
clawteam launch --list
```

---

## Configuration

ClawTeam stores team state in `.clawteam/` in your project root:

```
.clawteam/
  teams/
    my-team/
      manifest.json       # Team metadata and member list
      tasks.json          # Task board
      inbox/              # Per-agent message queues
        alice.jsonl
        bob.jsonl
```

Global config:

```bash
# Show current config
clawteam config show

# Run health check
clawteam config health

# Set default agent type
clawteam config set default_agent openclaw

# Set default transport (file or zmq)
clawteam config set transport file
```

---

## Real Code Examples

### Python: orchestrate a team programmatically

```python
import subprocess
import json

def clawteam(*args) -> str:
    """Run a clawteam command and return stdout."""
    result = subprocess.run(
        ["clawteam", *args],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

# Create team
clawteam("team", "spawn-team", "myteam",
         "--description", "Build REST API",
         "--agent-name", "leader")

# Spawn workers
clawteam("spawn", "--team", "myteam",
         "--agent-name", "backend",
         "--task", "Implement CRUD endpoints for /users")

clawteam("spawn", "--team", "myteam",
         "--agent-name", "tester",
         "--task", "Write pytest suite for /users endpoints")

# Monitor until done
import time
while True:
    output = clawteam("task", "list", "myteam", "--status", "in_progress")
    remaining = [l for l in output.splitlines() if l.strip()]
    if not remaining:
        break
    print(f"Tasks in progress: {len(remaining)}")
    time.sleep(30)

# Merge all worktrees
clawteam("worktree", "merge-all", "myteam", "--into", "main")
clawteam("worktree", "cleanup", "myteam")
print("Done. All branches merged.")
```

### Python: send a message and wait for reply

```python
import subprocess, time, json

TEAM = "myteam"

def send(to: str, msg: str):
    subprocess.run(
        ["clawteam", "inbox", "send", TEAM, to, msg],
        check=True
    )

def receive(agent: str, timeout: int = 300) -> list[str]:
    """Poll inbox until a message arrives or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["clawteam", "inbox", "receive", TEAM, agent],
            capture_output=True, text=True
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if lines:
            return lines
        time.sleep(5)
    raise TimeoutError(f"No message for {agent} within {timeout}s")

# Leader asks backend for API spec location
send("backend", "Where is the final OpenAPI spec? Reply with the file path.")
replies = receive("leader")
spec_path = replies[0]
print(f"Backend replied: {spec_path}")
```

### TOML team template

Create `templates/fullstack.toml`:

```toml
[team]
name = "{team_name}"
description = "{goal}"

[[agents]]
name = "architect"
task = "Design system architecture and OpenAPI spec for: {goal}"
agent = "openclaw"

[[agents]]
name = "backend"
task = "Implement backend per architect's OpenAPI spec"
agent = "openclaw"
blocked_by = ["architect"]

[[agents]]
name = "frontend"
task = "Build React frontend consuming the backend API"
agent = "openclaw"
blocked_by = ["architect"]

[[agents]]
name = "tester"
task = "Write integration and unit tests covering backend and frontend"
agent = "openclaw"
blocked_by = ["backend", "frontend"]

[[agents]]
name = "devops"
task = "Set up CI/CD pipeline, Dockerfile, and deployment config"
agent = "openclaw"
blocked_by = ["backend"]
```

Launch it:

```bash
clawteam launch fullstack \
  --team myapp \
  --goal "Build a SaaS invoicing tool with Stripe integration"
```

### Shell script: full pipeline

```bash
#!/usr/bin/env bash
set -euo pipefail

TEAM="research-$(date +%s)"
GOAL="${1:-Optimize the training loop in train.py}"

echo "==> Creating team: $TEAM"
clawteam team spawn-team "$TEAM" \
  --description "$GOAL" \
  --agent-name leader

echo "==> Spawning research agents"
for i in 1 2 3 4; do
  clawteam spawn --team "$TEAM" \
    --agent-name "researcher$i" \
    --task "Research direction $i for: $GOAL"
done

echo "==> Watching live dashboard (Ctrl-C to exit)"
clawteam board live "$TEAM" &
DASH_PID=$!

echo "==> Waiting for all tasks to complete"
clawteam task wait "$TEAM"
kill $DASH_PID 2>/dev/null || true

echo "==> Merging results"
clawteam worktree merge-all "$TEAM" --into main
clawteam worktree cleanup "$TEAM"

echo "==> Done. Check git log for merged branches."
git log --oneline -10
```

---

## Common Patterns

### Pattern 1: Dependency chain (sequential with parallelism)

```bash
# Step 1: architect works alone
clawteam spawn --team myteam --agent-name architect \
  --task "Design schema and API contracts"

# Step 2: backend + frontend unblock after architect
ARCH_TASK=$(clawteam task list myteam --owner architect --json | jq -r '.[0].id')

clawteam task create myteam \
  --title "Backend implementation" \
  --owner backend \
  --blocked-by "$ARCH_TASK"

clawteam task create myteam \
  --title "Frontend implementation" \
  --owner frontend \
  --blocked-by "$ARCH_TASK"

# Both backend and frontend agents spawn and self-block until architect finishes
clawteam spawn --team myteam --agent-name backend --task "See assigned task"
clawteam spawn --team myteam --agent-name frontend --task "See assigned task"
```

### Pattern 2: Leader-worker with result aggregation

```bash
# Leader spawns workers, waits, then reads inbox for results
clawteam spawn --team myteam --agent-name worker1 --task "Analyze logs/server1.log"
clawteam spawn --team myteam --agent-name worker2 --task "Analyze logs/server2.log"

# Workers send results to leader when done:
# clawteam inbox send myteam leader "server1: 3 critical errors at 14:32, 15:01, 15:44"

# Leader polls inbox
clawteam inbox receive myteam leader
```

### Pattern 3: Broadcast coordination

```bash
# Notify all agents of a shared context change
clawteam inbox broadcast myteam \
  "IMPORTANT: API base URL changed to https://api.example.com/v2. Update all clients."
```

### Pattern 4: ZeroMQ P2P transport for low-latency teams

```bash
pip install -e ".[p2p]"
clawteam config set transport zmq

# All subsequent spawns use ZeroMQ instead of file-based queues
clawteam spawn --team myteam --agent-name worker1 --task "Low-latency task"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `clawteam: command not found` | pip bin not in PATH | `ln -sf $(which clawteam) ~/bin/clawteam` and add `~/bin` to PATH |
| Spawned agents can't find `clawteam` | Fresh shell misses pip PATH | Verify `~/bin/clawteam` symlink exists |
| Agents block on permission prompts | exec-approvals security is "full" | Run the exec approvals config step above |
| `openclaw approvals` fails | Gateway not running | Run `openclaw gateway` first, then retry |
| `exec-approvals.json not found` | OpenClaw never initialized | Run `openclaw` once, then retry the approvals step |
| `pip install -e .` fails | Missing build deps | `pip install hatchling` first |
| Tasks stuck in `pending` | Worker crashed or not spawned | `clawteam board show <team>` — check tmux windows |
| Merge conflicts on `merge-all` | Two workers edited same file | Resolve in each worktree branch before merging |
| Web dashboard not loading | Port in use | `clawteam board serve --port 9090` |

### Debugging a stuck team

```bash
# See all team members and their last-known status
clawteam team show myteam

# Check task board
clawteam task list myteam

# Attach to all agent windows in a tiled tmux view
clawteam board attach myteam

# Read any unread messages
clawteam inbox peek myteam leader
clawteam inbox peek myteam alice
```

### Reset a team

```bash
# Clean up worktrees without merging (destructive)
clawteam worktree cleanup myteam --force

# Delete team state entirely
rm -rf .clawteam/teams/myteam
```

---

## Supported Agents Reference

| Agent | Spawn syntax | Notes |
|---|---|---|
| OpenClaw | `clawteam spawn --team T --agent-name N` | Default — no flag needed |
| Claude Code | `clawteam spawn tmux claude --team T --agent-name N` | Requires `claude` in PATH |
| Codex | `clawteam spawn tmux codex --team T --agent-name N` | Requires `codex` in PATH |
| nanobot | `clawteam spawn tmux nanobot --team T --agent-name N` | Requires `nanobot` in PATH |
| Cursor | `clawteam spawn subprocess cursor --team T --agent-name N` | Experimental |
| Custom script | `clawteam spawn subprocess python myscript.py --team T --agent-name N` | Any CLI agent |

---

## Resources

- **Repo:** https://github.com/win4r/ClawTeam-OpenClaw
- **Upstream:** https://github.com/HKUDS/ClawTeam
- **OpenClaw:** https://openclaw.ai
- **Demo video:** https://youtu.be/aZT9d8qrirY
- **Autoresearch use case:** https://github.com/novix-science/autoresearch
```
