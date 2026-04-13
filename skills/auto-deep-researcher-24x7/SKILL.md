```markdown
---
name: auto-deep-researcher-24x7
description: Autonomous AI agent that runs deep learning experiments 24/7 using a Leader-Worker architecture with zero-cost monitoring and constant-size memory.
triggers:
  - "run my deep learning experiments automatically"
  - "set up autonomous experiment agent"
  - "automate hyperparameter tuning overnight"
  - "run experiments while I sleep"
  - "set up 24/7 ML experiment loop"
  - "autonomous deep learning research agent"
  - "auto experiment with GPU monitoring"
  - "continuous experiment iteration with LLM"
---

# Auto Deep Researcher 24x7

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

An autonomous AI agent that runs your deep learning experiments 24/7. It follows a **THINK → EXECUTE → REFLECT** loop: plans the next experiment, launches training on GPU, monitors at zero LLM cost (just process checks + log reads), then reflects and iterates — all without human intervention.

**Key properties:**
- Zero-cost monitoring during training (no LLM calls while GPU trains)
- Constant-size memory via rolling `MEMORY_LOG.md`
- Leader-Worker architecture for multi-GPU / multi-project
- ~$0.08/day average LLM cost
- Human stays in control via `PROJECT_BRIEF.md` and `HUMAN_DIRECTIVE.md`

---

## Installation

### Prerequisites

- Python 3.10+
- 1+ NVIDIA GPU
- Anthropic or OpenAI API key
- Claude Code, Codex CLI, or similar AI coding agent

### Clone and Install

```bash
git clone https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7.git
cd auto-deep-researcher-24x7
pip install -r requirements.txt
```

### Set API Key

```bash
# For Anthropic (Claude)
export ANTHROPIC_API_KEY=your_key_here

# For OpenAI (Codex)
export OPENAI_API_KEY=your_key_here
```

---

## Quickstart (3 Steps)

### Step 1 — Create your project folder

```bash
mkdir my-experiment
cd my-experiment
```

### Step 2 — Write `PROJECT_BRIEF.md`

This is the only required file. It defines the goal, constraints, and search strategy.

```markdown
# Goal
Train a ResNet-50 on CIFAR-100 to reach 80%+ top-1 accuracy.

# Codebase
Create the training code from scratch in PyTorch.

# What to Try
- Start with a basic ResNet-50 baseline (lr=0.1, SGD, cosine schedule).
- If accuracy < 75%, improve optimizer and learning rate schedule.
- If accuracy is 75-80%, try data augmentation (CutMix, AutoAugment).
- If accuracy > 80%, stop and write final report.

# Constraints
- Use GPU 0 only.
- Max 100 epochs per run.
- Do not change the backbone architecture.
```

### Step 3 — Launch the agent

```bash
# Inside Claude Code or Codex CLI:
/auto-experiment --project /path/to/my-experiment --gpu 0
```

The `workspace/` directory is auto-created. The agent starts its loop immediately.

---

## Project Structure

```
my-experiment/
├── PROJECT_BRIEF.md          # ← You write this (required)
├── HUMAN_DIRECTIVE.md        # ← Optional: override next cycle direction
├── config.yaml               # ← Optional: override defaults
└── workspace/                # ← Auto-created by agent
    ├── MEMORY_LOG.md         # Rolling memory of results and decisions
    ├── experiments/          # Per-run code, configs, logs
    ├── progress_tracking/    # Local text notes (if no Obsidian)
    │   ├── Dashboard.txt
    │   └── Daily/YYYY-MM-DD.txt
    └── results/              # Parsed results per experiment
```

---

## Key CLI Commands

```bash
# Start autonomous experiment loop
/auto-experiment --project /path/to/project --gpu 0

# Check current status (goal, best result, cycle count, running jobs)
/experiment-status

# Generate structured progress report
/progress-report

# Manually sync Obsidian notes (if configured)
/obsidian-sync

# Run a single THINK-EXECUTE-REFLECT cycle (manual trigger)
/run-cycle --project /path/to/project
```

---

## Configuration (`config.yaml`)

All fields are optional. The agent runs with defaults if this file is absent.

```yaml
# config.yaml — place in your project root

agent:
  model: "claude-opus-4-5"          # or "gpt-4o", "claude-sonnet-4-5"
  max_cycles: 50                    # stop after N cycles (0 = unlimited)
  cycle_timeout_hours: 6            # max hours per experiment before abort
  reflect_on_failure: true          # still call REFLECT if training crashes

gpu:
  devices: [0]                      # list of GPU IDs to use
  memory_fraction: 0.9              # reserved VRAM fraction per job

monitoring:
  poll_interval_seconds: 60         # how often to check process + logs
  log_tail_lines: 50                # lines from training log to read

memory:
  max_log_entries: 20               # rolling window size in MEMORY_LOG.md
  summarize_on_overflow: true       # summarize old entries instead of delete

obsidian:
  enabled: false                    # set true to sync to Obsidian vault
  vault_path: "~/Documents/MyVault" # path to vault root
  auto_append_daily: true           # append to daily note each cycle

safety:
  dry_run_before_launch: true       # always do a quick dry-run first
  block_path_traversal: true        # harden tool execution
  block_shell_injection: true
```

---

## Controlling the Agent at Runtime

### Stable direction — `PROJECT_BRIEF.md`

Edit this to change the overall goal or constraints. Takes effect at the next THINK phase.

### Temporary redirect — `HUMAN_DIRECTIVE.md`

Create or edit this file to override the next cycle only. The agent reads it, acts on it, then clears it.

```markdown
# HUMAN_DIRECTIVE.md (temporary — agent will clear after reading)

- Stop exploring augmentation for now.
- Return to the last baseline and try a different learning rate schedule.
- Do not change batch size this cycle.
```

### Common directive patterns

```markdown
# Narrow the search space
- Only tune learning rate and weight decay this cycle.
- Do not modify the model architecture.

# Prevent thrashing on a weak branch
- If accuracy gain is below 0.3 points for 3 consecutive runs, abandon this direction.
- Return to the best checkpoint and try a new idea.

# Force result verification
- If a result is more than 2 points above the previous best, rerun with a new seed.
- Do not record improvement until both runs agree within 0.5 points.

# Emergency stop
- Stop all new launches after the current run finishes.
- Write a summary of results so far.
```

---

## Code Examples

### Minimal training script the agent can generate and iterate on

The agent writes and modifies this itself, but knowing the pattern helps you write `PROJECT_BRIEF.md` clearly:

```python
# workspace/experiments/run_003/train.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import argparse, json, time, pathlib

def main(args):
    device = torch.device(f"cuda:{args.gpu}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = torchvision.datasets.CIFAR100("./data", train=True,
                                              download=True, transform=transform_train)
    val_ds   = torchvision.datasets.CIFAR100("./data", train=False,
                                              transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=256,
                                               shuffle=False, num_workers=4)

    model = resnet50(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        acc = correct / total * 100
        best_acc = max(best_acc, acc)

        # Log in a format the agent can parse
        print(f"[EPOCH {epoch+1}/{args.epochs}] val_acc={acc:.2f} best={best_acc:.2f}",
              flush=True)

    # Final result JSON — agent reads this in REFLECT phase
    pathlib.Path("result.json").write_text(json.dumps({
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "batch_size": args.batch_size,
    }))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    main(p.parse_args())
```

### Reading the rolling memory log in your own scripts

```python
# Read current agent memory state
from pathlib import Path
import re

memory_log = Path("workspace/MEMORY_LOG.md").read_text()

# Extract best result so far
match = re.search(r"best_val_acc[=:]\s*([\d.]+)", memory_log)
best = float(match.group(1)) if match else None
print(f"Best accuracy so far: {best}")
```

### Checking experiment status programmatically

```python
import subprocess, json

# Get JSON status from the agent CLI
result = subprocess.run(
    ["python", "-m", "deep_researcher.cli", "status", "--format", "json"],
    capture_output=True, text=True
)
status = json.loads(result.stdout)
print(f"Cycle: {status['cycle_count']}")
print(f"Best:  {status['best_result']}")
print(f"State: {status['current_state']}")  # THINK / EXECUTE / REFLECT / IDLE
```

---

## The THINK → EXECUTE → REFLECT Loop

```
┌──────────────────────────────────────────────────────────┐
│  THINK (LLM, ~$0.05)                                     │
│  • Read PROJECT_BRIEF.md + MEMORY_LOG.md                 │
│  • Read HUMAN_DIRECTIVE.md if present (then clear it)    │
│  • Decide next experiment config                         │
│  • Write/modify training code and config                 │
│  • Schedule a dry-run                                    │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│  EXECUTE (zero LLM cost during training)                 │
│  • Dry-run to catch syntax/config errors                 │
│  • Launch: python train.py --gpu 0 ... &                 │
│  • Sleep loop: kill -0 $PID + tail log (no API calls)   │
│  • Wake when process exits                               │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│  REFLECT (LLM, ~$0.03)                                   │
│  • Parse result.json and training log                    │
│  • Compare with baseline in MEMORY_LOG.md                │
│  • Update MEMORY_LOG.md (rolling, constant size)         │
│  • Decide: next direction / stop / escalate              │
│  • Sync progress notes (Obsidian or local text)          │
└──────────────┬───────────────────────────────────────────┘
               ↓
          (repeat or stop)
```

---

## Memory Architecture

The agent uses a **rolling log** to keep context size constant regardless of how many cycles have run.

```markdown
<!-- workspace/MEMORY_LOG.md — managed by agent, human-readable -->

## Summary (auto-updated)
Best result: ResNet-50, val_acc=78.4, run_007, AutoAugment + CutMix, lr=0.05

## Recent Experiments (last 20)

### run_008 [2026-04-12 03:21]
- Config: lr=0.05, wd=1e-4, AutoAugment, CutMix, batch=128
- Result: val_acc=78.4 (best so far)
- Decision: Try label smoothing next

### run_007 [2026-04-11 22:05]
- Config: lr=0.1, wd=5e-4, AutoAugment only
- Result: val_acc=76.9
- Decision: Add CutMix, reduce lr

## Abandoned Directions
- MixUp alone: no gain over baseline after 3 runs
- SGD + StepLR: underperforms CosineAnnealingLR consistently
```

---

## Multi-GPU / Multi-Project Setup

Run separate agent instances per project, each pinned to its own GPU:

```bash
# Terminal 1 — project A on GPU 0
/auto-experiment --project ~/projects/cifar100 --gpu 0

# Terminal 2 — project B on GPU 1
/auto-experiment --project ~/projects/imagenet-subset --gpu 1

# Terminal 3 — project C on GPU 2,3 (multi-GPU training)
/auto-experiment --project ~/projects/diffusion --gpu 2,3
```

Or use `tmux` / `screen` for persistent sessions:

```bash
tmux new-session -d -s exp-cifar   "cd ~/projects/cifar100   && /auto-experiment --gpu 0"
tmux new-session -d -s exp-imgnet  "cd ~/projects/imagenet   && /auto-experiment --gpu 1"
tmux attach -t exp-cifar           # check in anytime
```

---

## Obsidian Dashboard (Optional)

Enable live progress notes in your Obsidian vault:

```yaml
# config.yaml
obsidian:
  enabled: true
  vault_path: "~/Documents/ResearchVault"
  auto_append_daily: true
```

The agent writes to:
- `ResearchVault/Experiments/Dashboard.md` — current status, best result, cycle count
- `ResearchVault/Daily/YYYY-MM-DD.md` — appended each cycle with what happened

Without Obsidian, the same content goes to:
```
workspace/progress_tracking/Dashboard.txt
workspace/progress_tracking/Daily/YYYY-MM-DD.txt
```

---

## Troubleshooting

### Agent keeps repeating the same experiment

Add a directive to force exploration:

```markdown
# HUMAN_DIRECTIVE.md
- The last 3 runs used the same config. Try something meaningfully different.
- Pick a direction not yet explored according to MEMORY_LOG.md.
```

### Training crashes immediately (CUDA OOM, import error, etc.)

The dry-run should catch most issues. If it doesn't:
1. Check `workspace/experiments/run_NNN/train.log` for the error
2. Add a constraint to `PROJECT_BRIEF.md`:
   ```markdown
   # Constraints
   - Batch size must not exceed 64 (GPU has 8GB VRAM).
   - Always import dependencies at top of file, not inside functions.
   ```

### Agent stops after N cycles unexpectedly

Check `config.yaml` for `max_cycles`. Set to `0` for unlimited:

```yaml
agent:
  max_cycles: 0
```

### Memory log growing too large

Reduce the rolling window:

```yaml
memory:
  max_log_entries: 10
  summarize_on_overflow: true
```

### LLM costs higher than expected

Verify `dry_run_before_launch: true` is set (avoids wasted REFLECT calls on broken runs).
Also check `poll_interval_seconds` — it should not trigger LLM calls (it uses `kill -0` and `tail`, which are free).

### API key not found

```bash
# Verify the env var is set in the current shell
echo $ANTHROPIC_API_KEY
# If empty, export it:
export ANTHROPIC_API_KEY=$(cat ~/.secrets/anthropic_key)
```

---

## Cost Reference

| Scenario | LLM Cost |
|---|---|
| 1 experiment cycle (5h training) | ~$0.08 |
| 24h continuous operation | ~$0.08–$0.40 |
| 30-day autonomous run | ~$3–$12 |
| Training time (any duration) | $0.00 |

Cost is dominated by THINK + REFLECT phases. Training time is always free.
```
