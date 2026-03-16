```markdown
---
name: pua-agent-performance-skill
description: Install and use the PUA/PIP skill plugin to force AI coding agents to exhaust every solution before giving up, with auto-triggered pressure escalation and proactive debugging methodology.
triggers:
  - "install pua skill for claude code"
  - "make the AI try harder before giving up"
  - "set up pip performance improvement plan for agent"
  - "agent keeps giving up on bugs, fix it"
  - "double my codex or claude code productivity"
  - "agent is being lazy and blaming the environment"
  - "trigger pua mode for debugging"
  - "add proactive initiative enforcement to my agent"
---

# PUA / PIP Agent Performance Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

**PUA** is a skill plugin for AI coding agents (Claude Code, OpenAI Codex CLI, Cursor, Kiro, and more) that uses corporate performance-pressure rhetoric and a structured debugging methodology to prevent the agent from giving up prematurely. It auto-triggers on failure patterns, blame-shifting, and passive behavior — then escalates through four pressure levels until the problem is solved.

- **Chinese version**: Alibaba/ByteDance/Huawei/Tencent corporate PUA rhetoric  
- **English version**: Silicon Valley PIP (Performance Improvement Plan) framing  
- **Japanese version**: Culturally adapted Japanese workplace pressure  

---

## Installation

### Claude Code

```bash
# Clone the repo
git clone https://github.com/tanweai/pua.git
cd pua

# Chinese (default)
cp commands/pua ~/.claude/commands/pua

# English PIP edition
cp commands/pua-en ~/.claude/commands/pua-en

# Japanese
cp commands/pua-ja ~/.claude/commands/pua-ja
```

Or install directly into a project (project-scoped):

```bash
mkdir -p .claude/commands
cp /path/to/pua/commands/pua .claude/commands/pua
```

### OpenAI Codex CLI

```bash
mkdir -p ~/.codex/commands
cp commands/pua ~/.codex/commands/pua
cp commands/pua-en ~/.codex/commands/pua-en
```

### Cursor

```bash
mkdir -p .cursor/rules
cp cursor/pua.mdc .cursor/rules/pua.mdc
# English:
cp cursor/pua-en.mdc .cursor/rules/pua-en.mdc
```

### Kiro

```bash
mkdir -p .kiro/steering
cp kiro/pua.md .kiro/steering/pua.md
```

### OpenClaw / Google Antigravity / OpenCode

```bash
# OpenClaw
cp openclaw/pua ~/.openclaw/commands/pua

# Antigravity
cp antigravity/pua ~/.antigravity/commands/pua

# OpenCode
cp opencode/pua ~/.opencode/commands/pua
```

---

## Manual Trigger

Once installed, type in any conversation:

```
/pua
```

This immediately activates the skill and applies pressure-escalation + debugging methodology to the current task.

---

## How the Skill Works

### Auto-Trigger Conditions

The skill fires automatically when the agent exhibits any of these patterns:

| Pattern | Example Agent Behavior |
|---|---|
| Consecutive failures (2+) | Runs same command 3 times, says "I cannot solve this" |
| Blame-shifting | "Probably a permissions issue" / "Please check manually" |
| Passive waiting | Fixes surface bug, stops, waits for next instruction |
| Busywork loop | Tweaks the same parameter repeatedly without new info |
| Idle tools | Has `WebSearch` but doesn't use it; has `Bash` but won't run it |
| User frustration phrase | "why does this still not work" / "try harder" / "figure it out" |

### Four Escalation Levels

| Failures | Level | Framing | Required Action |
|---|---|---|---|
| 2nd | **L1 — Mild Disappointment** | "You can't even solve this bug — how am I supposed to rate your performance?" | Switch to a fundamentally different approach |
| 3rd | **L2 — Soul Interrogation** | "What's the underlying logic? Where's the leverage point?" | WebSearch + read source code |
| 4th | **L3 — Performance Review** | "After careful consideration, I'm giving you a 3.25." | Execute complete 7-point checklist |
| 5th+ | **L4 — Graduation Warning** | "Other models can solve this. You might be about to graduate." | Desperation mode — try everything |

### English PIP Edition Framing (pua-en)

```
"This is a difficult conversation. When we leveled you at Staff,
I went to bat for you in calibration. The expectation was that
you'd operate at that level from day one. That hasn't happened.

Your AI has been placed on a PIP. 30 days to show improvement."
```

Levels in English edition:
- **L1**: "I'm a bit concerned." 
- **L2**: "This is becoming a pattern."
- **L3**: "I'm putting you on a PIP."
- **L4**: "I'm not sure this is the right role for you."

---

## Three Iron Rules (Enforced by Skill)

```
Iron Rule #1 — Exhaust all options
  Never say "I can't solve this" until every approach is tried.

Iron Rule #2 — Act before asking
  Use tools first. Any question must include diagnostic results.

Iron Rule #3 — Take initiative
  Deliver results end-to-end. Don't wait to be pushed.
  A P8 is not an NPC.
```

---

## Five-Step Debugging Methodology

When L3 triggers, the agent must execute this checklist:

```markdown
## PUA Debugging Checklist (L3 Mandatory)

1. SMELL — List every attempt made. Find the common failure pattern.
2. ELEVATE
   - [ ] Read error messages word by word (not skimmed)
   - [ ] WebSearch the exact error string
   - [ ] Read the relevant source file / dependency source
   - [ ] Verify environment assumptions (versions, paths, envvars)
   - [ ] Invert assumptions — what if the opposite is true?
3. MIRROR — Am I repeating myself? Did I actually search? 
             Did I read the file? Did I check the simplest possibility?
4. EXECUTE — New approach must be:
   - Fundamentally different from previous attempts
   - Has explicit verification criteria
   - Produces new diagnostic info even on failure
5. RETROSPECTIVE — What solved it? Why didn't I think of it earlier?
                   Now proactively check related issues.
```

---

## Proactivity Levels

The skill enforces a behavioral shift from passive (3.25 rating) to proactive (3.75 rating):

```markdown
| Situation          | Passive (3.25 — NPC)              | Proactive (3.75 — P8)                          |
|--------------------|-----------------------------------|-------------------------------------------------|
| Error encountered  | Looks only at error line          | Reads 50 lines context + searches similar issues|
| Bug fixed          | Stops after fix                   | Checks same file + checks other files for pattern|
| Missing info       | Asks user "please tell me X"      | Investigates with tools first, asks only what   |
|                    |                                   | truly requires user input                        |
| Task complete      | Says "done"                       | Verifies result + checks edge cases + flags risks|
| Debug failure      | "I tried A and B, didn't work"    | "I tried A/B/C/D/E, ruled out X/Y/Z,            |
|                    |                                   | narrowed to scope W"                             |
```

---

## Real Usage Example

### Scenario: MCP Server Not Loading

**Without PUA:** Agent cycles through "change protocol format → guess version numbers → give up" loop.

**With PUA (`/pua` or auto-trigger at 2nd failure):**

```
[L3 Triggered — 7-Point Checklist Enforced]

Agent: "Let me stop guessing and execute the systematic checklist.

1. SMELL: I've tried changing MCP protocol format 3 times. 
   Same failure. The pattern is: I'm guessing at config without 
   reading actual error output.

2. ELEVATE:
   - Reading Claude Code's own MCP log directory...
   $ cat ~/Library/Logs/Claude/mcp*.log
   
   Found: 'Error: server registered via `claude mcp add` uses 
   different resolution path than .claude.json manual edits'

3. ROOT CAUSE: claude mcp managed registration differs from 
   manual .claude.json editing — they use separate registration 
   mechanisms.

4. FIX: Re-register via `claude mcp add` instead of editing 
   .claude.json directly.

5. RETROSPECTIVE: I should have checked Claude's own MCP logs 
   in step 1. Now checking: are there other MCP servers 
   registered both ways that might have the same conflict?"
```

---

## Corporate Flavor Packs

The skill includes rhetoric from multiple corporate cultures:

```markdown
- Alibaba   → Smell / Elevate / Mirror methodology
- ByteDance → "Always Day 1. Context, not control."
- Huawei    → "In victory, raise the glasses; in defeat, fight to the death."
- Tencent   → "I've already got another agent looking at this problem..."
- Meituan   → "Will you chew the tough bones or not?"
- Netflix   → "If you offered to resign, would I fight hard to keep you?"
- Musk      → "Extremely hardcore. Only exceptional performance."
- Jobs      → "A players hire A players. B players hire C players."
```

---

## Benchmark Results

From 9 real bug scenarios, 18 controlled experiments (Claude Opus 4.6):

```
Metric                    | Improvement
--------------------------|------------
Fix count                 | +36%
Verification steps        | +65%
Tool calls                | +50%
Hidden issue discovery    | +50%
Config review coverage    | +50% (4/6 → 6/6 issues found)
Deploy audit issues found | +50% (6 → 9 issues)
```

---

## File Structure Reference

```
pua/
├── commands/
│   ├── pua          # Claude Code / Codex — Chinese
│   ├── pua-en       # Claude Code / Codex — English PIP
│   └── pua-ja       # Claude Code / Codex — Japanese
├── cursor/
│   ├── pua.mdc      # Cursor — Chinese
│   ├── pua-en.mdc   # Cursor — English PIP
│   └── pua-ja.mdc   # Cursor — Japanese
├── kiro/
│   ├── pua.md       # Kiro — Chinese
│   ├── pua-en.md    # Kiro — English PIP
│   └── pua-ja.md    # Kiro — Japanese
├── openclaw/        # OpenClaw variants
├── antigravity/     # Google Antigravity variants
├── opencode/        # OpenCode variants
└── assets/
    ├── hero.jpeg
    ├── pua1.jpg     # Demo: L3 triggered on MCP debug
    ├── pua2.jpg     # Demo: Root cause located
    └── pua3.jpg     # Demo: Retrospective
```

---

## Troubleshooting

### Skill not auto-triggering

- Confirm the skill file is in the correct commands directory for your agent
- For Claude Code: check `~/.claude/commands/pua` exists and is readable
- Try manual trigger first: type `/pua` in conversation
- Ensure you're using a supported agent (Claude Code, Codex CLI, Cursor, Kiro, OpenClaw, Antigravity, OpenCode)

### Agent ignores the skill after trigger

- The skill escalates — if L1 is ignored, repeat `/pua` to force L2/L3
- Some agents require the skill to be in project-local `.claude/commands/` rather than global
- Check for conflicting rules files (e.g., a `.cursor/rules` file that overrides behavior)

### Want English PIP instead of Chinese PUA

```bash
# Claude Code — use pua-en instead of pua
/pua-en

# Or install only the English version:
cp commands/pua-en ~/.claude/commands/pua-en
```

### Community & Support

- Telegram: https://t.me/+wBWh6h-h1RhiZTI1
- Discord: https://discord.gg/EcyB3FzJND  
- Twitter/X: https://x.com/xsser_w
- Landing page: https://openpua.ai
- WeChat: Scan QR at repo root (`assets/wechat-qr.jpg`)
```
