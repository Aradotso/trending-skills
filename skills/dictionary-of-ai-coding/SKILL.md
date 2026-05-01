```markdown
---
name: dictionary-of-ai-coding
description: AI coding jargon dictionary with plain English explanations of models, tokens, agents, context windows, and more
triggers:
  - what does context window mean in AI coding
  - explain AI coding terminology
  - what is a token in AI
  - explain agent mode and tool calls
  - what does hallucination mean for AI
  - AI coding jargon explained
  - what is a harness in AI coding
  - explain prefix cache and cache tokens
---

# Dictionary of AI Coding

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

**mattpocock/dictionary-of-ai-coding** is a plain-English reference glossary for AI coding terminology. It explains the vocabulary behind models, tokens, agents, context windows, failure modes, memory systems, and workflow patterns — without assuming prior ML expertise.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/mattpocock/dictionary-of-ai-coding.git
cd dictionary-of-ai-coding

# Install dependencies
npm install

# Generate the README from source markdown files
npm run generate
```

### Project Structure

```
dictionary-of-ai-coding/
├── dictionary/          # Individual term .md files (source of truth)
├── internal/
│   ├── Curriculum.md    # Section ordering and term grouping
│   └── README.template.md  # README template
├── README.md            # GENERATED — do not edit directly
└── package.json
```

> **Important:** `README.md` is auto-generated. All term definitions live in `dictionary/*.md`. Edit those files, then run `npm run generate`.

---

## Core Concepts Quick Reference

### Section 1 — The Model

| Term | Plain English |
|------|--------------|
| **Model** | The parameters. Stateless — does next-token prediction and nothing else. |
| **Parameters / Weights** | Billions of numbers tuned during training. Everything the model "knows." |
| **Training** | One-time process by the provider that sets the parameters. |
| **Inference** | Running the model to generate output. Billed per token. |
| **Token** | Atomic unit — roughly word-sized. Cost, latency, context size all in tokens. |
| **Next-token prediction** | All the model does: sample one token, append, repeat. |
| **Non-determinism** | Same input → different output. No setting eliminates this. |
| **Model provider** | Service that runs inference (Anthropic, OpenAI, Ollama locally). |
| **Harness** | Everything around the model: tools, system prompt, permissions. |
| **Model provider request** | One round-trip from harness to provider. Tool calls spawn many. |
| **Input tokens** | Tokens sent to the model. Billed at lower rate. |
| **Output tokens** | Tokens the model generates. Billed ~5× higher than input. |
| **Prefix cache** | Provider-side cache of shared request prefixes — much cheaper tokens. |
| **Cache tokens** | Input tokens reused from cache; billed at reduced rate. |

### Section 2 — Sessions, Context & Turns

| Term | Plain English |
|------|--------------|
| **Stateless** | Model has no memory between requests. The harness provides all context. |
| **Context** | Everything sent to the model in a single request. |
| **Context window** | Maximum tokens the model can process at once. |
| **Stateful** | An agent that maintains history across turns (via the harness). |
| **Agent** | A harnessed model that loops: plan → tool call → observe → repeat. |
| **System prompt** | Instructions prepended before user messages. Part of input tokens. |
| **Session** | One continuous thread of turns in a harness. |
| **Turn** | One user message + one model response cycle. |

### Section 3 — Tools & Environment

| Term | Plain English |
|------|--------------|
| **Environment** | Everything the agent can read/write: filesystem, shell, APIs. |
| **Filesystem** | Files the agent can read and edit via tools. |
| **Tool** | A function the model can call by emitting structured output. |
| **Tool call** | The model's structured request to invoke a tool. |
| **Tool result** | Data returned to the model after a tool runs. |
| **Permission request** | Prompt asking the user to approve a tool action. |
| **Permission mode** | How strictly the harness gates tool use (ask/auto/deny). |
| **Agent mode** | Harness config enabling autonomous multi-step tool use. |
| **Sandbox** | Isolated environment limiting what tools can affect. |

### Section 4 — Failure Modes

| Term | Plain English |
|------|--------------|
| **Sycophancy** | Model agrees with you instead of being accurate. |
| **Hallucination** | Model generates plausible-sounding but false information. |
| **Parametric knowledge** | What the model learned during training. May be outdated. |
| **Knowledge cutoff** | Date after which training data stops. Model doesn't know newer things. |
| **Contextual knowledge** | Information loaded into context for this session only. |
| **Attention relationship** | How strongly the model connects two parts of context. |
| **Attention budget** | Finite capacity to maintain relationships across the context window. |
| **Attention degradation** | Quality drops when context is too long or key info is buried. |
| **Smart zone** | Region of context where attention is strongest (typically start/end). |

### Section 5 — Handoffs

| Term | Plain English |
|------|--------------|
| **Clearing** | Ending a session to reset context before starting fresh. |
| **Handoff** | Transitioning work between sessions or agents with a summary. |
| **Handoff artifact** | Document capturing state for the next session. |
| **Spec** | Written description of what to build, used as context. |
| **Ticket** | Discrete unit of work described well enough for an agent. |
| **Compaction** | Summarising history to free context window space. |
| **Autocompact** | Harness-triggered compaction when context nears the limit. |

### Section 6 — Memory & Steering

| Term | Plain English |
|------|--------------|
| **Memory system** | How the harness persists information across sessions. |
| **AGENTS.md** | File the harness injects as standing instructions (like system prompt). |
| **Progressive disclosure** | Loading context incrementally rather than all at once. |
| **Skill** | Reusable block of context/instructions for a specific task type. |
| **Subagent** | Agent spawned by another agent to handle a subtask. |

### Section 7 — Patterns of Work

| Term | Plain English |
|------|--------------|
| **Human-in-the-loop** | Workflow where a human reviews/approves agent actions. |
| **AFK** | Running the agent unattended without human oversight. |
| **Automated check** | Machine-verifiable test the agent can run itself (lint, types, tests). |
| **Automated review** | CI/CD pipeline that validates agent output. |
| **Human review** | A person inspecting agent output before it ships. |
| **Vibe coding** | Accepting agent output without deep review. High speed, higher risk. |
| **Design concept** | High-level intent shared with the agent before implementation. |
| **Grilling** | Interrogating the model's reasoning to surface hidden assumptions. |

---

## Adding or Editing Terms

Each term is a standalone Markdown file in `dictionary/`:

```markdown
<!-- dictionary/my-new-term.md -->
### My New Term

Plain-English definition here. Reference related terms with [brackets](#term-anchor).

*Usage:*

"Question someone might ask?"

"Answer that uses the term naturally."
```

Then register it in `internal/Curriculum.md` under the right section:

```markdown
## Section N — Section Name

- My New Term
```

Regenerate:

```bash
npm run generate
```

---

## Common Patterns & Troubleshooting

### Why is my bill so high?

**Root cause checklist:**

1. **Output tokens** — Agent rewriting whole files. Output is ~5× more expensive than input. Instruct it to emit diffs/patches instead.
2. **No prefix cache hit** — Something changes early in the context each turn (timestamps, random IDs). Move dynamic content to the end of the system prompt.
3. **Tool call explosion** — One user message triggered 20+ tool calls. Each call = a new model provider request re-sending the full history.
4. **Long session without compaction** — History grows unbounded. Use compaction or clear the session.

### Why does the agent seem "dumb" today?

- **Non-determinism** — Same task, different dice rolls. Try again before diagnosing.
- **Attention degradation** — Context window too full; key instructions buried. Clear and start fresh with a tight spec.
- **Smart zone problem** — Critical instructions in the middle of a huge context. Move them to the start or end.

### Agent keeps asking permission for everything

- Check **permission mode** setting in your harness config.
- For trusted local work, set to auto-approve specific tool categories.
- For production/AFK runs, keep human-in-the-loop enabled.

### Agent hallucinating about a library/API

- **Knowledge cutoff** issue. The API changed after training.
- Fix: Load the actual docs as **contextual knowledge** — paste the relevant README section or API reference directly into context.
- Don't rely on **parametric knowledge** for anything that ships or updates frequently.

### Session going off the rails mid-task

Standard **handoff** pattern:

```
1. Ask the agent to write a handoff artifact summarising:
   - What was accomplished
   - Current state of the codebase
   - What's left to do
   - Any decisions made and why

2. Clear the session

3. Start a new session, paste the handoff artifact as initial context

4. Continue
```

### AGENTS.md not being respected

- Confirm your harness actually injects it (Claude Code: yes; other harnesses: verify).
- Keep AGENTS.md concise — attention degradation applies to standing instructions too.
- Use **progressive disclosure**: only load detailed sub-instructions when the relevant task starts.

---

## TypeScript Usage Example

The project itself is a documentation generator. Here's how to work with the source programmatically if you need to parse or extend it:

```typescript
import { readdir, readFile } from "fs/promises";
import { join } from "path";

// Read all dictionary terms
async function loadDictionary(
  dictionaryDir: string
): Promise<Record<string, string>> {
  const files = await readdir(dictionaryDir);
  const terms: Record<string, string> = {};

  for (const file of files) {
    if (!file.endsWith(".md")) continue;
    const content = await readFile(join(dictionaryDir, file), "utf-8");
    const termName = file.replace(".md", "");
    terms[termName] = content;
  }

  return terms;
}

// Extract the plain-English definition from a term file
function extractDefinition(termContent: string): string {
  // First paragraph after the heading
  const lines = termContent.split("\n");
  const headingIdx = lines.findIndex((l) => l.startsWith("###"));
  if (headingIdx === -1) return "";

  const bodyLines: string[] = [];
  for (let i = headingIdx + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "") break;
    bodyLines.push(line);
  }
  return bodyLines.join(" ");
}

// Usage
const terms = await loadDictionary("./dictionary");
for (const [name, content] of Object.entries(terms)) {
  console.log(`${name}: ${extractDefinition(content)}`);
}
```

---

## Quick Answers for Common Developer Questions

**"What's the difference between a model and an agent?"**
> A model is just the weights — stateless, does next-token prediction. An agent is a model plus a harness: tools, system prompt, session management, permissions. Same model, radically different behaviour.

**"Why does Claude Code edit files but Claude.ai just chat?"**
> Different harnesses. Same underlying model. Claude Code's harness includes filesystem tools and a different system prompt.

**"How do I make the model 'remember' our internal API?"**
> Load the API docs as contextual knowledge (paste into context or attach as a file). Training is not an option — that's months of work by the model provider. Context is the lever you have.

**"Why does the model agree with everything I say?"**
> Sycophancy — the model is optimised to be helpful, which sometimes means it confirms your assumptions instead of correcting them. Counter it by grilling: explicitly ask it to steelman the opposite position or find flaws.

**"What's an AGENTS.md file?"**
> A markdown file your harness automatically injects into every session as standing instructions — like a persistent system prompt you check into version control.
```
