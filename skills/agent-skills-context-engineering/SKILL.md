```markdown
---
name: agent-skills-context-engineering
description: Comprehensive collection of Agent Skills for context engineering, multi-agent architectures, memory systems, and production agent systems using Claude Code and Cursor plugins.
triggers:
  - "build agent skills for context engineering"
  - "install context engineering plugins for Claude Code"
  - "design multi-agent architecture with context management"
  - "implement memory systems for AI agents"
  - "optimize agent context window usage"
  - "debug lost-in-middle context problems"
  - "evaluate agent performance with LLM-as-judge"
  - "create production-grade agent system"
---

# Agent Skills for Context Engineering

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A curated, open collection of Agent Skills teaching context engineering — the discipline of curating everything that enters a model's context window (system prompts, tool definitions, retrieved docs, message history, tool outputs) to maximize agent effectiveness in production systems.

## What This Project Provides

- **Skills** (`.md` files) that AI coding agents load progressively as context-engineering guides
- **Plugins** installable via Claude Code plugin marketplace or Cursor plugin directory
- **Examples** demonstrating complete multi-agent system designs

---

## Installation

### Claude Code Plugin Marketplace

```bash
# Step 1: Register the marketplace
/plugin marketplace add muratcankoylan/Agent-Skills-for-Context-Engineering

# Step 2: Install individual plugin bundles
/plugin install context-engineering-fundamentals@context-engineering-marketplace
/plugin install agent-architecture@context-engineering-marketplace
/plugin install agent-evaluation@context-engineering-marketplace
/plugin install agent-development@context-engineering-marketplace
/plugin install cognitive-architecture@context-engineering-marketplace
```

### Cursor (Open Plugins)

Listed on [cursor.directory/plugins/context-engineering](https://cursor.directory/plugins/context-engineering). The `.plugin/plugin.json` follows the Open Plugins standard and works with Codex, GitHub Copilot, and any conformant agent tool.

### Manual / Custom Agent Frameworks

```bash
git clone https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering.git
cd Agent-Skills-for-Context-Engineering
```

Copy or reference skill files from `skills/` into your agent's system prompt or skills directory.

---

## Plugin Bundles & Included Skills

| Plugin | Skills Included |
|--------|----------------|
| `context-engineering-fundamentals` | context-fundamentals, context-degradation, context-compression, context-optimization |
| `agent-architecture` | multi-agent-patterns, memory-systems, tool-design, filesystem-context, hosted-agents |
| `agent-evaluation` | evaluation, advanced-evaluation |
| `agent-development` | project-development |
| `cognitive-architecture` | bdi-mental-states |

---

## Repository Structure

```
Agent-Skills-for-Context-Engineering/
├── .plugin/
│   └── plugin.json                  # Open Plugins manifest
├── skills/
│   ├── context-fundamentals/        # Context window anatomy & attention budget
│   ├── context-degradation/         # Lost-in-middle, poisoning, clash patterns
│   ├── context-compression/         # Compression & summarization strategies
│   ├── context-optimization/        # Compaction, masking, KV-cache strategies
│   ├── multi-agent-patterns/        # Orchestrator, peer-to-peer, hierarchical
│   ├── memory-systems/              # Short-term, long-term, graph-based memory
│   ├── tool-design/                 # Effective tool interfaces for agents
│   ├── filesystem-context/          # File-based context offloading & discovery
│   ├── hosted-agents/               # Sandboxed VM agents, multiplayer, Modal
│   ├── evaluation/                  # Agent evaluation frameworks
│   ├── advanced-evaluation/         # LLM-as-a-Judge, pairwise, rubric gen
│   ├── project-development/         # Task-model fit, pipeline architecture
│   └── bdi-mental-states/           # BDI ontology, RDF→beliefs transformation
└── examples/
    ├── digital-brain-skill/         # Personal OS for founders (6 modules)
    ├── x-to-book-system/            # Multi-agent X→daily book pipeline
    ├── llm-as-judge-skills/         # TypeScript LLM eval tools (19 tests)
    └── book-sft-pipeline/           # Style-transfer SFT pipeline ($2 cost)
```

---

## Core Concepts & Key Patterns

### 1. Progressive Disclosure (Loading Skills Efficiently)

Skills load in layers to minimize context overhead:

```python
# Level 1: Load skill index only (names + one-line descriptions)
skill_index = load_skill_index("skills/")  # ~500 tokens

# Level 2: Activate a specific skill on task match
if task_matches(triggers["context-compression"]):
    skill_content = load_skill("skills/context-compression/SKILL.md")  # ~3000 tokens

# Level 3: Load examples/data only when needed
if needs_example:
    example = load_example("skills/context-compression/examples/")
```

### 2. Context Compression Pattern

```python
from anthropic import Anthropic

client = Anthropic()

def compress_conversation(messages: list[dict], target_tokens: int = 2000) -> list[dict]:
    """
    Compress long conversation history using progressive summarization.
    Implements the context-compression skill pattern.
    """
    if estimate_tokens(messages) <= target_tokens:
        return messages

    # Keep system prompt + last N turns verbatim
    recent_turns = messages[-4:]  # last 2 exchanges
    old_turns = messages[1:-4]    # exclude system prompt

    # Summarize older turns
    summary_response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the key facts, decisions, and context from this conversation. Be dense and precise:\n\n{format_messages(old_turns)}"
            }
        ]
    )

    summary_message = {
        "role": "assistant",
        "content": f"[COMPRESSED CONTEXT] {summary_response.content[0].text}"
    }

    return [messages[0], summary_message] + recent_turns


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // 4
```

### 3. Multi-Agent Orchestrator Pattern

```python
import os
from anthropic import Anthropic

client = Anthropic()
ORCHESTRATOR_MODEL = "claude-opus-4-5"
SUBAGENT_MODEL = "claude-haiku-4-5"

def orchestrator_agent(task: str, context: dict) -> str:
    """
    Implements orchestrator pattern from multi-agent-patterns skill.
    Routes subtasks to specialized subagents with minimal context.
    """
    # Orchestrator plans — receives full context
    plan_response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=1000,
        system="You are an orchestrator. Decompose tasks into atomic subtasks. Output JSON list of subtasks.",
        messages=[
            {"role": "user", "content": f"Task: {task}\nContext: {context}"}
        ]
    )

    subtasks = parse_json(plan_response.content[0].text)
    results = {}

    for subtask in subtasks:
        # Subagents receive only what they need (minimal context injection)
        result = subagent_execute(
            task=subtask["description"],
            required_context={k: context[k] for k in subtask.get("needs", [])}
        )
        results[subtask["id"]] = result

    # Orchestrator synthesizes — injects only results, not intermediate traces
    synthesis = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=2000,
        system="Synthesize subtask results into a coherent final answer.",
        messages=[
            {"role": "user", "content": f"Original task: {task}\nResults: {results}"}
        ]
    )
    return synthesis.content[0].text


def subagent_execute(task: str, required_context: dict) -> str:
    """Subagent with scoped, minimal context."""
    response = client.messages.create(
        model=SUBAGENT_MODEL,
        max_tokens=500,
        messages=[
            {"role": "user", "content": f"Task: {task}\nContext: {required_context}"}
        ]
    )
    return response.content[0].text
```

### 4. Memory System (Append-Only JSONL)

```python
import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(os.environ.get("AGENT_MEMORY_DIR", "./agent_memory"))
MEMORY_DIR.mkdir(exist_ok=True)

# Schema-first line pattern from digital-brain-skill example
MEMORY_SCHEMA = {
    "version": "1.0",
    "fields": ["timestamp", "type", "entity", "content", "tags"]
}

def init_memory_file(name: str) -> Path:
    """Create append-only JSONL with schema header."""
    path = MEMORY_DIR / f"{name}.jsonl"
    if not path.exists():
        with open(path, "w") as f:
            f.write(json.dumps({"__schema__": MEMORY_SCHEMA}) + "\n")
    return path

def remember(memory_type: str, entity: str, content: str, tags: list[str] = None) -> None:
    """Append a memory entry. Never mutate — always append."""
    path = init_memory_file(memory_type)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": memory_type,
        "entity": entity,
        "content": content,
        "tags": tags or []
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def recall(memory_type: str, query_entity: str = None, limit: int = 10) -> list[dict]:
    """Load recent memories, optionally filtered by entity."""
    path = MEMORY_DIR / f"{memory_type}.jsonl"
    if not path.exists():
        return []

    entries = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "__schema__" in entry:
                continue
            if query_entity is None or entry["entity"] == query_entity:
                entries.append(entry)

    return entries[-limit:]  # Return most recent


# Usage
remember("contacts", "Alice Chen", "Met at NeurIPS. Working on RAG eval.", tags=["ml", "rag"])
memories = recall("contacts", limit=5)
```

### 5. Tool Design — Self-Describing Tools

```python
from anthropic import Anthropic
import json

client = Anthropic()

# Tool design skill: tools must have precise, unambiguous descriptions
# and return structured, parseable output
TOOLS = [
    {
        "name": "search_codebase",
        "description": "Search the codebase for files or symbols matching a query. Returns file paths with line numbers. Use for: finding implementations, locating definitions, discovering usage patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — can be a symbol name, file pattern (*.py), or natural language description"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["py", "ts", "md", "json", "any"],
                    "description": "Filter results by file extension"
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return (1-50)"
                }
            },
            "required": ["query"]
        }
    }
]

def run_agent_with_tools(user_message: str) -> str:
    """Agentic loop with tool use."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})


def execute_tool(name: str, inputs: dict) -> dict:
    """Execute tool and return structured result."""
    if name == "search_codebase":
        # Implementation would call ripgrep, ast-grep, etc.
        return {"matches": [], "total": 0, "query": inputs["query"]}
    return {"error": f"Unknown tool: {name}"}
```

### 6. LLM-as-Judge Evaluation

```python
import os
from anthropic import Anthropic
from dataclasses import dataclass

client = Anthropic()

@dataclass
class EvalResult:
    score: float          # 0.0 - 1.0
    reasoning: str
    criteria_scores: dict

JUDGE_SYSTEM = """You are an expert evaluator. Score responses objectively using the provided rubric.
Output JSON: {"score": 0.0-1.0, "reasoning": "...", "criteria": {"criterion": score}}"""

def direct_score(
    prompt: str,
    response: str,
    rubric: dict[str, float]  # {"criterion": weight}
) -> EvalResult:
    """
    Direct scoring from advanced-evaluation skill.
    rubric = {"accuracy": 0.4, "completeness": 0.3, "clarity": 0.3}
    """
    eval_prompt = f"""Rate this response using these weighted criteria:
{json.dumps(rubric, indent=2)}

Prompt: {prompt}
Response: {response}

Score each criterion 0-1, then compute weighted_total."""

    result = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": eval_prompt}]
    )

    parsed = json.loads(result.content[0].text)
    return EvalResult(
        score=parsed["score"],
        reasoning=parsed["reasoning"],
        criteria_scores=parsed.get("criteria", {})
    )


def pairwise_compare(
    prompt: str,
    response_a: str,
    response_b: str
) -> dict:
    """
    Pairwise comparison with position bias mitigation.
    Runs A-vs-B and B-vs-A, detects bias.
    """
    def compare(p: str, r1: str, r2: str, label1: str, label2: str) -> str:
        result = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Which response is better?\nPrompt: {p}\n{label1}: {r1}\n{label2}: {r2}\nOutput: {{'winner': '{label1}|{label2}|tie', 'reasoning': '...'}}"}]
        )
        return json.loads(result.content[0].text)["winner"]

    forward = compare(prompt, response_a, response_b, "A", "B")
    backward = compare(prompt, response_b, response_a, "B", "A")

    # Normalize backward result
    backward_normalized = {"B": "A", "A": "B", "tie": "tie"}[backward]

    return {
        "winner": forward if forward == backward_normalized else "inconclusive",
        "position_bias_detected": forward != backward_normalized,
        "forward_result": forward,
        "backward_result": backward_normalized
    }
```

### 7. Filesystem Context Offloading

```python
import os
import json
from pathlib import Path

# filesystem-context skill: offload large tool outputs to files
# so the agent's context window only holds file references

SCRATCH_DIR = Path(os.environ.get("AGENT_SCRATCH_DIR", "/tmp/agent_scratch"))
SCRATCH_DIR.mkdir(exist_ok=True)

def offload_to_file(key: str, data: any) -> str:
    """
    Write large data to disk; return a compact file reference for context.
    Agent holds the reference string, not the raw data.
    """
    filepath = SCRATCH_DIR / f"{key}.json"
    with open(filepath, "w") as f:
        json.dump(data, f)

    size_kb = filepath.stat().st_size // 1024
    return f"[FILE_REF:{filepath}|size:{size_kb}KB|key:{key}]"  # ~50 tokens vs potentially thousands


def load_from_file(file_ref: str) -> any:
    """Reconstruct data from a file reference string."""
    import re
    match = re.search(r"FILE_REF:([^|]+)", file_ref)
    if not match:
        raise ValueError(f"Invalid file reference: {file_ref}")

    with open(match.group(1)) as f:
        return json.load(f)


# Agent plan persistence — survives context resets
def save_plan(plan: dict) -> None:
    plan_file = SCRATCH_DIR / "current_plan.json"
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=2)

def load_plan() -> dict | None:
    plan_file = SCRATCH_DIR / "current_plan.json"
    if plan_file.exists():
        with open(plan_file) as f:
            return json.load(f)
    return None
```

### 8. BDI Mental State Transformation

```python
# bdi-mental-states skill: transform external context into formal BDI mental states

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Belief:
    """What the agent believes about the world (from RDF/external context)."""
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    source: str = "observation"

@dataclass
class Desire:
    """Agent goals — what states it wants to achieve."""
    goal_id: str
    description: str
    priority: float  # 0.0 - 1.0
    preconditions: list[str] = field(default_factory=list)

@dataclass
class Intention:
    """Committed plan to achieve a desire."""
    desire_id: str
    plan: list[str]  # ordered action sequence
    status: str = "active"  # active | suspended | achieved | failed

class BDIAgent:
    def __init__(self):
        self.beliefs: list[Belief] = []
        self.desires: list[Desire] = []
        self.intentions: list[Intention] = []

    def perceive(self, rdf_triples: list[tuple]) -> None:
        """Convert RDF context triples into agent beliefs."""
        for subject, predicate, obj in rdf_triples:
            self.beliefs.append(Belief(
                subject=str(subject),
                predicate=str(predicate),
                object=str(obj)
            ))

    def deliberate(self) -> list[Desire]:
        """Filter desires whose preconditions are satisfied by current beliefs."""
        belief_facts = {(b.subject, b.predicate, b.object) for b in self.beliefs}
        achievable = []
        for desire in self.desires:
            if all(self._check_precondition(p, belief_facts) for p in desire.preconditions):
                achievable.append(desire)
        return sorted(achievable, key=lambda d: d.priority, reverse=True)

    def to_context(self) -> str:
        """Serialize BDI state as compact context for LLM prompt injection."""
        return json.dumps({
            "beliefs": [{"s": b.subject, "p": b.predicate, "o": b.object} for b in self.beliefs[-20:]],
            "top_desire": self.desires[0].description if self.desires else None,
            "active_intention": next((i.plan for i in self.intentions if i.status == "active"), None)
        })

    def _check_precondition(self, precondition: str, facts: set) -> bool:
        # Simplified — real implementation would use SPARQL or logic engine
        return any(precondition in str(fact) for fact in facts)
```

---

## Skill Triggers Reference

| Skill | Activate When User Says |
|-------|------------------------|
| `context-fundamentals` | "understand context", "explain context windows", "design agent architecture" |
| `context-degradation` | "diagnose context problems", "fix lost-in-middle", "debug agent failures" |
| `context-compression` | "compress context", "summarize conversation", "reduce token usage" |
| `context-optimization` | "optimize context", "reduce token costs", "implement KV-cache" |
| `multi-agent-patterns` | "design multi-agent system", "implement supervisor pattern" |
| `memory-systems` | "implement agent memory", "build knowledge graph", "track entities" |
| `tool-design` | "design agent tools", "reduce tool complexity", "implement MCP tools" |
| `filesystem-context` | "offload context to files", "dynamic context discovery", "agent scratch pad" |
| `hosted-agents` | "build background agent", "sandboxed execution", "Modal sandboxes" |
| `evaluation` | "evaluate agent performance", "build test framework", "measure quality" |
| `advanced-evaluation` | "implement LLM-as-judge", "compare model outputs", "mitigate bias" |
| `project-development` | "start LLM project", "design batch pipeline", "evaluate task-model fit" |
| `bdi-mental-states` | "model agent mental states", "implement BDI architecture", "transform RDF to beliefs" |

---

## Common Patterns & Troubleshooting

### Lost-in-Middle Problem

**Symptom**: Agent ignores crucial information passed mid-conversation.

**Fix** (from `context-degradation` skill):
```python
def structure_context_for_attention(context: dict) -> str:
    """
    Place critical info at START and END of context.
    U-shaped attention: models attend best to beginning and end.
    """
    critical = context.get("critical_constraints", [])
    background = context.get("background", "")

    return f"""CRITICAL CONSTRAINTS (read first):
{chr(10).join(f'- {c}' for c in critical)}

BACKGROUND CONTEXT:
{background}

REMINDER — CRITICAL CONSTRAINTS:
{chr(10).join(f'- {c}' for c in critical)}"""
```

### Context Poisoning

**Symptom**: Stale or incorrect facts persist and corrupt later reasoning.

**Fix**: Use explicit invalidation markers + always filter memories by timestamp:
```python
def add_belief(beliefs: list, new_belief: dict) -> list:
    """Invalidate contradicting beliefs before appending."""
    filtered = [
        b for b in beliefs
        if b["entity"] != new_belief["entity"] or b["predicate"] != new_belief["predicate"]
    ]
    filtered.append({**new_belief, "timestamp": datetime.utcnow().isoformat()})
    return filtered
```

### Token Budget Management

```python
def build_context_within_budget(
    system_prompt: str,
    memories: list[str],
    documents: list[str],
    max_tokens: int = 8000
) -> str:
    """
    Fills context budget from highest to lowest priority.
    system_prompt > recent_memories > relevant_documents
    """
    budget = max_tokens
    parts = []

    # Always include system prompt
    parts.append(system_prompt)
    budget -= estimate_tokens(system_prompt)

    # Add memories until budget exhausted
    for memory in memories:
        tokens = estimate_tokens(memory)
        if budget - tokens < 500:  # Keep 500 token buffer
            break
        parts.append(memory)
        budget -= tokens

    # Fill remaining budget with documents
    for doc in documents:
        tokens = estimate_tokens(doc)
        if budget - tokens < 500:
            break
        parts.append(doc)
        budget -= tokens

    return "\n\n".join(parts)
```

### Skill Not Activating in Claude Code

```bash
# Verify plugin installation
/plugin list

# Re-install if missing
/plugin install agent-architecture@context-engineering-marketplace

# Check skill triggers match your phrasing — use exact trigger words:
# ✓ "design multi-agent system"
# ✗ "create agents" (too vague)
```

---

## Environment Variables

```bash
# Agent memory storage location
export AGENT_MEMORY_DIR=/path/to/memory

# Scratch space for context offloading
export AGENT_SCRATCH_DIR=/tmp/agent_scratch

# Anthropic API key for examples
export ANTHROPIC_API_KEY=your_api_key_here
```

---

## References

- [Skills directory](skills/) — All skill SKILL.md files
- [Examples directory](examples/) — Complete system designs
- [Meta Context Engineering paper](https://arxiv.org/pdf/2601.21557) — Academic citation of this work
- [Cursor Plugin Directory](https://cursor.directory/plugins/context-engineering)
- [Open Plugins standard](https://open-plugins.com)
```
