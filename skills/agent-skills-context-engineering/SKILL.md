```markdown
---
name: agent-skills-context-engineering
description: Comprehensive collection of Agent Skills for context engineering, multi-agent architectures, memory systems, and production agent systems using Claude Code, Cursor, or any agent platform.
triggers:
  - "context engineering for agents"
  - "build multi-agent system"
  - "implement agent memory"
  - "design agent architecture"
  - "optimize context window"
  - "install agent skills Claude Code"
  - "debug agent context problems"
  - "evaluate agent performance"
---

# Agent Skills for Context Engineering

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A comprehensive, open collection of Agent Skills teaching context engineering principles for production-grade AI agent systems. Context engineering is the discipline of curating the language model's context window holistically — system prompts, tool definitions, retrieved documents, message history, and tool outputs — to maximize agent effectiveness within attention constraints.

## What This Project Provides

- **Skills**: Modular, loadable knowledge units for agent platforms (Claude Code, Cursor, Codex, Open Plugins)
- **Architectural patterns**: Orchestrator, peer-to-peer, hierarchical multi-agent designs
- **Memory systems**: Short-term, long-term, and graph-based memory architectures
- **Evaluation frameworks**: LLM-as-Judge, pairwise comparison, rubric generation
- **Complete examples**: Digital brain, X-to-book pipeline, book SFT pipeline, LLM-as-Judge tools

---

## Installation

### Claude Code (Plugin Marketplace)

```bash
# Register the marketplace
/plugin marketplace add muratcankoylan/Agent-Skills-for-Context-Engineering

# Install individual plugin bundles
/plugin install context-engineering-fundamentals@context-engineering-marketplace
/plugin install agent-architecture@context-engineering-marketplace
/plugin install agent-evaluation@context-engineering-marketplace
/plugin install agent-development@context-engineering-marketplace
/plugin install cognitive-architecture@context-engineering-marketplace
```

### Cursor

Listed on [cursor.directory/plugins/context-engineering](https://cursor.directory/plugins/context-engineering). Install via Cursor's plugin UI or reference the `.plugin/plugin.json` manifest directly.

### Manual / Custom Agent Frameworks

Clone the repo and reference skill files directly:

```bash
git clone https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering.git
```

Skills live under `skills/<skill-name>/SKILL.md`. Load them into your agent's system prompt or context as needed.

---

## Plugin Bundles and Included Skills

| Plugin | Skills Included |
|--------|----------------|
| `context-engineering-fundamentals` | context-fundamentals, context-degradation, context-compression, context-optimization |
| `agent-architecture` | multi-agent-patterns, memory-systems, tool-design, filesystem-context, hosted-agents |
| `agent-evaluation` | evaluation, advanced-evaluation |
| `agent-development` | project-development |
| `cognitive-architecture` | bdi-mental-states |

---

## Core Concepts

### Context Degradation Patterns

Agents fail in predictable ways when context is mismanaged:

```python
# Lost-in-middle: critical info buried in long context
# Models attend strongly to beginning and end, weakly to middle
# Mitigation: place high-signal tokens at start or end of context

# Context poisoning: stale/contradictory information accumulates
# Mitigation: explicit versioning, append-only logs with timestamps

# Attention scarcity: too many tools/docs dilute attention
# Mitigation: progressive disclosure — load only what's needed now
```

### Progressive Disclosure Pattern

Load context incrementally; never frontload everything:

```python
# Level 1: skill index (names + one-line descriptions) — always loaded
# Level 2: skill overview (key concepts, triggers) — loaded on activation
# Level 3: full skill content (examples, edge cases) — loaded on demand

class ProgressiveContextLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self._index = None

    def get_index(self) -> list[dict]:
        """Level 1: minimal — always in context."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def activate_skill(self, skill_name: str) -> str:
        """Level 2: load skill overview on trigger match."""
        path = f"{self.skills_dir}/{skill_name}/SKILL.md"
        with open(path) as f:
            content = f.read()
        # Return only frontmatter + first section for overview
        return self._extract_overview(content)

    def deep_load_skill(self, skill_name: str) -> str:
        """Level 3: full content for active task."""
        path = f"{self.skills_dir}/{skill_name}/SKILL.md"
        with open(path) as f:
            return f.read()

    def _load_index(self) -> list[dict]:
        import os, yaml
        index = []
        for skill_dir in os.listdir(self.skills_dir):
            skill_path = f"{self.skills_dir}/{skill_dir}/SKILL.md"
            if os.path.exists(skill_path):
                with open(skill_path) as f:
                    raw = f.read()
                # Parse YAML frontmatter
                if raw.startswith("---"):
                    fm_end = raw.index("---", 3)
                    fm = yaml.safe_load(raw[3:fm_end])
                    index.append({
                        "name": fm.get("name"),
                        "description": fm.get("description"),
                        "triggers": fm.get("triggers", []),
                    })
        return index

    def _extract_overview(self, content: str) -> str:
        lines = content.split("\n")
        # Return frontmatter + first 40 lines
        return "\n".join(lines[:40])
```

---

## Multi-Agent Patterns

### Orchestrator Pattern

```python
from anthropic import Anthropic

client = Anthropic()

ORCHESTRATOR_SYSTEM = """
You are an orchestrator agent. Break tasks into subtasks and delegate to
specialist agents. Maintain a task plan in your context. Never perform
specialist work yourself — delegate, collect results, synthesize.

Current plan format:
TASK: <overall goal>
SUBTASKS:
  [ ] subtask_id: description -> assigned_agent
  [x] subtask_id: description -> completed
RESULTS: {subtask_id: result_summary}
"""

def orchestrate(task: str, specialist_agents: dict) -> str:
    """
    specialist_agents: {agent_name: callable(task_description) -> str}
    """
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=ORCHESTRATOR_SYSTEM,
            messages=messages,
        )
        reply = response.content[0].text

        # Check if orchestrator is delegating
        if "DELEGATE:" in reply:
            agent_name, subtask = parse_delegation(reply)
            if agent_name in specialist_agents:
                result = specialist_agents[agent_name](subtask)
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"AGENT_RESULT from {agent_name}:\n{result}"
                })
                continue

        # Orchestrator produced final synthesis
        return reply


def parse_delegation(text: str) -> tuple[str, str]:
    """Extract DELEGATE: agent_name | subtask from orchestrator output."""
    for line in text.split("\n"):
        if line.startswith("DELEGATE:"):
            parts = line.replace("DELEGATE:", "").split("|", 1)
            return parts[0].strip(), parts[1].strip()
    return "", ""
```

### Peer-to-Peer Pattern

```python
class AgentMessage:
    def __init__(self, sender: str, recipient: str, content: str, msg_type: str = "task"):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.msg_type = msg_type  # task | result | clarification

class PeerAgentNetwork:
    def __init__(self):
        self.agents: dict[str, callable] = {}
        self.message_queue: list[AgentMessage] = []

    def register(self, name: str, handler: callable):
        self.agents[name] = handler

    def send(self, msg: AgentMessage):
        self.message_queue.append(msg)

    def process(self, max_rounds: int = 10) -> list[AgentMessage]:
        results = []
        for _ in range(max_rounds):
            if not self.message_queue:
                break
            msg = self.message_queue.pop(0)
            if msg.recipient in self.agents:
                response = self.agents[msg.recipient](msg)
                if response:
                    results.append(response)
                    self.message_queue.append(response)
        return results
```

---

## Memory Systems

### Append-Only JSONL Memory (Agent-Friendly)

```python
import json
from datetime import datetime, timezone
from pathlib import Path

class AgentMemory:
    """
    Append-only JSONL memory. Schema-first line enables fast parsing.
    Compatible with filesystem-context skill pattern.
    """

    SCHEMA = {
        "_schema": "1.0",
        "fields": ["timestamp", "type", "content", "tags", "importance"]
    }

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            self._write_line(self.SCHEMA)  # Schema as first line

    def remember(self, content: str, mem_type: str = "observation",
                 tags: list[str] = None, importance: float = 0.5):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": mem_type,
            "content": content,
            "tags": tags or [],
            "importance": importance,
        }
        self._write_line(entry)

    def recall(self, query_tags: list[str] = None,
               min_importance: float = 0.0,
               limit: int = 20) -> list[dict]:
        entries = []
        with open(self.path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if "_schema" in entry:
                    continue
                if entry["importance"] < min_importance:
                    continue
                if query_tags:
                    if not any(t in entry["tags"] for t in query_tags):
                        continue
                entries.append(entry)

        # Return most recent, highest importance first
        entries.sort(key=lambda e: (e["importance"], e["timestamp"]), reverse=True)
        return entries[:limit]

    def compress(self, summarizer: callable, keep_last: int = 10):
        """
        Summarize old entries to reduce context load.
        summarizer: callable(list[dict]) -> str
        """
        entries = self.recall(limit=10000)
        if len(entries) <= keep_last:
            return

        old = entries[keep_last:]
        summary_text = summarizer(old)

        # Rewrite file: schema + summary + recent
        recent = entries[:keep_last]
        new_path = Path(str(self.path) + ".tmp")
        with open(new_path, "w") as f:
            f.write(json.dumps(self.SCHEMA) + "\n")
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "compressed_summary",
                "content": summary_text,
                "tags": ["summary"],
                "importance": 0.9,
            }) + "\n")
            for entry in reversed(recent):
                f.write(json.dumps(entry) + "\n")
        new_path.replace(self.path)

    def _write_line(self, obj: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(obj) + "\n")


# Usage
memory = AgentMemory("agent_memory.jsonl")
memory.remember("User prefers concise responses", mem_type="preference",
                tags=["user", "style"], importance=0.8)
memory.remember("Completed refactor of auth module", mem_type="action",
                tags=["code", "auth"], importance=0.6)

recent = memory.recall(query_tags=["user"], min_importance=0.5)
```

---

## Context Compression

### Sliding Window with Summarization

```python
from anthropic import Anthropic

client = Anthropic()

class CompressedConversation:
    """
    Maintains a conversation within token budget using
    summary + sliding window pattern.
    """

    def __init__(self, max_messages: int = 20, summary_threshold: int = 15):
        self.messages: list[dict] = []
        self.summary: str = ""
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) >= self.summary_threshold:
            self._compress()

    def get_context(self) -> tuple[str, list[dict]]:
        """Returns (summary_for_system_prompt, recent_messages)."""
        return self.summary, self.messages

    def _compress(self):
        """Summarize oldest half, keep recent half."""
        mid = len(self.messages) // 2
        to_summarize = self.messages[:mid]
        self.messages = self.messages[mid:]

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarize
        )

        if self.summary:
            prompt = (
                f"Previous summary:\n{self.summary}\n\n"
                f"New conversation to incorporate:\n{conversation_text}\n\n"
                "Produce a concise updated summary preserving all decisions, "
                "facts, and context needed for future turns."
            )
        else:
            prompt = (
                f"Summarize this conversation, preserving all decisions, "
                f"facts, and context:\n{conversation_text}"
            )

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        self.summary = response.content[0].text


# Usage
conv = CompressedConversation(max_messages=20, summary_threshold=15)

def chat(user_input: str) -> str:
    conv.add("user", user_input)
    summary, messages = conv.get_context()

    system = "You are a helpful assistant."
    if summary:
        system += f"\n\nConversation history summary:\n{summary}"

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    reply = response.content[0].text
    conv.add("assistant", reply)
    return reply
```

---

## Tool Design

### Minimal, Composable Tools

```python
import anthropic
import json

client = anthropic.Anthropic()

# Good tool design: single responsibility, minimal parameters,
# returns structured output agents can parse

tools = [
    {
        "name": "search_codebase",
        "description": (
            "Search for files or symbols in the codebase. "
            "Returns file paths and relevant line numbers. "
            "Use before read_file to locate what you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name, function name, or keyword to search for"
                },
                "file_type": {
                    "type": "string",
                    "description": "Filter by extension, e.g. 'py', 'ts'. Omit for all types.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a specific file. Returns content with line numbers. "
            "Prefer reading only the section you need using start_line/end_line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
            },
            "required": ["path"],
        },
    },
]


def search_codebase(query: str, file_type: str = None) -> dict:
    """Stub — replace with actual grep/ripgrep implementation."""
    return {"matches": [], "query": query}


def read_file(path: str, start_line: int = None, end_line: int = None) -> dict:
    """Read file with optional line range."""
    try:
        with open(path) as f:
            lines = f.readlines()
        if start_line and end_line:
            lines = lines[start_line - 1:end_line]
        content = "".join(f"{i+1}: {l}" for i, l in enumerate(lines))
        return {"path": path, "content": content}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}


def run_agent_loop(task: str) -> str:
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(
                b.text for b in response.content if hasattr(b, "text")
            )

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search_codebase":
                    result = search_codebase(**block.input)
                elif block.name == "read_file":
                    result = read_file(**block.input)
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

---

## Evaluation: LLM-as-Judge

```python
from anthropic import Anthropic
from dataclasses import dataclass

client = Anthropic()

@dataclass
class EvalResult:
    score: float          # 0.0 - 1.0
    reasoning: str
    criteria_scores: dict[str, float]

JUDGE_SYSTEM = """
You are an expert evaluator. Score responses against criteria.
Always respond with valid JSON matching the schema provided.
Be calibrated: reserve 0.9+ for genuinely exceptional responses.
"""

def direct_score(
    response: str,
    criteria: dict[str, float],  # {criterion: weight}
    context: str = "",
) -> EvalResult:
    """
    Score a single response against weighted criteria.
    criteria: {"accuracy": 0.4, "clarity": 0.3, "completeness": 0.3}
    """
    criteria_list = "\n".join(
        f"- {name} (weight: {w:.0%})" for name, w in criteria.items()
    )
    schema = {
        "overall_score": "float 0-1",
        "reasoning": "string",
        "criteria_scores": {name: "float 0-1" for name in criteria},
    }

    prompt = f"""Evaluate this response:

CONTEXT: {context or 'None provided'}

RESPONSE TO EVALUATE:
{response}

CRITERIA:
{criteria_list}

Respond with JSON matching this schema:
{schema}"""

    result = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    data = json.loads(result.content[0].text)
    return EvalResult(
        score=data["overall_score"],
        reasoning=data["reasoning"],
        criteria_scores=data["criteria_scores"],
    )


def pairwise_compare(
    response_a: str,
    response_b: str,
    criteria: list[str],
    context: str = "",
    swap_and_average: bool = True,  # Mitigate position bias
) -> dict:
    """
    Compare two responses. swap_and_average=True runs both orderings
    and averages to mitigate position bias.
    """

    def _compare(ra, rb, label_a, label_b) -> dict:
        prompt = f"""Compare these two responses:

CONTEXT: {context or 'None'}

RESPONSE {label_a}:
{ra}

RESPONSE {label_b}:
{rb}

CRITERIA: {", ".join(criteria)}

Which response is better overall? Respond with JSON:
{{"winner": "{label_a} or {label_b} or tie", "confidence": "float 0-1", "reasoning": "string"}}"""

        result = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        return json.loads(result.content[0].text)

    result1 = _compare(response_a, response_b, "A", "B")

    if not swap_and_average:
        return result1

    result2 = _compare(response_b, response_a, "B", "A")
    # Normalize result2 winner labels back to original A/B
    if result2["winner"] == "B":
        result2["winner"] = "A"
    elif result2["winner"] == "A":
        result2["winner"] = "B"

    # Aggregate
    votes = [result1["winner"], result2["winner"]]
    if votes[0] == votes[1]:
        winner = votes[0]
    else:
        winner = "tie"

    return {
        "winner": winner,
        "confidence": (result1["confidence"] + result2["confidence"]) / 2,
        "reasoning": f"Run1: {result1['reasoning']} | Run2: {result2['reasoning']}",
    }
```

---

## BDI Mental States (Cognitive Architecture)

```python
from dataclasses import dataclass, field
from enum import Enum

class IntentionStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"

@dataclass
class Belief:
    """What the agent believes to be true about the world."""
    predicate: str          # e.g. "file_exists"
    subject: str            # e.g. "src/auth.py"
    value: object           # e.g. True
    confidence: float = 1.0
    source: str = "observation"

@dataclass
class Desire:
    """Goals the agent wants to achieve."""
    goal: str               # e.g. "refactor auth module"
    priority: float = 0.5   # 0.0 (low) to 1.0 (critical)
    preconditions: list[str] = field(default_factory=list)

@dataclass
class Intention:
    """Committed plans — desires the agent has decided to pursue."""
    desire: Desire
    plan: list[str]         # Ordered action steps
    status: IntentionStatus = IntentionStatus.PENDING
    current_step: int = 0

class BDIAgent:
    def __init__(self):
        self.beliefs: list[Belief] = []
        self.desires: list[Desire] = []
        self.intentions: list[Intention] = []

    def update_belief(self, predicate: str, subject: str, value: object,
                      confidence: float = 1.0, source: str = "observation"):
        # Remove outdated belief
        self.beliefs = [
            b for b in self.beliefs
            if not (b.predicate == predicate and b.subject == subject)
        ]
        self.beliefs.append(Belief(predicate, subject, value, confidence, source))

    def add_desire(self, goal: str, priority: float = 0.5,
                   preconditions: list[str] = None):
        self.desires.append(Desire(goal, priority, preconditions or []))
        self.desires.sort(key=lambda d: d.priority, reverse=True)

    def deliberate(self) -> Intention | None:
        """Select highest-priority desire whose preconditions are met."""
        active_intentions = {i.desire.goal for i in self.intentions
                             if i.status == IntentionStatus.ACTIVE}

        for desire in self.desires:
            if desire.goal in active_intentions:
                continue
            if self._preconditions_met(desire):
                return self._form_intention(desire)
        return None

    def _preconditions_met(self, desire: Desire) -> bool:
        for precondition in desire.preconditions:
            pred, subj, val = precondition.split(":", 2)
            matching = [
                b for b in self.beliefs
                if b.predicate == pred and b.subject == subj
            ]
            if not matching or str(matching[0].value) != val:
                return False
        return True

    def _form_intention(self, desire: Desire) -> Intention:
        # Simple plan generation — extend with actual planning logic
        plan = [f"execute: {desire.goal}"]
        intention = Intention(desire=desire, plan=plan,
                              status=IntentionStatus.ACTIVE)
        self.intentions.append(intention)
        return intention


# Usage
agent = BDIAgent()
agent.update_belief("file_exists", "src/auth.py", True)
agent.update_belief("tests_passing", "auth_suite", False)

agent.add_desire("fix failing tests", priority=0.9,
                 preconditions=["file_exists:src/auth.py:True"])
agent.add_desire("add documentation", priority=0.3)

intention = agent.deliberate()
print(f"Agent intends to: {intention.desire.goal}")
print(f"Plan: {intention.plan}")
```

---

## Filesystem Context Pattern

Use files to offload context that exceeds window limits:

```python
import json
from pathlib import Path
from datetime import datetime, timezone

class FilesystemContext:
    """
    Offload large context to files. Agents read only what they need.
    Implements the filesystem-context skill pattern.
    """

    def __init__(self, workspace: str = ".agent_workspace"):
        self.root = Path(workspace)
        self.root.mkdir(exist_ok=True)
        (self.root / "plans").mkdir(exist_ok=True)
        (self.root / "outputs").mkdir(exist_ok=True)
        (self.root / "scratch").mkdir(exist_ok=True)

    def save_plan(self, plan_id: str, steps: list[dict]) -> str:
        """Persist a multi-step plan. Returns file path for agent reference."""
        path = self.root / "plans" / f"{plan_id}.json"
        data = {
            "plan_id": plan_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "steps": steps,
            "status": "active",
        }
        path.write_text(json.dumps(data, indent=2))
        return str(path)

    def load_plan(self, plan_id: str) -> dict | None:
        path = self.root / "plans" / f"{plan_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_tool_output(self, tool_name: str, output: str) -> str:
        """
        Offload large tool outputs to files instead of keeping in context.
        Returns a short reference the agent includes in its context.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{tool_name}_{timestamp}.txt"
        path = self.root / "outputs" / filename
        path.write_text(output)
        # Return brief reference — agent includes this, not the full output
        return f"[Tool output saved: {path} ({len(output)} chars)]"

    def scratch(self, key: str, value: str = None) -> str | None:
        """Simple key-value scratch pad for agent working memory."""
        path = self.root / "scratch" / f"{key}.txt"
        if value is not None:
            path.write_text(value)
            return value
        return path.read_text() if path.exists() else None


# Usage
ctx = FilesystemContext()

# Save a large plan — reference path in context, not full content
plan_path = ctx.save_plan("refactor_auth", steps=[
    {"step": 1, "action": "read current auth.py", "status": "pending"},
    {"step": 2, "action": "identify security issues", "status": "pending"},
    {"step": 3, "action": "implement fixes", "status": "pending"},
    {"step": 4, "action": "run tests", "status": "pending"},
])

# Offload large output
big_output = "... " * 10000  # Large tool result
ref = ctx.save_tool_output("grep_search", big_output)
# Agent context contains only: "[Tool output saved: .agent_workspace/outputs/grep_search_.... (40000 chars)]"
```

---

## Common Patterns

### Skill Trigger Matching

```python
import re

SKILL_TRIGGERS = {
    "context-fundamentals": [
        "understand context", "explain context windows", "design agent architecture"
    ],
    "context-degradation": [
        "diagnose context problems", "fix lost-in-middle", "debug agent failures"
    ],
    "context-compression": [
        "compress context", "summarize conversation", "reduce token usage"
    ],
    "multi-agent-patterns": [
        "design multi-agent system", "implement supervisor pattern"
    ],
    "memory-systems": [
        "implement agent memory", "build knowledge graph", "track entities"
    ],
    "evaluation": [
        "evaluate agent performance", "build test framework", "measure quality"
    ],
    "advanced-evaluation": [
        "implement LLM-as-judge", "compare model outputs", "mitigate bias"
    ],
    "bdi-mental-states": [
        "model agent mental states", "implement BD
