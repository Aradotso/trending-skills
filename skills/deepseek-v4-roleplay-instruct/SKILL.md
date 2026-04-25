```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning in the <think> block.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion mode
  - switch deepseek think block style
  - deepseek roleplay control instructions
  - deepseek think tag inner os
  - deepseek v4 pure analysis mode
  - deepseek roleplay instruct marker
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct provides **special control markers** that steer how the DeepSeek-V4 model reasons inside its `<think>` block during roleplay sessions. By appending a short instruction to the **first user message**, you can reliably shift the model between:

| Mode | Think Block Behavior |
|------|----------------------|
| **Default** | Model chooses automatically based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses — like an actor in character |
| **Pure Analysis** (`no_inner_os`) | Cold, director-style logic — no in-character inner voice |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only
- API: `deepseek-v4-flash` and `deepseek-v4-pro`
- Web Quick Mode is **not** supported

> Triggering is probabilistic (~stable, not 100%). Re-roll if the first attempt doesn't match the expected format.

---

## Core Markers (copy-paste ready)

### Character Immersion Marker

```python
INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)
```

### Pure Analysis Marker

```python
NO_INNER_OS_MARKER = (
    "\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可\n"
    "2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代\n"
    "3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演"
)
```

---

## Installation / Setup

No package to install. The technique is a prompt-engineering convention built into DeepSeek-V4's training. Use it by:

1. Copying the marker string into your code / chat client.
2. Appending it to the **first user message only** (not system prompt, not later turns).

For API usage, set your key via environment variable:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

---

## API Usage (Python)

### Full Working Example

```python
import os
from openai import OpenAI  # DeepSeek uses an OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# ── Markers ──────────────────────────────────────────────────────────────────

INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)

NO_INNER_OS_MARKER = (
    "\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可\n"
    "2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代\n"
    "3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演"
)

# ── Helper ────────────────────────────────────────────────────────────────────

def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",          # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Construct the initial message list with the correct marker appended
    to the first user turn.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Multi-turn roleplay session ───────────────────────────────────────────────

SYSTEM = "你是一个傲娇的女高中生，表面冷漠，内心在意对方。"

# Round 1 — marker injected automatically
messages = build_messages(
    system_prompt=SYSTEM,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)
reply = chat(messages)
print("[Round 1]", reply)

# Round 2+ — just append normally; marker persists in history
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("[Round 2]", reply)

# Round 3
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("[Round 3]", reply)
```

---

## Common Patterns

### Pattern 1 — Reusable Session Class

```python
class DeepSeekRoleplaySession:
    """Stateful multi-turn roleplay session with think-mode control."""

    MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-flash",
    ):
        assert model in self.MODELS, f"model must be one of {self.MODELS}"
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self._messages: list[dict] = []
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            # Inject marker only on the first user turn
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        self._messages.append({"role": "user", "content": user_message})

        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *self._messages,
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
        )
        reply = response.choices[0].message.content
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self._messages = []
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一名冷静的侦探，正在审讯嫌疑人。",
    mode="no_inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「坐在审讯室对面」你知道昨晚十点你在哪里吗？"))
print(session.send("「把一张照片推到桌上」这个人你认识吗？"))
```

### Pattern 2 — Web / Chat Client Usage (plain text)

Paste this into the **first message** in DeepSeek Expert Mode:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages are plain — no extra instruction needed.

### Pattern 3 — Mode Selection at Runtime

```python
def start_session(system: str, mode: str) -> list[dict]:
    """Factory: pick mode at runtime (e.g. from user config or CLI arg)."""
    modes = {"immersive": "inner_os", "analytical": "no_inner_os", "auto": "default"}
    resolved = modes.get(mode, "default")
    return build_messages(system, "「开始角色扮演」", resolved)
```

---

## Configuration Reference

| Parameter | Values | Notes |
|-----------|--------|-------|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Injected on turn 1 only |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support the markers |
| Marker placement | End of **first user message** | System prompt placement is less reliable |
| Web mode | Expert Mode only | Quick Mode unsupported |

---

## Troubleshooting

### Marker didn't trigger — model used wrong think style
- **Re-roll**: The trigger is probabilistic. Send the same message again.
- **Verify placement**: The marker must be in the **first user message**, not the system prompt and not a later turn.
- **Check web mode**: Ensure you're using **Expert Mode**, not Quick Mode.

### Think block is empty or missing
- Some very short prompts suppress the think block. Add more scene detail to the first message.

### Mode bleeds across sessions
- Each conversation is independent. Start a new conversation (or call `session.reset()`) to change modes — do not try to override mid-conversation.

### API authentication errors
```bash
# Ensure the env var is exported before running your script
export DEEPSEEK_API_KEY=your_api_key_here
python your_script.py
```

### Want to verify the mode worked?
- Web: Click **"查看思考过程"** (View thinking process) after the first reply.
- API: Check the `reasoning_content` field if exposed, or log the raw `<think>...</think>` block in streaming responses.

---

## Key Rules (Summary)

1. **Inject once** — first user message only. The full conversation history carries it forward automatically.
2. **Don't put markers in system prompt** — training injection point is the user turn.
3. **Choose one mode per conversation** — switching mid-session is unreliable; open a new chat instead.
4. **Markers affect thinking, not final reply format** — the visible response style is influenced indirectly (immersive → more emotional; analytical → more structured).
```
