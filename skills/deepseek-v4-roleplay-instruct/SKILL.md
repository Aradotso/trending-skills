```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion prompt
  - switch deepseek thinking style
  - deepseek roleplay instruct marker
  - deepseek think tag control
  - deepseek v4 roleplay api
  - deepseek roleplay inner os
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control markers injected into the **first user message** to steer how the model reasons inside its `<think>` tags during roleplay. Two modes are available beyond the default:

- **Character Immersion** (`inner_os`): The model thinks in first-person inner monologue wrapped in parentheses — like an actor "in character."
- **Pure Analysis** (`no_inner_os`): The model thinks as a director — cold, structural, no inner-character performance.

These markers work on:
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs

They do **not** work in the web Quick Mode.

---

## How It Works

The model sees its full conversation history on every turn. By appending the marker to the **first user message only**, the instruction stays in context for the entire session automatically — no need to repeat it.

Markers are **probabilistic**, not guaranteed. If a mode doesn't activate on the first attempt, regenerate.

---

## The Two Markers (copy-ready)

### Character Immersion Marker

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python API Integration

### Setup

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```

### Define Markers

```python
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
```

### Message Builder

```python
def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    Build the initial message list with optional thinking-mode marker.

    Args:
        system_prompt: Character/scenario description for the system role.
        user_first_message: The player's opening action or dialogue.
        mode: One of "default", "inner_os", or "no_inner_os".

    Returns:
        List of message dicts ready for the DeepSeek chat API.
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
```

### Full Multi-Turn Session Example

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷漠，内心其实很在意对方。"

# --- Round 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",  # character immersion
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
reply = response.choices[0].message.content
print("Assistant:", reply)

# Append assistant reply to history
messages.append({"role": "assistant", "content": reply})

# --- Round 2+: no marker needed, history carries it ---
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
reply = response.choices[0].message.content
print("Assistant:", reply)

messages.append({"role": "assistant", "content": reply})

# --- Round 3 ---
messages.append({"role": "user", "content": "「我把一块巧克力推到她桌上」"给你。""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
print("Assistant:", response.choices[0].message.content)
```

### Streaming Version

```python
def stream_roleplay_turn(messages: list[dict], model: str = "deepseek-v4-pro") -> str:
    """Stream a single roleplay turn and return the full text."""
    full_response = ""
    with client.chat.completions.stream(
        model=model,
        messages=messages,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response += delta
    print()  # newline after stream
    return full_response
```

### Session Manager Class

```python
class DeepSeekRoleplaySession:
    """Manages a multi-turn DeepSeek roleplay session with thinking-mode control."""

    MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-pro",
    ):
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}")
        if mode not in ("default", "inner_os", "no_inner_os"):
            raise ValueError("mode must be 'default', 'inner_os', or 'no_inner_os'")

        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn_done = False
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def send(self, user_message: str) -> str:
        """Send a user message and return the assistant reply."""
        # Inject marker only on the very first user turn
        if not self._first_turn_done:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn_done = True

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt: str | None = None) -> None:
        """Start a fresh session, optionally with a new system prompt."""
        sp = system_prompt or self.messages[0]["content"]
        self.messages = [{"role": "system", "content": sp}]
        self._first_turn_done = False


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个在咖啡店工作的内敛青年，暗恋常客已久。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「我推开咖啡店的门」"你好，请问还有位置吗？""))
print(session.send("「我坐到窗边的位置」"来一杯美式。""))
print(session.send("「我注意到你手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / App Usage

In the DeepSeek web **Expert Mode** or official APP, paste the marker at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no marker — type naturally.

To verify: click **"查看思考过程"** (View Thinking Process) after a reply.

---

## Mode Comparison

| | Default | `inner_os` (Character Immersion) | `no_inner_os` (Pure Analysis) |
|---|---|---|---|
| **Think style** | Auto-selected | First-person inner monologue in `（）` | Direct analytical statements |
| **Best for** | General use | Emotional depth, immersive RP | Consistent structure, complex plots |
| **Reply impact** | Neutral | More emotionally authentic | More structurally stable |
| **Marker position** | None | End of first user message | End of first user message |

---

## Configuration Reference

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key (env var) |
| `model` | `"deepseek-v4-pro"` or `"deepseek-v4-flash"` |
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` |

---

## Common Patterns

### Switch mode mid-story
Start a **new conversation/session** and inject the new marker on the first message. Do not try to inject into an ongoing session.

### Marker not activating
Regenerate the response. These markers increase probability but are not deterministic. Two or three attempts usually succeed.

### System prompt vs. user message injection
Always inject into the **first user message**, not the system prompt. The model was trained to respond to the marker in that position.

### Using with async Python

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_send(messages: list[dict], model: str = "deepseek-v4-pro") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        "你是一个侦探。",
        "「一具尸体出现在图书馆」"你有什么发现？"",
        mode="no_inner_os",
    )
    reply = await async_send(messages)
    print(reply)

asyncio.run(main())
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Thinking mode unchanged after adding marker | Quick Mode on web, or marker not on first message | Switch to Expert Mode; ensure marker is in turn 1 |
| `AuthenticationError` | Missing or wrong API key | Set `DEEPSEEK_API_KEY` env var correctly |
| `model not found` | Wrong model string | Use exactly `"deepseek-v4-pro"` or `"deepseek-v4-flash"` |
| Marker activates then fades after many turns | Context window pressure | Keep sessions focused; reset if drift appears |
| No `<think>` block visible | API doesn't expose reasoning tokens by default | Check if your API tier includes reasoning output; use web Expert Mode to see it |
```
