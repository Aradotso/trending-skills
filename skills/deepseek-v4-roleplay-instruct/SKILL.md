```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4's thinking chain style during roleplay with special marker instructions for immersive character inner monologue or pure analytical planning modes.
triggers:
  - "add deepseek roleplay thinking mode"
  - "switch deepseek inner monologue mode"
  - "use deepseek character immersion instruct"
  - "set deepseek v4 roleplay thinking style"
  - "inject deepseek roleplay marker"
  - "configure deepseek think tag behavior"
  - "enable deepseek pure analysis mode"
  - "deepseek roleplay control instruction"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes special control markers you can inject into the **first user message** of a conversation to steer how the model reasons inside its `<think>` tags during roleplay. There are two modes beyond the default:

| Mode | Effect on `<think>` content |
|---|---|
| **Default** | Model auto-selects based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses `（心想：…）` |
| **Pure Analysis** (`no_inner_os`) | Cold director-style planning, no inner monologue |

**Supported surfaces:**
- DeepSeek official APP / web **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- *(Web Quick Mode is NOT supported)*

**Key constraint:** Markers go in the **first user message only** — they remain in conversation history and auto-apply to all subsequent turns.

---

## Installation / Setup

No package installation required. This is a prompting technique. For API usage, copy the marker strings into your application code.

### Environment Variables

```bash
# Your DeepSeek API key
export DEEPSEEK_API_KEY="your_api_key_here"
export DEEPSEEK_API_BASE="https://api.deepseek.com"  # or your endpoint
```

---

## The Marker Strings

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

## Core API Usage Pattern

### Full Working Example (Python)

```python
import os
from openai import OpenAI  # DeepSeek is OpenAI-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
)

# --- Marker definitions ---
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

MARKERS = {
    "inner_os": INNER_OS_MARKER,
    "no_inner_os": NO_INNER_OS_MARKER,
    "default": "",
}


def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    Build the initial messages list with the appropriate marker appended
    to the first user message.

    Args:
        system_prompt: The character/scenario system prompt.
        user_first_message: The first user turn content.
        mode: One of "inner_os", "no_inner_os", or "default".

    Returns:
        List of message dicts ready to send to the API.
    """
    marker = MARKERS.get(mode, "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message + marker},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    """Send messages and return the assistant reply text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# --- Example: multi-turn roleplay with character immersion mode ---

system_prompt = "你是一个傲娇的女高中生，表面上对人冷漠，内心其实很在意别人的感受。"

# Turn 1 — marker is injected here automatically
messages = build_messages(
    system_prompt=system_prompt,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)
reply = chat(messages)
print("Turn 1:", reply)

# Turn 2 — no special handling needed; marker stays in history
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Turn 2:", reply)

# Turn 3
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Turn 3:", reply)
```

---

## Common Patterns

### Pattern 1: Conversation Manager Class

```python
class DeepSeekRoleplay:
    """Manages a stateful DeepSeek roleplay conversation."""

    MARKERS = {
        "inner_os": (
            "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
            "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
            "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
        ),
        "no_inner_os": (
            "\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可\n"
            "2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代\n"
            "3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演"
        ),
        "default": "",
    }

    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        )

    def send(self, user_message: str) -> str:
        content = user_message
        if self._first_turn:
            content += self.MARKERS.get(self.mode, "")
            self._first_turn = False

        self.messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt: str | None = None, mode: str | None = None):
        """Start a fresh conversation, optionally changing system prompt or mode."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        if mode:
            self.mode = mode
        self._first_turn = True


# Usage
rp = DeepSeekRoleplay(
    system_prompt="你是咖啡店里沉默寡言的男店员，暗恋常客。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(rp.send("「我推开咖啡店的门」"你好，请问还有位置吗？""))
print(rp.send("「我坐到窗边」"来一杯美式。""))

# Switch to pure analysis mode for a new scenario
rp.reset(system_prompt="你是一个冷静的侦探，正在审讯嫌疑人。", mode="no_inner_os")
print(rp.send("「我把一张照片推到桌上」"你认识这个人吗？""))
```

### Pattern 2: Web Prompt Template (Copy-Paste Format)

For use in DeepSeek web Expert Mode, structure your first message like:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no special formatting.

### Pattern 3: Async API Usage

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
)

INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)

async def roleplay_turn(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    system = "你是一个温柔的图书管理员，喜欢推荐冷门好书。"
    first_user = "「我走进图书馆，四处张望」"请问有什么好看的书推荐吗？"" + INNER_OS_MARKER

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_user},
    ]

    reply = await roleplay_turn(messages)
    print(reply)

asyncio.run(main())
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"`, `"no_inner_os"`, `"default"` | Determines which marker (if any) is appended |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Both support marker injection |
| Marker position | First user message **only** | System prompt placement is less stable |
| Marker placement | Append at **end** of message | Separate from main content with a blank line |

---

## Troubleshooting

### Marker not taking effect

- **Cause:** Markers are probabilistic, not guaranteed 100%.
- **Fix:** Retry the generation (re-roll). The marker increases probability but doesn't guarantee the format on every single generation.

### Mode applied to wrong surface

- **Cause:** Web Quick Mode is not supported.
- **Fix:** Switch to **Expert Mode** on the DeepSeek web interface, or use the API directly.

### Marker placed in system prompt doesn't work reliably

- **Cause:** The model was trained with markers in the user message position.
- **Fix:** Always inject the marker into the **first user message**, not `system`.

### Mode bleeds into a new conversation

- **Cause:** The marker is still in conversation history.
- **Fix:** Start a new conversation/session. Markers only persist within a single conversation's context window.

### Want to switch modes mid-story

- **Cause:** Markers only take effect when present in the first turn.
- **Fix:** Start a new conversation with the desired marker in the first message. There is no mid-conversation mode switch mechanism.

### Thinking content not visible

- **Cause:** You need to click "查看思考过程" (View thinking process) in the web UI, or parse the `<think>...</think>` block from the API response.
- **Fix (API):**
  ```python
  raw = response.choices[0].message.content
  import re
  think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
  think_content = think_match.group(1).strip() if think_match else ""
  reply_content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
  ```
```
