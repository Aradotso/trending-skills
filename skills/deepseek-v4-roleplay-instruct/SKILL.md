```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning within <think> tags
triggers:
  - "deepseek roleplay thinking mode"
  - "deepseek inner monologue roleplay"
  - "switch deepseek think style"
  - "deepseek character immersion mode"
  - "deepseek v4 roleplay instructions"
  - "control deepseek thinking process"
  - "deepseek pure analysis mode"
  - "deepseek roleplay prompt engineering"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions that influence **how the model thinks** inside its `<think>` tags during roleplay. You can switch between two modes:

- **角色沉浸 (Character Immersion)** — thinking reads like an actor's inner monologue (first-person, bracketed thoughts)
- **纯分析 (Pure Analysis)** — thinking reads like a director's cold, structured planning (no inner monologue)

These instructions are appended to the **first user message** of a conversation and persist automatically throughout the entire chat via context history.

---

## Supported Models & Surfaces

| Surface | Supported |
|---|---|
| DeepSeek official APP — Expert Mode | ✅ |
| DeepSeek web — Expert Mode | ✅ |
| `deepseek-v4-flash` API | ✅ |
| `deepseek-v4-pro` API | ✅ |
| DeepSeek web — Quick Mode | ❌ |

> **Note**: Triggering is probabilistic (~stable but not 100%). If the mode doesn't activate, retry the first message.

---

## The Two Control Instructions

### Character Immersion Mode (`inner_os`)

Appended to first user message to make `<think>` content feel like first-person character thoughts:

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Result in `<think>`:**
```
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
```

---

### Pure Analysis Mode (`no_inner_os`)

Appended to first user message to keep `<think>` as structured, director-style analysis:

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

**Result in `<think>`:**
```
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
```

---

## Python API Usage

### Constants

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
def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking mode marker
    injected into the first user message.
    
    Args:
        system_prompt: The character/scenario system prompt
        user_first_message: The first user turn content
        mode: Thinking style — "inner_os" for character immersion,
              "no_inner_os" for pure analysis, "default" for automatic
    
    Returns:
        List of message dicts ready to pass to the API
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # mode == "default" → no marker appended

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Conversation Example

```python
import os
from openai import OpenAI  # DeepSeek is OpenAI-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

# --- Round 1: inject marker into first user message ---
messages = build_messages(
    system_prompt="你是一个傲娇的女高中生，表面冷漠，内心其实很在乎对方。",
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",  # Character immersion
)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print(assistant_reply)

# --- Append assistant reply ---
messages.append({"role": "assistant", "content": assistant_reply})

# --- Round 2+: NO marker needed — it's already in history ---
messages.append({
    "role": "user",
    "content": "「我在她旁边坐下」"今天心情不好吗？""
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
print(response.choices[0].message.content)
```

### Streaming Version

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    system_prompt="你是一个神秘的图书馆管理员，说话总是意味深长。",
    user_first_message="「我推开图书馆的门，四周安静得出奇」"请问……有人在吗？"",
    mode="no_inner_os",  # Pure analysis mode
)

with client.chat.completions.stream(
    model="deepseek-v4-flash",
    messages=messages,
) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
```

### Async Version

```python
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_session(system_prompt: str, opening: str, mode: str = "inner_os"):
    messages = build_messages(system_prompt, opening, mode=mode)
    
    response = await client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return messages, reply

async def main():
    messages, reply = await roleplay_session(
        system_prompt="你是一位温柔的咖啡师，暗恋着常客。",
        opening="「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"",
        mode="inner_os",
    )
    print(reply)

asyncio.run(main())
```

---

## Web / Chat UI Usage

Paste the marker at the **end** of your first message, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages need **no special treatment**:

```
Round 2: 「我坐到窗边的位置」"来一杯美式。"
Round 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

To verify the mode activated: click **"查看思考过程"** (View thinking process) on any reply.

---

## Session Manager Class

```python
import os
from openai import OpenAI
from typing import Literal

ThinkingMode = Literal["default", "inner_os", "no_inner_os"]

class DeepSeekRoleplaySession:
    """
    Manages a multi-turn DeepSeek roleplay conversation with
    automatic thinking-mode injection on the first turn.
    """

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

    def __init__(
        self,
        system_prompt: str,
        mode: ThinkingMode = "inner_os",
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True

        self.client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
        )

    def _inject_marker(self, content: str) -> str:
        if self.mode == "inner_os":
            return content + self.INNER_OS_MARKER
        elif self.mode == "no_inner_os":
            return content + self.NO_INNER_OS_MARKER
        return content

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant's reply."""
        if self._first_turn:
            user_message = self._inject_marker(user_message)
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt: str | None = None):
        """Reset the conversation, optionally changing the system prompt."""
        sp = system_prompt or self.messages[0]["content"]
        self.messages = [{"role": "system", "content": sp}]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的咖啡师，对常客有特殊感情但从不承认。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.chat("「我推开咖啡店的门」"老地方，美式。""))
print(session.chat("「我看着她的背影」"今天好像有点不一样……""))
print(session.chat("「我轻声问」"你还好吗？""))
```

---

## Common Patterns

### Pattern 1: Switch Modes Between Conversations

```python
# Immersive emotional scene
emotional_session = DeepSeekRoleplaySession(
    system_prompt="你是一个暗恋主角的室友。",
    mode="inner_os",
)

# Structured narrative planning
planning_session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘侦探，思维缜密。",
    mode="no_inner_os",
)
```

### Pattern 2: Retry on Mode Failure

```python
import re

def chat_with_retry(session: DeepSeekRoleplaySession, message: str, max_retries: int = 3) -> str:
    """
    Retry if the expected thinking mode markers don't appear.
    Only retries on the first turn where mode injection matters most.
    """
    for attempt in range(max_retries):
        # Reset and retry from scratch for first-turn failures
        if attempt > 0:
            system_prompt = session.messages[0]["content"]
            session.reset(system_prompt)

        reply = session.chat(message)

        # Verify inner_os mode activated by checking for bracketed thoughts
        # (This requires access to the raw response with <think> content)
        return reply  # Return on success

    return reply  # Return last attempt regardless
```

### Pattern 3: Expose Thinking Content

```python
def chat_with_thinking(session: DeepSeekRoleplaySession, user_message: str):
    """Return both thinking process and final reply separately."""
    if session._first_turn:
        user_message = session._inject_marker(user_message)
        session._first_turn = False

    session.messages.append({"role": "user", "content": user_message})

    response = session.client.chat.completions.create(
        model=session.model,
        messages=session.messages,
    )

    choice = response.choices[0]
    reply = choice.message.content

    # Some API versions expose reasoning_content separately
    thinking = getattr(choice.message, "reasoning_content", None)

    session.messages.append({"role": "assistant", "content": reply})

    return {"thinking": thinking, "reply": reply}
```

---

## Configuration Reference

| Parameter | Values | Effect |
|---|---|---|
| `mode` | `"inner_os"` | First-person bracketed thoughts in `<think>` |
| `mode` | `"no_inner_os"` | Pure analytical planning in `<think>` |
| `mode` | `"default"` | Model chooses automatically based on scene complexity |
| `model` | `"deepseek-v4-pro"` | Higher quality, slower |
| `model` | `"deepseek-v4-flash"` | Faster, lighter |
| Marker position | First user message only | Persists via context window throughout session |

---

## Troubleshooting

### Mode didn't activate
- The instruction is probabilistic. **Retry** the first message 2–3 times.
- Ensure the marker is appended to the **first user message**, not the system prompt.
- Verify you're using **Expert Mode** on web, not Quick Mode.

### Marker in system prompt not working
The model was trained to respond to this marker in the **user turn**, not the system prompt. Always inject into `user_first_message`.

```python
# ❌ Wrong — system prompt injection
{"role": "system", "content": f"{system_prompt}\n{INNER_OS_MARKER}"}

# ✅ Correct — first user message injection  
{"role": "user", "content": f"{first_message}{INNER_OS_MARKER}"}
```

### Mode stops working mid-conversation
The marker in message history should persist automatically. If it breaks:
1. Check that assistant replies are being appended to `messages` correctly
2. Confirm the first user message (with marker) is still present in the history
3. Context truncation on very long conversations can drop early messages — re-inject if needed

### Chinese characters rendering issues
Ensure your Python source file or environment uses **UTF-8 encoding**:

```python
# At top of file (Python 2 compatibility, good practice)
# -*- coding: utf-8 -*-

# Or set environment variable
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
```
```
