```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - control deepseek think tag
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - switch deepseek reasoning style
  - deepseek roleplay instruct marker
  - deepseek think tag roleplay
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides **special control markers** you append to the first user message in a conversation to steer how the model's `<think>` block behaves during roleplay. Two modes are available:

- **角色沉浸 (Character Immersion)** — the model thinks in first-person inner monologue inside parentheses, like an actor in character.
- **纯分析 (Pure Analysis)** — the model thinks in cold, structured director-style analysis with no inner monologue.

These markers work with:
- DeepSeek official APP / web **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs

> **Note:** Triggers are probabilistic, not 100% guaranteed. Re-roll if not triggered on first attempt.

---

## How It Works

The model sees the full conversation history on every turn. By injecting a marker into the **first user message only**, the instruction stays in context for all subsequent turns automatically. No need to repeat it.

---

## The Two Markers

### Character Immersion Marker (`INNER_OS_MARKER`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Think block looks like:**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

---

### Pure Analysis Marker (`NO_INNER_OS_MARKER`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

**Think block looks like:**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Python Integration

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
    mode: str = "default"
) -> list[dict]:
    """
    Args:
        system_prompt: Character/scene description for the system role.
        user_first_message: The opening roleplay message from the user.
        mode: One of "default", "inner_os", or "no_inner_os".
    Returns:
        List of messages ready to send to the DeepSeek API.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" passes through unchanged

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Example

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷漠，内心温柔，对主角有好感但死要面子。"

def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",   # or "deepseek-v4-flash"
        messages=messages,
    )
    return response.choices[0].message.content

# --- Turn 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",   # character immersion
)
reply = chat(messages)
print(reply)

# --- Turn 2+: append normally, no marker needed ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print(reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print(reply)
```

### Switching Modes Per Session

```python
def new_session(system_prompt: str, opening_message: str, mode: str):
    """Start a fresh session with a given mode."""
    messages = build_messages(system_prompt, opening_message, mode)
    reply = chat(messages)
    messages.append({"role": "assistant", "content": reply})
    return messages

# Immersive session
immersive = new_session(SYSTEM_PROMPT, "「推开门」"我来了。"", mode="inner_os")

# Analytical session (same scene, different thinking style)
analytical = new_session(SYSTEM_PROMPT, "「推开门」"我来了。"", mode="no_inner_os")
```

---

## Web / Chat UI Usage

Paste marker at the end of your **first message only**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages require no special formatting:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Click **"查看思考过程"** (View Thinking Process) to verify the mode is active.

---

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mode` | `"default"` | No marker appended; model decides |
| `mode` | `"inner_os"` | Appends `INNER_OS_MARKER` to first user message |
| `mode` | `"no_inner_os"` | Appends `NO_INNER_OS_MARKER` to first user message |
| Supported models | `deepseek-v4-flash`, `deepseek-v4-pro` | API usage |
| Supported UI | DeepSeek APP / web **Expert Mode** only | Quick Mode not supported |
| Placement | End of **first** user message | System prompt placement is less effective |

---

## Common Patterns

### Async Client

```python
import asyncio
import os
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages: list[dict]) -> str:
    response = await async_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        "你是一个神秘的图书馆馆长。",
        "「我走进昏暗的图书馆」"请问……这里有禁书区吗？"",
        mode="inner_os",
    )
    reply = await async_chat(messages)
    print(reply)

asyncio.run(main())
```

### Streaming

```python
def stream_chat(messages: list[dict]):
    stream = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()

messages = build_messages(
    "你是一个冷酷的雇佣兵。",
    "「我把信封推到桌子对面」"任务很简单。"",
    mode="no_inner_os",
)
stream_chat(messages)
```

### Session Manager Class

```python
class RoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "default", model: str = "deepseek-v4-pro"):
        self.system_prompt = system_prompt
        self.mode = mode
        self.model = model
        self.messages: list[dict] = []
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            user_message_with_marker = user_message
            if self.mode == "inner_os":
                user_message_with_marker += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message_with_marker += NO_INNER_OS_MARKER

            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message_with_marker},
            ]
            self._first_turn = False
        else:
            self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

# Usage
session = RoleplaySession(
    system_prompt="你是一个落魄的前特工，现在在小城市经营一家修表店。",
    mode="inner_os",
)
print(session.send("「我推开店门，听到叮铃声」"修表吗？""))
print(session.send("「我把怀表放到柜台上」"这块表……很重要。""))
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Marker has no effect | Re-roll (probabilistic); ensure you're using Expert Mode on web or API, not Quick Mode |
| Think tag shows wrong style | Check marker was appended to first **user** message, not system prompt |
| Want to change mode mid-conversation | Start a **new conversation/session**; markers only work from the first turn |
| API returns no `<think>` content | Confirm model is `deepseek-v4-flash` or `deepseek-v4-pro`; other models may not expose reasoning |
| Marker appears in final reply | This is a rare edge case; try reducing whitespace between message and marker |

---

## Quick Reference

```python
# Environment setup
# export DEEPSEEK_API_KEY=your_key_here

from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

# One-liner session start
messages = build_messages("system prompt here", "first user message", mode="inner_os")
reply = client.chat.completions.create(model="deepseek-v4-pro", messages=messages).choices[0].message.content
```
```
