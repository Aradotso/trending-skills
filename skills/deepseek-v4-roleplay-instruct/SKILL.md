```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking chain style during roleplay via special inline instructions for immersive character mode or pure analysis mode.
triggers:
  - "add deepseek roleplay instructions"
  - "switch deepseek thinking mode"
  - "deepseek inner monologue roleplay"
  - "deepseek v4 character immersion mode"
  - "control deepseek thinking chain style"
  - "deepseek roleplay analysis mode"
  - "inject deepseek roleplay marker"
  - "deepseek v4 roleplay api setup"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 roleplay instruct provides **special control instructions** (markers) that you append to the first user message in a conversation. These markers influence how DeepSeek-V4 structures its internal `<think>` reasoning chain during roleplay:

- **Character Immersion Mode** (`inner_os`): The model thinks in first-person, using parenthesized inner monologue like `（心想：……）` — like an actor staying in character.
- **Pure Analysis Mode** (`no_inner_os`): The model thinks analytically without inner-character voice — like a director planning the scene.
- **Default Mode**: No marker appended; model auto-selects based on scene complexity.

**Supported targets:**
- DeepSeek official APP / web (Expert Mode only)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> ⚠️ Web "Quick Mode" does NOT support these instructions.

---

## Key Concepts

| Mode | Trigger | Think-tag behavior |
|---|---|---|
| Default | No marker | Auto-selected by model |
| Character Immersion | Append `INNER_OS_MARKER` to first user message | First-person inner monologue in `<think>` |
| Pure Analysis | Append `NO_INNER_OS_MARKER` to first user message | Logical planning only, no inner-character voice |

**Why first user message?** The markers were injected at this position during training, so this placement is the most stable and reliable trigger point.

---

## Marker Text (Copy-Ready)

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

## Python API Usage

### Installation & Setup

```bash
pip install openai  # DeepSeek uses OpenAI-compatible API
```

Set your API key:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

### Core Helper Module

```python
# deepseek_roleplay.py
from openai import OpenAI

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


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate marker injected.

    Args:
        system_prompt: Character/scene setup for the system role.
        user_first_message: The first user turn content.
        mode: One of "inner_os", "no_inner_os", or "default".

    Returns:
        List of message dicts ready for the DeepSeek API.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Conversation Example

```python
import os
from openai import OpenAI
from deepseek_roleplay import build_messages

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

system_prompt = "你是一个傲娇的女高中生，暗恋着班级里的转学生，但总是嘴硬不承认。"

# --- Round 1: inject marker once ---
messages = build_messages(
    system_prompt=system_prompt,
    user_first_message="「我走进教室，看到你坐在窗边」"早上好。"",
    mode="inner_os",  # Character Immersion
)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply = response.choices[0].message.content
print("Turn 1:", reply)

# --- Round 2+: just append normally, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply2 = response.choices[0].message.content
print("Turn 2:", reply2)

# Continue appending for subsequent turns
messages.append({"role": "assistant", "content": reply2})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
print("Turn 3:", response.choices[0].message.content)
```

### Switching Modes Between Conversations

```python
def start_new_session(system_prompt: str, first_user_msg: str, mode: str):
    """Start a fresh conversation with the chosen thinking mode."""
    messages = build_messages(system_prompt, first_user_msg, mode=mode)
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return messages, reply

# Immersive roleplay session
msgs, reply = start_new_session(
    system_prompt="你是一个神秘的吸血鬼伯爵，举止优雅却暗藏危机。",
    first_user_msg="「我推开古堡的大门」"有人在吗？"",
    mode="inner_os",
)

# Director/analytical session (e.g. for quality testing)
msgs, reply = start_new_session(
    system_prompt="你是一个神秘的吸血鬼伯爵，举止优雅却暗藏危机。",
    first_user_msg="「我推开古堡的大门」"有人在吗？"",
    mode="no_inner_os",
)
```

### Streaming Response Example

```python
import os
from openai import OpenAI
from deepseek_roleplay import build_messages

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    system_prompt="你是一个孤独的星际旅行者，漂泊在宇宙深处。",
    user_first_message="「信号灯突然亮起」"有人在那里吗？"",
    mode="inner_os",
)

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

---

## Web Usage (No Code)

1. Open DeepSeek APP or web in **Expert Mode** (专家模式).
2. In your **first message**, write your roleplay opening, add a blank line, then paste the marker:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

3. All subsequent messages are sent **without** any marker — the history carries it automatically.
4. Click **"查看思考过程"** (View Thinking Process) to verify the mode is active.

---

## Common Patterns

### Pattern: Reusable Session Class

```python
import os
from openai import OpenAI
from deepseek_roleplay import build_messages, INNER_OS_MARKER, NO_INNER_OS_MARKER

class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-pro"):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.system_prompt = system_prompt
        self.mode = mode
        self.messages = []
        self._initialized = False

    def send(self, user_message: str) -> str:
        if not self._initialized:
            self.messages = build_messages(self.system_prompt, user_message, mode=self.mode)
            self._initialized = True
        else:
            self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, mode: str = None):
        """Start a fresh conversation, optionally changing mode."""
        self.messages = []
        self._initialized = False
        if mode:
            self.mode = mode


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的咖啡师，暗恋常客但总是嘴硬。",
    mode="inner_os",
)

print(session.send("「我走进咖啡店」"老规矩。""))
print(session.send("「我注意到她多给了我一块糖」"这是……？""))

# Switch to analysis mode in new session
session.reset(mode="no_inner_os")
print(session.send("「我走进咖啡店」"老规矩。""))
```

### Pattern: Async API Calls

```python
import os
import asyncio
from openai import AsyncOpenAI
from deepseek_roleplay import build_messages

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_turn(messages: list) -> str:
    response = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        system_prompt="你是一只会说话的猫，高冷但偶尔撒娇。",
        user_first_message="「我打开家门」"我回来了！"",
        mode="inner_os",
    )
    reply = await roleplay_turn(messages)
    print(reply)

asyncio.run(main())
```

---

## Troubleshooting

### Marker didn't trigger / mode not active

- **Verify Expert Mode**: Web Quick Mode does not support these instructions.
- **Re-roll**: Triggering is probabilistic (~not 100%). If the first response doesn't show the expected think-style, regenerate the response.
- **Placement**: Marker must be in the **first user message**, not the system prompt and not a later turn.
- **Model**: Only `deepseek-v4-flash` and `deepseek-v4-pro` support this. Confirm your `model` string is correct.

### Think block not visible

- In the web UI, click **"查看思考过程"** to expand the reasoning chain.
- In the API, check `response.choices[0].message` — some clients may hide the think block depending on API version. Confirm the full raw response is being parsed.

### Marker injected on wrong turn

```python
# ❌ Wrong — marker in second turn, won't reliably work
messages = build_messages(system_prompt, "First message", mode="default")
# ... then append marker to a later user turn

# ✅ Correct — marker in first user turn only
messages = build_messages(system_prompt, "First message", mode="inner_os")
```

### System prompt placement

```python
# ❌ Avoid — placing marker in system prompt
system = "你是一个傲娇角色。" + INNER_OS_MARKER  # Not the trained injection point

# ✅ Correct — marker appended to first user message
messages = build_messages(system_prompt="你是一个傲娇角色。", user_first_message="...", mode="inner_os")
```

### Switching modes mid-conversation

Modes cannot be switched mid-conversation. Start a **new conversation** (new `messages` list) with the desired marker.

---

## Quick Reference

```python
# Minimal working example
import os
from openai import OpenAI
from deepseek_roleplay import build_messages

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

# mode options: "inner_os" | "no_inner_os" | "default"
messages = build_messages("你是角色A...", "第一条用户消息", mode="inner_os")
r = client.chat.completions.create(model="deepseek-v4-pro", messages=messages)
print(r.choices[0].message.content)
```
```
