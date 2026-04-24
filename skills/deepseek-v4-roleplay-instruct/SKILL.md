```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning in the <think> block.
triggers:
  - "deepseek roleplay thinking mode"
  - "deepseek inner monologue roleplay"
  - "switch deepseek think block style"
  - "deepseek v4 character immersion mode"
  - "deepseek pure analysis mode roleplay"
  - "deepseek roleplay control instructions"
  - "deepseek think tag roleplay prompt"
  - "deepseek v4 flash pro roleplay api"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 (including `deepseek-v4-flash` and `deepseek-v4-pro`) supports a reasoning/thinking mode that exposes a `<think>` block before the final response. During roleplay, this thinking block can behave in two distinct ways — controlled by special marker instructions injected into the **first user message**:

| Mode | Think Block Behavior |
|------|----------------------|
| **Default** | Model auto-selects based on scene complexity |
| **角色沉浸 (Character Immersion)** | First-person inner monologue wrapped in parentheses |
| **纯分析 (Pure Analysis)** | Cold, director-style logical planning — no inner voice |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only (not Quick Mode)
- API: `deepseek-v4-flash`, `deepseek-v4-pro`

---

## Installation / Setup

No package to install. This is a prompting technique — copy the marker strings and inject them into your first user message.

For API usage, install the DeepSeek SDK or use `openai` compatible client:

```bash
pip install openai
```

Set your API key:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

---

## The Two Control Markers

### Character Immersion Marker (角色沉浸)

Inject this at the end of the first user message to make the model think *as* the character:

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

### Pure Analysis Marker (纯分析)

Inject this to make the model think like a writer/director — no character voice:

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

### Full Working Example

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# --- Marker constants ---
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


def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    mode options:
        "default"     — no marker injected
        "inner_os"    — character immersion (角色沉浸)
        "no_inner_os" — pure analysis (纯分析)
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# --- Multi-turn roleplay session ---

system_prompt = "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。"
first_user_msg = "「我走进教室」"早上好。""

# Round 1: inject marker once
messages = build_messages(system_prompt, first_user_msg, mode="inner_os")
reply = chat(messages)
print("Round 1:", reply)

# Round 2+: just append normally — marker stays in history, auto-applies
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Round 2:", reply)

# Round 3
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Round 3:", reply)
```

### Switching Models

```python
# Flash model (faster, lighter)
reply = chat(messages, model="deepseek-v4-flash")

# Pro model (more capable)
reply = chat(messages, model="deepseek-v4-pro")
```

---

## Web / App Usage

Paste the marker at the end of your **first message only**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are sent normally — no need to repeat the marker.

---

## Common Patterns

### Pattern: Reusable Session Class

```python
class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-flash"):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的图书管理员，知晓所有禁书的秘密。",
    mode="inner_os",
)

print(session.send("「我悄悄走进图书馆的地下室」"这里……有什么？""))
print(session.send("「我拿起一本封面破损的古书」"这本书……可以借吗？""))
```

### Pattern: Async API Calls

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        "你是一个冷静的侦探。",
        "「凌晨三点，我推开了那扇门」"现场有什么发现？"",
        mode="no_inner_os",
    )
    reply = await async_chat(messages)
    print(reply)

asyncio.run(main())
```

### Pattern: Streaming Response

```python
def chat_stream(messages: list[dict], model: str = "deepseek-v4-flash"):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()  # newline at end

messages = build_messages(
    "你是一个话很少的剑客。",
    "「我挡住了他的去路」"你要去哪里？"",
    mode="inner_os",
)
chat_stream(messages)
```

---

## Configuration Reference

| Parameter | Values | Notes |
|-----------|--------|-------|
| `mode` | `"default"`, `"inner_os"`, `"no_inner_os"` | Only affects `<think>` block content |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Flash = faster; Pro = higher quality |
| Marker injection point | First user message only | Training-aligned position; system prompt placement is less reliable |
| Trigger probability | Not 100% | Re-roll if mode doesn't activate on first attempt |

---

## Troubleshooting

**Mode didn't activate (think block looks wrong):**
- Retry the request — activation is probabilistic, not guaranteed
- Ensure the marker is appended to the **first user message**, not the system prompt
- Check you're using Expert Mode on web (Quick Mode is unsupported)
- Verify model is `deepseek-v4-flash` or `deepseek-v4-pro`

**Want to change mode mid-conversation:**
- Start a new conversation/session — inject the new marker in the first message of the fresh context

**Marker affects final reply content:**
- The markers only target `<think>` block behavior
- Indirect effect: immersion mode → more emotionally authentic replies; analysis mode → more structurally consistent replies

**System prompt placement doesn't work:**
```python
# ❌ Less reliable
{"role": "system", "content": system_prompt + INNER_OS_MARKER}

# ✅ Correct — append to first user message
{"role": "user", "content": first_message + INNER_OS_MARKER}
```
```
