```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special control instructions.
triggers:
  - "add inner monologue to DeepSeek roleplay"
  - "switch DeepSeek thinking mode"
  - "DeepSeek roleplay control instructions"
  - "make DeepSeek think in character"
  - "disable inner OS in DeepSeek"
  - "DeepSeek chain of thought roleplay"
  - "deepseek-v4 roleplay instruct markers"
  - "control DeepSeek think tag behavior"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions injected into the **first user message** to steer how the model thinks inside `<think>` tags during roleplay. This enables two modes beyond the default:

- **Character Immersion Mode**: The model's reasoning contains first-person inner monologue wrapped in parentheses — like an actor "in character."
- **Pure Analysis Mode**: The model's reasoning is purely logical and structured — like a director planning the scene, no inner monologue.

**Supported surfaces:**
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- ⚠️ Web **Quick Mode** is NOT supported

---

## Core Concept

The control instruction is appended to the **first user message only**. Because DeepSeek keeps full conversation history in context, the instruction remains active for all subsequent turns automatically — you never repeat it.

```
Thinking mode comparison:

Character Immersion (inner_os):        Pure Analysis (no_inner_os):
<think>                                <think>
（He greeted me… heart racing.）        Scene: user greets, character is tsundere.
I need to act indifferent.             Strategy: feign disinterest, body language reveals truth.
（Don't let him see I'm happy!）        Keep ~150 chars, action then dialogue.
</think>                               </think>
```

---

## Instruction Strings

Copy these exactly. Do not use the bracket labels — use the full instruction text.

### Character Immersion Mode (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode (`no_inner_os`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python API Integration

### Marker Constants

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
    Build the initial messages list with the appropriate thinking mode marker.
    Only call this for the FIRST turn. Subsequent turns append normally.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no marker appended; model chooses automatically

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Conversation Example

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

# ── Turn 1: inject marker once ──────────────────────────────────────────────
messages = build_messages(
    system_prompt="你是一个傲娇的女高中生，对喜欢的人总是口是心非。",
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",  # character immersion
)
reply = chat(messages)
print("Assistant:", reply)

# ── Turn 2+: append normally, marker stays in history ───────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Assistant:", reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上的疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Assistant:", reply)
```

### Async Variant

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages: list[dict]) -> str:
    response = await client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        system_prompt="You are a calm coffee shop barista.",
        user_first_message="「I push open the café door.」 "Any seats left?"",
        mode="no_inner_os",  # pure analysis
    )
    reply = await async_chat(messages)
    print(reply)

asyncio.run(main())
```

### Streaming with Thinking Tag Capture

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def stream_chat(messages: list[dict]):
    """Stream response and separately capture <think> content."""
    think_buf = []
    reply_buf = []
    in_think = False

    with client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            # Naive tag tracking — adjust for production use
            if "<think>" in delta:
                in_think = True
            if "</think>" in delta:
                in_think = False
                continue
            (think_buf if in_think else reply_buf).append(delta)
            if not in_think:
                print(delta, end="", flush=True)

    return "".join(think_buf), "".join(reply_buf)

messages = build_messages(
    system_prompt="你是一个神秘的占卜师。",
    user_first_message="「我推开占卜师的门」"你真的能看见未来吗？"",
    mode="inner_os",
)
thinking, reply = stream_chat(messages)
print("\n\n--- Thinking Process ---")
print(thinking)
```

---

## Web Usage (No Code)

Paste the full instruction at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are sent normally — the instruction remains active.

---

## Configuration Reference

| Parameter | Value | Notes |
|---|---|---|
| `mode` | `"default"` | Model auto-selects thinking style |
| `mode` | `"inner_os"` | Character immersion, parenthesized inner monologue |
| `mode` | `"no_inner_os"` | Pure analysis, no inner monologue |
| Injection point | First user message (end) | Training-aligned; most stable placement |
| System prompt | Any | Marker goes in user message, not system |
| Supported models | `deepseek-v4-flash`, `deepseek-v4-pro` | Expert/API mode only |

---

## Common Patterns

### Pattern 1: Reusable Session Class

```python
import os
from openai import OpenAI

class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os"):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = "deepseek-v4-pro"
        self.messages = []
        self.mode = mode
        self.system_prompt = system_prompt

    def send(self, user_message: str) -> str:
        if not self.messages:
            # First turn: inject system + marked first user message
            first_turn = user_message
            if self.mode == "inner_os":
                first_turn += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                first_turn += NO_INNER_OS_MARKER

            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": first_turn},
            ]
        else:
            self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个冷酷的剑客，内心深藏柔情。",
    mode="inner_os",
)

print(session.send("「陌生人挡住了去路」"让开。""))
print(session.send("「我没有动」"你是在保护谁？""))
print(session.send("「雨开始下了」"你愿意告诉我你的名字吗？""))
```

### Pattern 2: Mode Switching (New Session)

```python
def new_session_with_mode(system: str, first_msg: str, mode: str):
    """Helper to quickly start a fresh session in a given mode."""
    session = DeepSeekRoleplaySession(system, mode=mode)
    return session, session.send(first_msg)

# Switch from immersive to analytical by starting a new session
session_a, _ = new_session_with_mode(
    "你是侦探。", "「案发现场」"尸体是何时发现的？"", mode="no_inner_os"
)

session_b, _ = new_session_with_mode(
    "你是侦探。", "「案发现场」"尸体是何时发现的？"", mode="inner_os"
)
```

### Pattern 3: Verify Thinking Mode Activated

```python
def check_think_tag(response_text: str) -> dict:
    """Parse whether inner_os markers appeared in the think block."""
    import re
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if not think_match:
        return {"has_think": False, "has_inner_os": False, "think_content": ""}

    think_content = think_match.group(1)
    has_inner_os = bool(re.search(r"[（(].+?[）)]", think_content))
    return {
        "has_think": True,
        "has_inner_os": has_inner_os,
        "think_content": think_content,
    }

result = check_think_tag(full_response_with_think_tag)
if not result["has_inner_os"] and expected_mode == "inner_os":
    print("Mode may not have triggered — try re-rolling this turn.")
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Instruction has no effect | Quick Mode active on web | Switch to Expert Mode |
| Think tag shows wrong style | Probabilistic — not 100% guaranteed | Re-send the same message; retry until it triggers |
| Marker placed in system prompt | Wrong injection point | Move marker to end of first **user** message |
| Mode stops working mid-conversation | Marker was removed from history | Keep full history; never truncate first user message |
| API returns no `<think>` block | Model or endpoint doesn't support extended thinking | Use `deepseek-v4-flash` or `deepseek-v4-pro` explicitly |
| Characters appear garbled | Encoding issue with Chinese characters | Ensure your Python source file is UTF-8; use `# -*- coding: utf-8 -*-` |

### Retry Helper

```python
import time

def chat_with_retry(messages: list[dict], expected_mode: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
        )
        content = response.choices[0].message.content
        result = check_think_tag(content)

        if expected_mode == "inner_os" and result["has_inner_os"]:
            return content
        if expected_mode == "no_inner_os" and not result["has_inner_os"]:
            return content
        if expected_mode == "default":
            return content

        print(f"Attempt {attempt + 1}: mode not triggered, retrying...")
        time.sleep(0.5)

    print("Warning: could not confirm mode after retries, returning last response.")
    return content
```

---

## Environment Setup

```bash
# Install the OpenAI-compatible client
pip install openai

# Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-key-from-platform.deepseek.com"
```

```python
# Minimal working setup check
import os
from openai import OpenAI

assert os.environ.get("DEEPSEEK_API_KEY"), "Set DEEPSEEK_API_KEY env var"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# Verify connectivity
models = client.models.list()
print("Connected. Available models:", [m.id for m in models.data])
```
```
