```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion prompt
  - switch deepseek thinking style
  - deepseek roleplay control instructions
  - deepseek think tag behavior
  - deepseek roleplay inner OS marker
  - deepseek v4 flash pro roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes special control markers that modify **how the model thinks inside `<think>` tags** during roleplay conversations. By appending a specific instruction to the **first user message**, you can steer the model toward:

- **角色沉浸 (Character Immersion)** — The model thinks in first-person as the character, with bracketed inner monologue `（心想：……）`
- **纯分析 (Pure Analysis)** — The model thinks analytically like a director, no in-character inner voice
- **Default** — Model chooses automatically based on scene complexity

The markers work because they match the model's training-time injection position: appended to the first user turn, they persist in context for all subsequent turns automatically.

**Supported surfaces:**
- DeepSeek official APP / web (Expert Mode / 专家模式)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

---

## The Two Marker Strings

### Character Immersion Marker (`INNER_OS_MARKER`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker (`NO_INNER_OS_MARKER`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Thinking Mode Output Comparison

```
Character Immersion Mode:              Pure Analysis Mode:
<think>                                <think>
（他跟我打招呼了……心跳加速。）           场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。               回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）             控制 150 字，先动作描写再对话。
</think>                               </think>
```

---

## Python API Integration

### Core Constants and Builder

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


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking mode marker.

    Args:
        system_prompt: Character/scenario description for the system role
        user_first_message: The player's first message / scene opener
        mode: One of "inner_os", "no_inner_os", or "default"

    Returns:
        List of message dicts ready for the DeepSeek chat API
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — append nothing

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Roleplay Session

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


# --- Round 1: inject marker once ---
system = "你是一个傲娇的女高中生，表面冷淡，内心其实很在乎对方。"
first_msg = "「我走进教室」"早上好。""

messages = build_messages(system, first_msg, mode="inner_os")
reply = chat(messages)
print("Assistant:", reply)

# --- Round 2+: just append, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Assistant:", reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Assistant:", reply)
```

### Async Version

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)


async def roleplay_session(system: str, opening: str, mode: str = "inner_os"):
    messages = build_messages(system, opening, mode=mode)

    while True:
        response = await client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"[Assistant]\n{reply}\n")

        user_input = input("[You] > ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        messages.append({"role": "user", "content": user_input})


asyncio.run(roleplay_session(
    system="你是一个神秘的古代剑客，话不多，但每句话都意味深长。",
    opening="「我在客栈门口看到一个蒙面人」"请问，这里可以住宿吗？"",
    mode="inner_os",
))
```

### Streaming with Think-Tag Parsing

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)


def stream_with_think_separation(messages: list[dict]):
    """Stream response and separate <think> content from final reply."""
    stream = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=True,
    )

    buffer = ""
    in_think = False
    think_content = []
    reply_content = []

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta

        # Simple state-machine to split <think>...</think> from reply
        while buffer:
            if not in_think:
                start = buffer.find("<think>")
                if start == -1:
                    reply_content.append(buffer)
                    buffer = ""
                else:
                    reply_content.append(buffer[:start])
                    buffer = buffer[start + 7:]
                    in_think = True
            else:
                end = buffer.find("</think>")
                if end == -1:
                    think_content.append(buffer)
                    buffer = ""
                else:
                    think_content.append(buffer[:end])
                    buffer = buffer[end + 8:]
                    in_think = False

    return {
        "thinking": "".join(think_content).strip(),
        "reply": "".join(reply_content).strip(),
    }


# Usage
messages = build_messages(
    "你是一个孤独的灯塔守护者，已独守灯塔二十年。",
    "「一艘小船靠近灯塔」"有人在吗？我需要帮助！"",
    mode="inner_os",
)
result = stream_with_think_separation(messages)
print("=== THINKING PROCESS ===")
print(result["thinking"])
print("\n=== REPLY ===")
print(result["reply"])
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

All subsequent messages need no modification — the marker remains in context history.

---

## Configuration Patterns

### Environment Setup

```bash
# .env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-v4-pro   # or deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    api_key: str = os.environ["DEEPSEEK_API_KEY"]
    base_url: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    think_mode: str = "inner_os"   # "inner_os" | "no_inner_os" | "default"
```

### Reusable Session Class

```python
class RoleplaySession:
    """Manages a multi-turn DeepSeek roleplay conversation."""

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-pro",
    ):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self._first_turn = True

        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

    def send(self, user_message: str) -> str:
        # Inject marker only on first user turn
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.messages = []
        self._first_turn = True


# Usage
session = RoleplaySession(
    system_prompt="你是一位严厉但内心温柔的剑道教练。",
    mode="inner_os",
)

print(session.send("「我第一次走进道场，紧张地鞠躬」"请收我为徒。""))
print(session.send("「我每天早到一小时练习」"教练，我今天来早了。""))
```

---

## Common Patterns

### Pattern 1: Mode Selection Based on Genre

```python
def select_mode_for_genre(genre: str) -> str:
    """Suggest a thinking mode based on roleplay genre."""
    immersive_genres = {"romance", "slice_of_life", "drama", "horror"}
    analytical_genres = {"mystery", "strategy", "debate", "world_building"}

    if genre in immersive_genres:
        return "inner_os"
    elif genre in analytical_genres:
        return "no_inner_os"
    return "default"


mode = select_mode_for_genre("romance")
session = RoleplaySession(system_prompt="...", mode=mode)
```

### Pattern 2: Switching Modes Between Conversations

```python
# To switch modes, simply start a new session — never mid-conversation
session_v1 = RoleplaySession(system_prompt=system, mode="inner_os")
# ... use session_v1 ...

# New conversation, different mode
session_v2 = RoleplaySession(system_prompt=system, mode="no_inner_os")
```

### Pattern 3: Marker Injection for Existing Message Arrays

```python
def inject_marker_into_history(
    messages: list[dict],
    mode: str
) -> list[dict]:
    """
    Inject the mode marker into the first user message of an existing
    messages list (e.g., loaded from storage).
    """
    marker = ""
    if mode == "inner_os":
        marker = INNER_OS_MARKER
    elif mode == "no_inner_os":
        marker = NO_INNER_OS_MARKER
    else:
        return messages

    result = []
    injected = False
    for msg in messages:
        if not injected and msg["role"] == "user":
            result.append({**msg, "content": msg["content"] + marker})
            injected = True
        else:
            result.append(msg)
    return result
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Marker has no effect | Placed in system prompt instead of first user turn | Move marker to end of first `user` message |
| Mode reverts after a few turns | Marker was added to a non-first turn | Always inject on turn 1; it persists via context window |
| Inner monologue appears in `no_inner_os` mode | Probabilistic — model doesn't guarantee 100% compliance | Re-roll (send again); compliance is increased, not guaranteed |
| `<think>` tag not visible | Using web Quick Mode (快速模式) | Switch to Expert Mode (专家模式) or use API |
| Marker text appears in final reply | Model leaked system instructions | This is rare; try rephrasing the system prompt to not mention the markers |
| API returns error on Chinese characters | Encoding issue | Ensure your Python source file is UTF-8; use `# -*- coding: utf-8 -*-` header |

### Verifying Mode Activation

```python
def verify_think_mode(response_text: str, expected_mode: str) -> dict:
    """Check whether the model's think block matched the expected mode."""
    import re

    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if not think_match:
        return {"activated": False, "reason": "No <think> block found"}

    think_content = think_match.group(1)
    has_brackets = bool(re.search(r"[（(].{1,50}[）)]", think_content))
    has_first_person = any(
        phrase in think_content for phrase in ["我心想", "我觉得", "我暗自", "内心OS"]
    )

    if expected_mode == "inner_os":
        activated = has_brackets or has_first_person
    elif expected_mode == "no_inner_os":
        activated = not has_brackets and not has_first_person
    else:
        activated = True  # default — no expectation

    return {
        "activated": activated,
        "has_brackets": has_brackets,
        "has_first_person": has_first_person,
        "think_preview": think_content[:200],
    }
```

---

## Key Rules Summary

1. **Inject once, on the first user turn** — never in `system`, never mid-conversation
2. **Blank line between story text and marker** — improves parse reliability  
3. **Do not repeat the marker** in subsequent turns — it stays in history automatically
4. **Not 100% deterministic** — re-roll if the first attempt doesn't activate
5. **Think content ≠ final reply** — markers only affect `<think>` blocks; final reply tone is indirectly influenced
6. **New mode = new conversation** — you cannot switch modes within an ongoing thread
```
