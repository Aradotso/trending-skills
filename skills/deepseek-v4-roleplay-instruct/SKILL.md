```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special control instructions.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek roleplay control instructions
  - deepseek think tag roleplay
  - deepseek v4 pure analysis mode
  - deepseek roleplay api setup
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides special control instructions that alter how DeepSeek-V4 models reason inside their `<think>` tags during roleplay sessions. You can make the model think like an actor (character-immersive inner monologue) or like a director (pure logical analysis), without changing the final response format.

---

## What It Does

DeepSeek-V4 (`deepseek-v4-flash`, `deepseek-v4-pro`) exposes its chain-of-thought inside `<think>` tags. By appending a special marker to the **first user message**, you steer the thinking style for the entire conversation:

| Mode | Thinking Style |
|---|---|
| **Default** | Model chooses automatically based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Director-style logical planning, no in-character inner voice |

**Scope:** DeepSeek official APP/web (Expert Mode) and the `deepseek-v4-flash` / `deepseek-v4-pro` APIs. Quick Mode on web is not supported.

---

## The Control Markers

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

> **Key rule:** Append the marker to the **first user message only**. It persists automatically in conversation history for all subsequent turns.

---

## Installation / Setup

No package to install. Copy the marker strings into your project directly, or clone the reference repo for the latest marker text:

```bash
git clone https://github.com/victorchen96/deepseek_v4_rolepaly_instruct.git
```

Set your DeepSeek API key as an environment variable:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

---

## Core Usage Pattern (Python)

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# ── Marker definitions ────────────────────────────────────────────────────────

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
    mode: str = "default",          # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """Build the initial messages list with the control marker injected."""
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]

def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    """Send messages and return the assistant reply text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

# ── Multi-turn roleplay session ───────────────────────────────────────────────

system_prompt = "你是一个傲娇的女高中生，名叫小雪，喜欢主角但死要面子。"

# Round 1 — inject marker once
messages = build_messages(
    system_prompt,
    "「我走进教室」"早上好。"",
    mode="inner_os",
)
reply = chat(messages)
print("Round 1:", reply)

# Round 2+ — just append, no marker needed
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

---

## Async Version

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_session():
    system_prompt = "你是一个神秘的咖啡师，话不多，但观察力极强。"
    messages = build_messages(
        system_prompt,
        "「我推开咖啡店的门」"你好，请问还有位置吗？"",
        mode="no_inner_os",   # director / pure-analysis mode
    )

    response = await client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    reply = response.choices[0].message.content
    print(reply)

asyncio.run(roleplay_session())
```

---

## Streaming Support

```python
def chat_stream(messages: list[dict], model: str = "deepseek-v4-flash"):
    """Stream tokens as they arrive."""
    with client.chat.completions.stream(
        model=model,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # newline after stream ends
```

---

## Mode Selection Guide

```python
def choose_mode(scenario: str) -> str:
    """
    Heuristic for picking the right thinking mode.

    inner_os    → emotional scenes, character development, romance, drama
    no_inner_os → action choreography, puzzle solving, world-building exposition
    default     → let the model decide (works well for mixed scenes)
    """
    emotional_keywords = ["romance", "confession", "argument", "grief", "jealousy"]
    analytical_keywords = ["battle", "mystery", "investigation", "strategy"]

    if any(k in scenario.lower() for k in emotional_keywords):
        return "inner_os"
    if any(k in scenario.lower() for k in analytical_keywords):
        return "no_inner_os"
    return "default"

# Usage
mode = choose_mode("romance confession scene")
messages = build_messages(system_prompt, first_user_message, mode=mode)
```

---

## Web / Chat UI Usage

Paste the marker at the end of your **first** message, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are sent normally — no need to repeat the marker.

---

## Configuration Reference

| Parameter | Value | Notes |
|---|---|---|
| `model` | `deepseek-v4-flash` / `deepseek-v4-pro` | Flash = faster/cheaper; Pro = higher quality |
| `base_url` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` | Set once on first message |
| Marker position | End of first **user** message | Training injection point — most reliable location |
| System prompt | Any | Marker goes in user turn, not system prompt |

---

## Common Patterns

### Pattern 1: Reusable Session Class

```python
class RoleplaySession:
    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._mode = mode
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            if self._mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self._mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

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
    system_prompt="你是一位冷峻的侦探，正在审讯嫌疑人。",
    mode="no_inner_os",
)
print(session.send("「我坐在审讯室的椅子上，双臂交叉」"我什么都不知道。""))
print(session.send("「我把一张照片推到桌子中央」"那这张照片怎么解释？""))
```

### Pattern 2: Switch Mode by Opening New Session

```python
# inner_os session
session_a = RoleplaySession(system_prompt, mode="inner_os")

# no_inner_os session (different conversation, different mode)
session_b = RoleplaySession(system_prompt, mode="no_inner_os")
```

### Pattern 3: Verify Mode Activated

```python
def verify_mode(reply_with_think: str, expected_mode: str) -> bool:
    """
    Check raw API response (including <think> block) for mode markers.
    Only works if the API returns reasoning_content or raw text with <think>.
    """
    think_block = ""
    if "<think>" in reply_with_think:
        start = reply_with_think.index("<think>") + len("<think>")
        end = reply_with_think.index("</think>")
        think_block = reply_with_think[start:end]

    if expected_mode == "inner_os":
        # Should contain parenthetical inner monologue
        return "（" in think_block or "心想" in think_block
    elif expected_mode == "no_inner_os":
        # Should NOT contain parenthetical inner monologue
        return "（" not in think_block and "心想" not in think_block
    return True
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Mode not activating | Marker placed in system prompt | Move marker to end of first **user** message |
| Mode not activating | Marker added after turn 1 | Start a new conversation; inject on turn 1 only |
| Inconsistent activation | Probabilistic by design | Re-roll (send again); ~stable but not 100% |
| Mode drifts mid-conversation | Normal LLM behavior | The marker stays in history and re-asserts each turn automatically |
| `deepseek-v4-flash` not found | Wrong model name | Check current model names at `api.deepseek.com` docs |
| Quick Mode (web) not working | Feature not supported there | Switch to Expert Mode in DeepSeek web/app |

---

## Key Facts for AI Agents

- **Marker injection point:** Always the **last part of the first user message**. Never in system prompt. Never in later turns.
- **Persistence:** The marker stays in conversation history automatically — no need to re-inject.
- **Probabilistic:** Not guaranteed every call. If the style is wrong, retry.
- **Final output unaffected:** The markers change `<think>` content only, not the visible reply.
- **Model family:** `deepseek-v4-flash` (fast) and `deepseek-v4-pro` (quality). Both support this feature via API.
- **API compatibility:** OpenAI Python SDK works directly with `base_url="https://api.deepseek.com/v1"`.
```
