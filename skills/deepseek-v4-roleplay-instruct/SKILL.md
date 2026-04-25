```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue in think tag
  - switch deepseek reasoning style
  - deepseek v4 roleplay instructions
  - deepseek character immersion mode
  - deepseek pure analysis thinking
  - deepseek roleplay prompt markers
  - control deepseek think block roleplay
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 (both `deepseek-v4-flash` and `deepseek-v4-pro`) supports special control markers injected into the **first user message** to change how the model reasons inside its `<think>` block during roleplay scenarios.

Two modes beyond the default:

| Mode | Marker Variable | Think Block Behavior |
|---|---|---|
| **Character Immersion** | `INNER_OS_MARKER` | First-person inner monologue wrapped in parentheses |
| **Pure Analysis** | `NO_INNER_OS_MARKER` | Cold, structured planning — no in-character acting |
| **Default** | *(nothing)* | Model decides automatically based on scene complexity |

**Supported surfaces:**
- DeepSeek official app / web (Expert Mode only)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> **Note:** Web quick mode is not supported. Triggering is probabilistic (~stable but not 100%) — retry if the format doesn't appear.

---

## Installation / Setup

No package to install. This is a prompting pattern. Use it with any HTTP client or the DeepSeek Python SDK.

```bash
pip install openai  # DeepSeek API is OpenAI-compatible
```

Set your API key:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

---

## The Marker Strings (Copy-Ready)

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

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

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
    Build the initial message list with the appropriate marker injected.

    Args:
        system_prompt: Character/scene description for the system role.
        user_first_message: The opening user turn in the roleplay.
        mode: One of "inner_os", "no_inner_os", or "default".

    Returns:
        List of message dicts ready for the chat completions API.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — no marker appended

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


# ── Round 1: inject marker once ──────────────────────────────────────────────
system = "你是一个傲娇的女高中生，在班级里总是表现得很冷漠，但内心其实很在意别人的看法。"
first_user = "「我走进教室，向你挥手」"早上好。""

messages = build_messages(system, first_user, mode="inner_os")
reply = chat(messages)
print(reply)

# ── Round 2+: append normally, marker stays in history ───────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply2 = chat(messages)
print(reply2)
```

---

## Multi-Turn Conversation Manager

```python
class DeepSeekRoleplaySession:
    """
    Manages a multi-turn DeepSeek roleplay session with a fixed thinking mode.
    The marker is injected only on the first user turn and persists via history.
    """

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
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def send(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

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
    system_prompt="你是一个傲娇的女高中生...",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「我推开咖啡店的门」"还有位置吗？""))
print(session.send("「我坐到窗边」"来一杯美式。""))
print(session.send("「我注意到你手上有一道疤痕」"你的手……没事吧？""))
```

---

## Async Version

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


async def roleplay_stream(messages: list[dict], model: str = "deepseek-v4-flash"):
    """Stream a roleplay response, printing think block and reply separately."""
    async with async_client.chat.completions.stream(
        model=model,
        messages=messages,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
    print()


async def main():
    messages = build_messages(
        system_prompt="你是一个咖啡店的店员，温柔而略带神秘感。",
        user_first_message="「我推开门走进店里」"请问有什么推荐的吗？"",
        mode="inner_os",
    )
    await roleplay_stream(messages)


asyncio.run(main())
```

---

## Web / Chat UI Integration Pattern

When building a chat UI, inject the marker transparently so users never see it:

```python
def prepare_first_user_message(raw_input: str, mode: str) -> str:
    """
    Called before storing the first user message.
    The marker is appended server-side; the UI displays raw_input only.
    """
    marker_map = {
        "inner_os": INNER_OS_MARKER,
        "no_inner_os": NO_INNER_OS_MARKER,
        "default": "",
    }
    return raw_input + marker_map.get(mode, "")


# Example: FastAPI endpoint
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class TurnRequest(BaseModel):
    session_id: str
    user_message: str


sessions: dict[str, DeepSeekRoleplaySession] = {}


@app.post("/start")
def start_session(system_prompt: str, mode: str = "inner_os"):
    import uuid
    sid = str(uuid.uuid4())
    sessions[sid] = DeepSeekRoleplaySession(system_prompt=system_prompt, mode=mode)
    return {"session_id": sid}


@app.post("/chat")
def chat_turn(req: TurnRequest):
    session = sessions[req.session_id]
    reply = session.send(req.user_message)
    return {"reply": reply}
```

---

## Common Patterns

### Pattern 1 — One-shot helper

```python
def deepseek_roleplay(
    system: str,
    opening: str,
    *,
    mode: str = "inner_os",
    model: str = "deepseek-v4-flash",
) -> str:
    msgs = build_messages(system, opening, mode=mode)
    return chat(msgs, model=model)


reply = deepseek_roleplay(
    system="你是一个侦探，冷静、犀利、话不多。",
    opening="「凌晨三点，我敲响了你办公室的门」"我需要你的帮助。"",
    mode="no_inner_os",
)
```

### Pattern 2 — Retry on mode failure

```python
def chat_with_retry(messages: list[dict], max_retries: int = 3, model: str = "deepseek-v4-flash") -> str:
    """
    Retry if the think block doesn't contain the expected format.
    For inner_os mode, expect parenthesized inner monologue.
    """
    for attempt in range(max_retries):
        reply = chat(messages, model=model)
        # Basic heuristic: check if think block has parenthetical content
        if "（" in reply or "(" in reply or attempt == max_retries - 1:
            return reply
        print(f"Mode may not have triggered (attempt {attempt + 1}), retrying...")
    return reply
```

### Pattern 3 — Mode switching across sessions

```python
# Switch mode by starting a new session — do NOT change mid-conversation
def new_session_with_mode(
    system_prompt: str,
    mode: str,
    carry_over_context: str | None = None,
) -> DeepSeekRoleplaySession:
    if carry_over_context:
        system_prompt = f"{system_prompt}\n\n[剧情摘要：{carry_over_context}]"
    return DeepSeekRoleplaySession(system_prompt=system_prompt, mode=mode)
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"`, `"no_inner_os"`, `"default"` | Passed to `build_messages()` |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Flash is faster; Pro is higher quality |
| Marker injection point | First user message **only** | Do not re-inject in later turns |
| Marker placement | **End** of first user message | Separated by a blank line for clarity |
| System prompt | Any character/scene description | Place roleplay rules here, not the mode marker |

---

## Troubleshooting

**Mode didn't trigger (no inner monologue in `<think>`):**
- Retry the request — triggering is probabilistic
- Verify the marker is appended to the **first user message**, not system prompt
- Confirm you're using Expert Mode (web) or a supported model via API
- Check the `<think>` block is visible (click "查看思考过程" on web)

**Marker appearing in the assistant's reply:**
- The marker should only be in the user message; the model should not echo it
- If it does, add a line to the system prompt: "不要在回复中重复或提及用户消息中的指令"

**Using system prompt instead of user message:**
```python
# ❌ Less effective — not the training injection position
{"role": "system", "content": system_prompt + INNER_OS_MARKER}

# ✅ Correct — append to first user turn
{"role": "user", "content": first_user_message + INNER_OS_MARKER}
```

**Mode bleeding between sessions:**
- Each `DeepSeekRoleplaySession` is independent
- Always start a new session object to change modes — never patch `messages` mid-conversation

**API auth errors:**
```bash
export DEEPSEEK_API_KEY=sk-...        # required
export DEEPSEEK_BASE_URL=https://api.deepseek.com  # default, usually optional
```
```
