```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning via special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 roleplay instruct
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek think tag roleplay
  - deepseek roleplay prompt markers
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct provides **special control markers** that you append to the first user message in a conversation to steer how DeepSeek-V4 reasons inside its `<think>` tags during roleplay scenarios.

Two modes are available:

| Mode | Marker constant | `<think>` behaviour |
|---|---|---|
| **Character Immersion** (`inner_os`) | `INNER_OS_MARKER` | First-person inner monologue wrapped in parentheses, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | `NO_INNER_OS_MARKER` | Cold, structured planning — no in-character inner voice |
| **Default** | *(nothing)* | Model decides automatically |

**Supported surfaces:**
- DeepSeek official app / web — **Expert Mode** only
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- ⚠️ Web "Quick Mode" is **not** supported

> **Probabilistic**: Markers increase the likelihood of the desired format but cannot guarantee 100% compliance. Re-roll if the first attempt does not match.

---

## Installation / Setup

No package to install. Copy the marker strings directly into your project.

### Environment / API key

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"   # adjust if needed
```

---

## Marker Strings (copy-paste ready)

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

---

## Core Helper — `build_messages`

```python
def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",          # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Build the initial messages list, injecting the correct marker into
    the first user turn.  Subsequent turns need no special handling.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no marker appended

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Why first user turn only?

The model sees the full conversation history on every call. Because the marker lives in the first user message it remains in context for all subsequent turns — no need to repeat it.

---

## Full Working Example (Python + OpenAI-compatible client)

```python
import os
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


def build_messages(system_prompt, user_first_message, mode="default"):
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

SYSTEM = "你是一个傲娇的女高中生，不善于表达感情，但内心其实很在意对方。"

# ── Turn 1: inject marker ──────────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # character-immersive thinking
)

resp1 = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
assistant_reply_1 = resp1.choices[0].message.content
print("Turn 1:", assistant_reply_1)

# ── Turn 2+: normal append, no extra marker needed ─────────────────────────
messages.append({"role": "assistant", "content": assistant_reply_1})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

resp2 = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
print("Turn 2:", resp2.choices[0].message.content)
```

---

## Streaming Example

```python
stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
)

think_buf, reply_buf = [], []
in_think = False

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    # DeepSeek wraps reasoning in <think>...</think>
    if "<think>" in delta:
        in_think = True
    if in_think:
        think_buf.append(delta)
    else:
        reply_buf.append(delta)
    if "</think>" in delta:
        in_think = False

print("THINKING:\n", "".join(think_buf))
print("\nREPLY:\n",   "".join(reply_buf))
```

---

## Web / App Usage (no code)

Paste the marker at the end of your **first** message in a new conversation, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages require no changes. Click **"查看思考过程"** (View thinking process) to verify the mode is active.

---

## Common Patterns

### Pattern 1 — Session manager class

```python
class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "default",
                 model: str = "deepseek-v4-pro"):
        self.model = model
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._mode = mode
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn:
            if self._mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self._mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages
        )
        reply = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个冷淡的图书馆管理员，内心其实渴望交流。",
    mode="inner_os",
)
print(session.chat("「我走到服务台」"请问有没有推理小说区？""))
print(session.chat("「我拿着书回到她面前」"谢谢，这本书适合我吗？""))
```

### Pattern 2 — Switch mode per new conversation

```python
def new_session(system, first_msg, mode="default"):
    """Convenience factory; returns (messages_list, first_reply)."""
    msgs = build_messages(system, first_msg, mode)
    resp = client.chat.completions.create(model="deepseek-v4-pro", messages=msgs)
    reply = resp.choices[0].message.content
    msgs.append({"role": "assistant", "content": reply})
    return msgs, reply

# Character-immersive
msgs, r = new_session("你是傲娇女生", "早上好", mode="inner_os")

# Pure analysis (same system, fresh conversation)
msgs2, r2 = new_session("你是傲娇女生", "早上好", mode="no_inner_os")
```

### Pattern 3 — Async with `httpx` / `aiohttp`

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

async def async_roleplay(system, first_msg, mode="inner_os"):
    messages = build_messages(system, first_msg, mode)
    resp = await async_client.chat.completions.create(
        model="deepseek-v4-pro", messages=messages
    )
    return resp.choices[0].message.content

result = asyncio.run(async_roleplay("你是一个神秘的占卜师", "「我走进占卜室」"))
print(result)
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Pass to `build_messages` or session constructor |
| `model` | `"deepseek-v4-pro"` / `"deepseek-v4-flash"` | Flash is faster/cheaper; Pro is higher quality |
| Marker position | **End of first user message only** | Training-time injection point — do not put in system prompt |
| Web surface | Expert Mode only | Quick Mode does not expose `<think>` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Thinking shows no inner monologue despite `inner_os` | Probabilistic — model ignored marker | Regenerate (re-roll) the response |
| Thinking still has parenthetical inner voice despite `no_inner_os` | Same probabilistic issue | Regenerate; consider adding stronger negative examples in system prompt |
| Marker appended but `<think>` block is empty | Model / surface doesn't support reasoning output | Ensure you're using Expert Mode (web) or a supported API model |
| `AttributeError` on `resp.choices[0].message.content` | Streaming vs non-streaming mismatch | Use the streaming example if `stream=True` |
| Marker appears verbatim in the final reply (not just thinking) | Wrong surface / model | Confirm model is `deepseek-v4-flash` or `deepseek-v4-pro` |
| Want to change mode mid-conversation | Not supported in-session | Start a **new conversation** with the other marker |

---

## Key Rules to Remember

1. **Marker goes in the first user turn** — not system prompt, not a later turn.
2. **One marker per conversation** — the history carries it forward automatically.
3. **New mode = new conversation** — you cannot switch mid-session.
4. **Probabilistic** — re-roll if the first attempt doesn't comply.
5. **Final reply is unaffected by design** — markers only shape the `<think>` reasoning block; output quality/style changes are indirect.
```
