```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 roleplay instruct
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay prompt markers
  - control deepseek think tags roleplay
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct documents **special control markers** that influence how DeepSeek-V4 models reason inside their `<think>` tags during roleplay scenarios. By appending a marker to the **first user message**, you can steer the model's internal thinking between two distinct styles:

| Mode | Thinking Style |
|------|---------------|
| **Default** | Model auto-selects based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue with parenthetical asides, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Director-like cold logic: plot analysis, reply planning, no character voice |

**Supported surfaces:**
- DeepSeek official App / Web — **Expert Mode** only
- API: `deepseek-v4-flash` and `deepseek-v4-pro`
- Web **Quick Mode** is NOT supported

> **Note:** Markers probabilistically increase the chance of the desired format. If it doesn't trigger, retry the generation.

---

## The Marker Strings

### Character Immersion Marker

```text
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker

```text
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python Integration

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
from typing import Literal

Mode = Literal["default", "inner_os", "no_inner_os"]

def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: Mode = "default",
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking-mode marker
    appended to the first user message.

    Args:
        system_prompt:      The character/scene system prompt.
        user_first_message: The first user turn content.
        mode:               "default" | "inner_os" | "no_inner_os"

    Returns:
        A messages list ready to pass to the DeepSeek API.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Roleplay Session

```python
import os
from openai import OpenAI  # DeepSeek uses an OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。"

# ── Round 1: inject marker once ──────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # or "no_inner_os" / "default"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print(assistant_reply)

# ── Round 2+: append normally, marker stays in history automatically ─────────
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print(assistant_reply)
```

### Streaming Version

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def roleplay_stream(messages: list[dict], model: str = "deepseek-v4-pro"):
    """Yield text chunks from a streaming DeepSeek roleplay call."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

# Usage
messages = build_messages(
    system_prompt="你是一个神秘的咖啡店老板，身怀秘密。",
    user_first_message="「我推开沉重的木门」"还开着吗？"",
    mode="inner_os",
)

full_reply = ""
for token in roleplay_stream(messages):
    print(token, end="", flush=True)
    full_reply += token
print()  # newline

messages.append({"role": "assistant", "content": full_reply})
```

---

## Reusable Session Class

```python
import os
from openai import OpenAI
from typing import Literal

class DeepSeekRoleplaySession:
    """
    Manages a multi-turn DeepSeek roleplay conversation with a fixed thinking mode.
    The mode marker is injected once into the first user message and persists
    automatically through the conversation history.
    """

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
    }

    def __init__(
        self,
        system_prompt: str,
        mode: Literal["default", "inner_os", "no_inner_os"] = "default",
        model: str = "deepseek-v4-pro",
    ):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def chat(self, user_message: str) -> str:
        if self._first_turn and self.mode in self.MARKERS:
            user_message += self.MARKERS[self.mode]
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
        """Start a fresh conversation, optionally with a new system prompt."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        self._first_turn = True


# ── Example usage ─────────────────────────────────────────────────────────────
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.chat("「我走进教室」"早上好。""))
print(session.chat("「我在她旁边坐下」"今天心情不好吗？""))
print(session.chat("「我注意到她手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / Manual Usage

Paste the marker at the end of your **first message only**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no special treatment:

```
Round 2: 「我坐到窗边的位置」"来一杯美式。"
Round 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Click **"查看思考过程"** (View thinking process) to verify the mode activated.

---

## Common Patterns

### Pattern 1 — Switch modes between sessions

```python
# Session A: immersive
session_a = DeepSeekRoleplaySession(system_prompt, mode="inner_os")

# Session B: analytical (new conversation, different mode)
session_b = DeepSeekRoleplaySession(system_prompt, mode="no_inner_os")
```

### Pattern 2 — Retry on mode failure

```python
import re

def chat_with_mode_check(session: DeepSeekRoleplaySession, user_msg: str, retries: int = 3) -> str:
    """
    Retry up to `retries` times if the expected thinking style isn't detected.
    For inner_os mode, look for parenthetical markers in the raw response.
    """
    for attempt in range(retries):
        reply = session.chat(user_msg)
        if session.mode == "inner_os":
            # Check if <think> block contains parenthetical inner voice
            if re.search(r"[（(].*?[）)]", reply):
                return reply
            print(f"[Attempt {attempt+1}] Mode not detected, retrying...")
            # Pop the last two messages (user + assistant) to retry
            session.messages = session.messages[:-2]
        else:
            return reply
    return reply  # Return last attempt regardless
```

### Pattern 3 — Inject marker into existing messages list

```python
def inject_marker(messages: list[dict], mode: Literal["inner_os", "no_inner_os"]) -> list[dict]:
    """
    Add the marker to the first user message in an existing messages list.
    Useful when you build messages elsewhere and want to add mode control.
    """
    marker = DeepSeekRoleplaySession.MARKERS.get(mode, "")
    result = list(messages)
    for i, msg in enumerate(result):
        if msg["role"] == "user":
            result[i] = {**msg, "content": msg["content"] + marker}
            break
    return result
```

---

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` | `deepseek-v4-pro` or `deepseek-v4-flash` | Both support markers |
| `base_url` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| `api_key` | `os.environ["DEEPSEEK_API_KEY"]` | Never hardcode |
| Marker position | End of **first** user message | Training injection point — most reliable |
| System prompt | Any character description | Marker goes in user turn, not system |

---

## Troubleshooting

**Marker not triggering (model ignores it)**
- The mechanism is probabilistic. Re-generate (re-roll) 2–3 times.
- Ensure the marker is in the **first** user message, not the system prompt or a later turn.
- Confirm you're using **Expert Mode** on web, not Quick Mode.
- Verify the model is `deepseek-v4-flash` or `deepseek-v4-pro`.

**Inner monologue appearing in final reply (not just `<think>`)**
- This is a model behavior issue unrelated to the marker. The marker only targets `<think>` content.
- Add an explicit instruction in your system prompt: `"最终回复中不要出现括号内心戏"`.

**Marker affects final reply tone**
- Expected and documented. Immersive mode → more emotionally authentic replies. Analysis mode → more structurally consistent replies. Choose accordingly.

**Using the wrong position**
```python
# ❌ Wrong — marker in system prompt (less effective)
messages = [
    {"role": "system", "content": system_prompt + INNER_OS_MARKER},
    {"role": "user",   "content": user_first_message},
]

# ✅ Correct — marker appended to first user message
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_first_message + INNER_OS_MARKER},
]
```

**Resuming a saved conversation**
- If you serialize/deserialize `messages`, the marker is already embedded in the first user message string — no re-injection needed.
```python
import json

# Save
with open("session.json", "w") as f:
    json.dump(session.messages, f, ensure_ascii=False)

# Restore — marker is preserved in history
with open("session.json") as f:
    restored_messages = json.load(f)
# Continue by appending new turns directly
```
```
