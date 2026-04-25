```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning via special prompt markers.
triggers:
  - "deepseek roleplay thinking mode"
  - "deepseek inner monologue in think tag"
  - "switch deepseek roleplay mode"
  - "deepseek character immersion prompt"
  - "deepseek v4 roleplay instruct"
  - "control deepseek think tag roleplay"
  - "deepseek pure analysis mode roleplay"
  - "deepseek roleplay control instruction"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

This project documents **special control instructions** for DeepSeek-V4 (and compatible models) that steer the content inside the model's `<think>` reasoning block during roleplay sessions.

Two modes are available:

| Mode | Effect on `<think>` block |
|---|---|
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses — the model "acts" like the character |
| **Pure Analysis** (`no_inner_os`) | Cold, third-person analytical planning — the model "directs" like a screenplay writer |
| **Default** | Model picks automatically based on scene complexity |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode**
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web **Quick Mode** is NOT supported

---

## Core Concept

The markers are appended to the **first user message only**. Because DeepSeek sends full conversation history on every turn, the marker persists automatically for all subsequent turns — you never repeat it.

---

## The Two Prompt Markers (copy-ready)

### Character Immersion Mode

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python API Integration

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
    mode: str = "default"   # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Inject the roleplay thinking-mode marker into the first user message.
    Subsequent turns are appended normally — the marker stays in history automatically.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Example

```python
import os
from openai import OpenAI   # DeepSeek is OpenAI-API-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"   # or "deepseek-v4-flash"

SYSTEM = "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。"

# ── Turn 1: inject marker once ──────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # character immersion
)

resp = client.chat.completions.create(model=MODEL, messages=messages)
assistant_reply = resp.choices[0].message.content
print("Turn 1:", assistant_reply)

# ── Turn 2+: just append, no marker needed ──────────────────────────────────
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

resp = client.chat.completions.create(model=MODEL, messages=messages)
assistant_reply = resp.choices[0].message.content
print("Turn 2:", assistant_reply)

# ── Repeat pattern for further turns ────────────────────────────────────────
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "「轻声」"其实……你笑起来很好看。""})

resp = client.chat.completions.create(model=MODEL, messages=messages)
print("Turn 3:", resp.choices[0].message.content)
```

### Pure Analysis Mode Example

```python
messages = build_messages(
    system_prompt="你是一名冷静的侦探，擅长逻辑推理。",
    user_first_message="「我把证据摆在桌上」"你怎么解释这些？"",
    mode="no_inner_os",       # analytical director mode
)

resp = client.chat.completions.create(model=MODEL, messages=messages)
print(resp.choices[0].message.content)
```

---

## Reusable Session Class

```python
import os
from openai import OpenAI
from typing import Literal

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

ThinkMode = Literal["default", "inner_os", "no_inner_os"]


class DeepSeekRoleplaySession:
    """
    Manages a stateful multi-turn roleplay conversation with DeepSeek-V4.
    The thinking-mode marker is injected automatically on the first user turn.
    """

    def __init__(
        self,
        system_prompt: str,
        mode: ThinkMode = "default",
        model: str = "deepseek-v4-pro",
    ):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个开朗活泼的咖啡师，喜欢和客人闲聊。",
    mode="inner_os",
)

print(session.chat("「我推开咖啡店的门」"你好，还有位置吗？""))
print(session.chat("「坐到窗边」"来一杯拿铁吧。""))
print(session.chat("「看着窗外雨景」"今天天气真差。""))
```

---

## Web / Manual Usage

Paste the marker at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are plain text — no marker needed:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` | Controls `<think>` block style |
| `model` | `"deepseek-v4-pro"` / `"deepseek-v4-flash"` | Both support this feature |
| Marker position | End of **first user message** | Training-aligned; more stable than system prompt |
| `DEEPSEEK_API_KEY` | env var | Never hardcode — use `os.environ["DEEPSEEK_API_KEY"]` |

---

## Common Patterns

### Pattern 1 — One-shot helper function

```python
def deepseek_roleplay_turn(messages: list[dict], user_text: str, mode: ThinkMode = "default") -> tuple[str, list[dict]]:
    """Stateless helper: pass messages list in, get (reply, updated_messages) back."""
    is_first = not any(m["role"] == "user" for m in messages)

    if is_first:
        if mode == "inner_os":
            user_text += INNER_OS_MARKER
        elif mode == "no_inner_os":
            user_text += NO_INNER_OS_MARKER

    messages = messages + [{"role": "user", "content": user_text}]
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(model="deepseek-v4-pro", messages=messages)
    reply = resp.choices[0].message.content
    messages = messages + [{"role": "assistant", "content": reply}]
    return reply, messages
```

### Pattern 2 — Switch mode by starting a new session

```python
# Start fresh with a different mode — don't try to inject mid-conversation
session_immersive = DeepSeekRoleplaySession(system_prompt=SYSTEM, mode="inner_os")
session_analytical = DeepSeekRoleplaySession(system_prompt=SYSTEM, mode="no_inner_os")
```

### Pattern 3 — Verify the mode fired

```python
# DeepSeek API returns reasoning_content separately on some endpoints
resp = client.chat.completions.create(model=MODEL, messages=messages)
think_block = getattr(resp.choices[0].message, "reasoning_content", None)
if think_block:
    has_inner_os = "（" in think_block or "心想" in think_block
    print("Inner-OS active:", has_inner_os)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Mode didn't activate | Probabilistic — not 100% guaranteed | Re-roll (send again); the marker raises probability, not certainty |
| Marker in system prompt doesn't work | Not the trained injection position | Move marker to end of first **user** message |
| Works on turn 1, breaks later | Marker was removed from history | Keep full `messages` list — never truncate turn 1 |
| Web Quick Mode not working | Feature unsupported there | Switch to **Expert Mode** in DeepSeek web/app |
| `no_inner_os` still shows parentheses | Model occasionally ignores | Re-roll; add stronger preamble to system prompt |
| API key error | Wrong env var name | Ensure `DEEPSEEK_API_KEY` is exported in your shell / `.env` |

---

## Key Rules Summary

1. **Inject the marker once** — in the first user message, at the end, after a blank line.
2. **Never repeat** the marker in subsequent turns.
3. **Never inject into system prompt** — less effective than user message position.
4. **Re-roll if it doesn't fire** — probabilistic, not deterministic.
5. **To switch modes** — start a new conversation/session with the other marker.
6. **Verify via `<think>` / `reasoning_content`** — check for `（` or `心想` to confirm immersion mode fired.
```
