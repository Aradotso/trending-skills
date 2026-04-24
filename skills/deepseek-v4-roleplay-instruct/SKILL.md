```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking chain style during roleplay via special instruction markers injected into the first user message
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue roleplay
  - deepseek v4 roleplay instructions
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis thinking
  - deepseek roleplay chain of thought control
  - deepseek think tag roleplay marker
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes its chain-of-thought reasoning inside `<think>` tags. During roleplay conversations this thinking block can either feel like an **actor's inner monologue** (first-person, parenthesised inner voice) or a **director's cold analysis** (purely logical, no character voice). By appending a special marker string to the **first user message**, you steer which style the model uses for the entire conversation — because every subsequent reply still sees the full history including that first marker.

Two modes beyond the default:

| Mode | Marker constant | `<think>` style |
|---|---|---|
| **Inner-OS / Character Immersion** | `INNER_OS_MARKER` | First-person inner monologue wrapped in `（）` brackets |
| **Pure Analysis** | `NO_INNER_OS_MARKER` | Clinical scene/strategy analysis, no character voice |

**Supported surfaces**
- DeepSeek official app / web — **Expert Mode** only
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web **Quick Mode** is NOT supported

---

## Installation / Setup

No package to install — this is a prompting pattern. You need access to the DeepSeek API.

```bash
pip install openai          # DeepSeek uses an OpenAI-compatible endpoint
export DEEPSEEK_API_KEY=sk-...
```

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```

---

## Core Marker Strings

Copy these exactly — the Chinese instruction text is what the model was trained on.

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

## Key API Pattern

### `build_messages` helper

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


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",          # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Append the appropriate marker to the first user message.
    mode="default"     → no marker, model chooses automatically
    mode="inner_os"    → character immersion / inner monologue
    mode="no_inner_os" → pure analytical thinking, no character voice
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

### Full multi-turn conversation example

```python
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Round 1: inject the marker once ──────────────────────────────────────────
system = "你是一个傲娇的女高中生，名叫雪乃，暗恋主角但绝不承认。"
messages = build_messages(
    system_prompt=system,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # character immersion
)
reply_1 = chat(messages)
print(reply_1)

# ── Round 2+: just append normally, marker stays in history ──────────────────
messages.append({"role": "assistant", "content": reply_1})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

reply_2 = chat(messages)
print(reply_2)

messages.append({"role": "assistant", "content": reply_2})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})

reply_3 = chat(messages)
print(reply_3)
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"`, `"inner_os"`, `"no_inner_os"` | Only applied to the **first** user message |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Both support the markers |
| Marker placement | End of first user message, separated by blank line | Training-aligned position; do NOT place in system prompt |

### Where NOT to put the marker

```python
# ❌ Wrong — system prompt placement is less reliable
messages = [
    {"role": "system", "content": system + INNER_OS_MARKER},
    {"role": "user",   "content": first_message},
]

# ✅ Correct — appended to first user message
messages = build_messages(system, first_message, mode="inner_os")
```

---

## Web / App Usage

Paste the marker at the end of your **first** message in a new conversation (Expert Mode):

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no changes — type them normally.

---

## Common Patterns

### Pattern 1 — Reusable roleplay session factory

```python
from dataclasses import dataclass, field

@dataclass
class RoleplaySession:
    system_prompt: str
    mode: str = "inner_os"
    model: str = "deepseek-v4-flash"
    messages: list[dict] = field(default_factory=list)
    _started: bool = False

    def send(self, user_message: str) -> str:
        if not self._started:
            self.messages = build_messages(self.system_prompt, user_message, self.mode)
            self._started = True
        else:
            self.messages.append({"role": "user", "content": user_message})

        reply = chat(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = RoleplaySession(
    system_prompt="你是一个神秘的剑客，寡言少语，但内心细腻。",
    mode="inner_os",
)
print(session.send("「陌生人坐到你对面」"这位，借个火。""))
print(session.send("「我点上烟，打量着你」"你是在等什么人？""))
```

### Pattern 2 — Switching modes per conversation

```python
def new_session(system: str, opening: str, mode: str) -> list[dict]:
    """Start a fresh conversation with the chosen thinking mode."""
    return build_messages(system, opening, mode)


# Immersive session
immersive = new_session(
    "你扮演一个失忆的侦探。",
    "「雨夜，你站在案发现场」",
    mode="inner_os",
)

# Analytical session for the same scenario
analytical = new_session(
    "你扮演一个失忆的侦探。",
    "「雨夜，你站在案发现场」",
    mode="no_inner_os",
)
```

### Pattern 3 — Streaming with thinking block visible

```python
def stream_chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    full_reply = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_reply.append(delta)
    print()
    return "".join(full_reply)


messages = build_messages(
    "你是一只会说话的猫，傲慢但偶尔撒娇。",
    "「我拎着一袋猫粮回家」"我回来了！"",
    mode="inner_os",
)
reply = stream_chat(messages)
messages.append({"role": "assistant", "content": reply})
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Marker had no effect | Wrong mode / Quick Mode on web | Switch to Expert Mode; retry (probabilistic, re-roll if needed) |
| Inner monologue appears in `no_inner_os` mode | Model compliance ~not 100% | Re-roll; consider stronger negative phrasing in system prompt |
| `<think>` tags not visible | Web UI collapses them | Click "查看思考过程" to expand |
| Marker effect fades after many turns | Context window pressure | Marker is still in history; if very long conversation, consider summarising mid-history while keeping round 1 intact |
| API returns no `<think>` block | Using non-thinking model or Quick endpoint | Ensure model is `deepseek-v4-flash` or `deepseek-v4-pro` with Expert/thinking mode enabled |

### Verify mode is active (web)

After the model replies, click **"查看思考过程"** (View thinking process). You should see:

- **Inner-OS mode**: text like `（心想：……）` or `(内心OS：……)` inside the block
- **Pure analysis mode**: no parenthesised inner voice, only direct analytical statements

### Re-roll tip

The trigger is probabilistic. If the first reply does not show the expected thinking style, simply regenerate the response (do not send a new message). The marker is still present; each generation is an independent attempt.

---

## Quick Reference

```python
# One-liner for a new immersive chat
msgs = build_messages("系统提示", "第一条用户消息", mode="inner_os")

# One-liner for analytical mode
msgs = build_messages("系统提示", "第一条用户消息", mode="no_inner_os")

# Continue conversation — NO special handling needed
msgs.append({"role": "assistant", "content": last_reply})
msgs.append({"role": "user", "content": next_user_message})
```
```
