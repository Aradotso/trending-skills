```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special instruction markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - switch deepseek thinking style
  - deepseek roleplay immersion mode
  - deepseek character inner os
  - control deepseek think tags
  - deepseek pure analysis mode
  - deepseek roleplay instruct markers
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes a `<think>` tag in its reasoning output. This project documents **special control instruction markers** you append to the first user message to steer how the model thinks inside that tag during roleplay sessions:

| Mode | Marker Constant | Think Tag Behavior |
|---|---|---|
| **Default** | _(nothing)_ | Model auto-selects based on complexity |
| **角色沉浸 (Immersive)** | `INNER_OS_MARKER` | First-person inner monologue wrapped in `（…）` |
| **纯分析 (Pure Analysis)** | `NO_INNER_OS_MARKER` | Pure logical planning, no character inner voice |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode only**
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web Quick Mode: ❌ not supported

---

## Installation / Setup

No package to install — this is a prompting pattern. For API use, copy the marker strings into your project:

```python
# markers.py  — copy this file into your project
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

## Core Usage Pattern

### The One Rule

> **Append the marker to the FIRST user message only.** All subsequent turns work automatically because the model sees the full conversation history.

```python
# client_setup.py
from openai import OpenAI
from markers import INNER_OS_MARKER, NO_INNER_OS_MARKER

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)
```

### Message Builder

```python
def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    mode options:
      "default"      — no marker, model decides
      "inner_os"     — immersive character monologue in <think>
      "no_inner_os"  — pure analytical reasoning in <think>
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

---

## Full API Example

```python
import os
from openai import OpenAI
from markers import INNER_OS_MARKER, NO_INNER_OS_MARKER

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",   # or "deepseek-v4-pro"
        messages=messages,
    )
    return response.choices[0].message.content


# ── Immersive roleplay session ──────────────────────────────────────────────
system = "你是一个傲娇的女高中生，喜欢主角但绝不承认。"
first_user = "「我走进教室」"早上好。""

messages = build_messages(system, first_user, mode="inner_os")

# Round 1 — marker is injected automatically
reply1 = chat(messages)
print(reply1)

# Round 2+ — just append, NO marker needed
messages.append({"role": "assistant", "content": reply1})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply2 = chat(messages)   # ✅ first-turn marker still in history, still active
print(reply2)

# Round 3
messages.append({"role": "assistant", "content": reply2})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply3 = chat(messages)
print(reply3)
```

---

## Pure Analysis Mode Example

```python
# When you want structured, director-style planning in <think>
system = "你是一个神秘的侦探，冷静理性，逻辑严密。"
first_user = "「我推开侦探事务所的门」"我需要你帮我找一个人。""

messages = build_messages(system, first_user, mode="no_inner_os")
reply = chat(messages)
# <think> will contain: scene analysis, reply strategy, word count plan — NO inner monologue
```

---

## Web / App Usage (No Code)

Paste the marker at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

After that, type normally — no marker needed in subsequent messages.

---

## Expected Think Tag Output

**Immersive mode (`inner_os`):**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
回复方向：先冷淡，但身体语言要出卖内心。
</think>
```

**Pure analysis mode (`no_inner_os`):**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Common Patterns

### Helper class for multi-turn sessions

```python
class RoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-flash"):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})
        reply = chat(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = RoleplaySession("你是一个温柔的图书馆员...", mode="inner_os")
print(session.send("「我走进图书馆」"请问有推理小说吗？""))
print(session.send("「我接过书单」"谢谢，你真的很了解书。""))
```

### Switching modes — start a new session

```python
# To change mode mid-conversation, you MUST start a new session.
# The marker only works reliably when injected on the first user turn.

old_session = RoleplaySession(system, mode="inner_os")
new_session = RoleplaySession(system, mode="no_inner_os")  # fresh context
```

### Verify mode is active (check think tag content)

```python
def get_think_content(raw_response: str) -> str | None:
    """Extract <think>…</think> block from raw streamed or non-streamed output."""
    import re
    match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    return match.group(1).strip() if match else None

think = get_think_content(reply1)
has_inner_os = "（" in think or "心想" in think or "内心" in think
print("Immersive mode active:", has_inner_os)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Marker has no effect | Quick Mode on web, or marker not in first user message | Switch to Expert Mode; ensure marker is in turn 1 |
| Mode stops working mid-conversation | Marker was added to turn 2+ instead of turn 1 | Always inject on turn 1; start new session to fix |
| Inconsistent behavior | Probabilistic — not 100% guaranteed | Re-roll (send again); try `deepseek-v4-pro` for higher consistency |
| `<think>` tag missing entirely | Model or surface doesn't expose reasoning | Confirm you're using Expert Mode or API with reasoning enabled |
| Inner monologue appears in pure analysis mode | Context window drift on very long sessions | Summarize history and restart session with fresh first-turn marker |

**Key constraints to remember:**
- ✅ Marker in **first user message** → reliable
- ❌ Marker in **system prompt** → less reliable (not the trained injection position)
- ❌ Marker in **turn 2+** → will not activate the mode
- ⚠️ Probabilistic — if it doesn't trigger, send again
```
