```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 roleplay instructions
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay prompt markers
  - deepseek think tag control
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes special control markers you can inject into the **first user message** of a conversation to steer how the model *thinks* inside its `<think>` tags during roleplay. Two modes are available:

| Mode | Effect |
|------|--------|
| **角色沉浸 (Character Immersion)** | `<think>` contains first-person inner monologue wrapped in parentheses |
| **纯分析 (Pure Analysis)** | `<think>` contains only cold logical planning — no character voice |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`

> Web "Quick Mode" does **not** support this feature.

---

## The Marker Strings

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

## Installation / Setup

No package to install. Clone or copy the markers directly:

```bash
git clone https://github.com/victorchen96/deepseek_v4_rolepaly_instruct
```

Or just copy the marker strings above into your project.

Set your DeepSeek API key as an environment variable:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

---

## Core Helper: `build_messages`

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

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
    Build the initial messages list with the appropriate thinking-mode marker.

    Args:
        system_prompt: The character/scenario system prompt.
        user_first_message: The player's opening action/line.
        mode: "inner_os"    → character immersion (inner monologue)
              "no_inner_os" → pure analysis (no inner voice)
              "default"     → let the model decide

    Returns:
        A messages list ready for the DeepSeek API.
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

## Full Working Example

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
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


def build_messages(system_prompt, user_first_message, mode="default"):
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


def chat(messages, model="deepseek-v4-flash"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# --- Round 1: inject the marker only here ---
system = "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。"
messages = build_messages(
    system_prompt=system,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # or "no_inner_os" or "default"
)
reply = chat(messages)
print("Round 1:", reply)

# --- Round 2+: just append normally, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Round 2:", reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Round 3:", reply)
```

---

## Multi-turn Conversation Manager

```python
class RoleplaySession:
    """Manages a DeepSeek roleplay session with a fixed thinking mode."""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-flash",
    ):
        self.client = client
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

session = RoleplaySession(
    client=client,
    system_prompt="你是一名神秘的侦探，擅长观察人心。",
    mode="no_inner_os",       # pure analysis mode
    model="deepseek-v4-pro",
)

print(session.send("「我推开侦探事务所的门」"我需要你的帮助。""))
print(session.send("「我放下一个信封」"这里有些照片，很奇怪。""))
```

---

## Web / App Usage (No Code)

Paste the marker at the end of your **first message only**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages need no special formatting — the marker persists in the conversation history automatically.

---

## Expected Think-Tag Output

**Character Immersion (`inner_os`):**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

**Pure Analysis (`no_inner_os`):**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

To verify the mode is active in the web UI: click **"查看思考过程"** (View thinking process).

---

## Configuration Reference

| Parameter | Values | Notes |
|-----------|--------|-------|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Only applied to first user message |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support thinking mode control |
| Marker injection position | End of first `user` message | Training-aligned position; system prompt placement is less reliable |

---

## Troubleshooting

**Marker didn't take effect / mode didn't change:**
- Retry — the effect is probabilistic, not guaranteed 100%. Re-roll the same message.
- Confirm you're on `deepseek-v4-flash` or `deepseek-v4-pro` (not a quick/lite model).
- Confirm the marker is in the **first user turn**, not the system prompt and not a later turn.
- On the web, confirm you're in **Expert Mode** (专家模式), not Quick Mode.

**Want to switch modes mid-conversation:**
- You cannot switch mid-session. Start a new conversation and paste the other marker in the first message.

**Marker in system prompt doesn't work reliably:**
- Per the training design, inject into the first `user` message, not `system`. System prompt placement is explicitly less stable.

**API base URL:**
```python
base_url="https://api.deepseek.com/v1"
```

**Both markers show up in context — conflict:**
- Use only one marker per conversation. The `build_messages` helper enforces this by appending only one based on `mode`.
```
