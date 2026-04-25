```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special control instructions
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue instruct
  - deepseek v4 roleplay control
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay api instructions
  - deepseek think tag roleplay
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions appended to the **first user message** to steer how the model thinks inside `<think>` tags during roleplay. Two modes are available beyond the default: **角色沉浸 (Character Immersion)** — the model thinks as the character using first-person inner monologue — and **纯分析 (Pure Analysis)** — the model thinks as a detached director/analyst with no in-character inner voice.

---

## What It Does

| Mode | Trigger | `<think>` Behavior |
|---|---|---|
| **Default** | Nothing added | Model auto-selects based on scene complexity |
| **Character Immersion** | Append `INNER_OS_MARKER` to first user message | First-person inner monologue wrapped in parentheses |
| **Pure Analysis** | Append `NO_INNER_OS_MARKER` to first user message | Pure logical planning, no in-character voice |

**Supported surfaces:**
- DeepSeek official APP / web (Expert Mode only — not Quick Mode)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> **Note:** Instructions are probabilistic — not guaranteed 100% trigger. Re-roll if the first attempt doesn't produce the desired format.

---

## Installation / Setup

No package to install. This is a prompting technique. For API use, set your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

Install the official client (or use `openai` SDK with base URL override):

```bash
pip install openai
```

---

## Key Constants

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

## Core API Usage Pattern

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


def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking mode marker.

    Args:
        system_prompt: The character/scenario system prompt
        user_first_message: The first user turn content
        mode: "inner_os" | "no_inner_os" | "default"

    Returns:
        List of message dicts ready for the API
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Round 1: marker injected automatically ──────────────────────────────────
messages = build_messages(
    system_prompt="你是一个傲娇的女高中生，内心其实很喜欢男主角，但表面总是冷漠相待。",
    user_first_message="「我走进教室，向你挥了挥手」"早上好。"",
    mode="inner_os",
)
reply = chat(messages)
print(reply)

# ── Round 2+: append normally, marker stays in history ──────────────────────
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
    The mode marker is injected once into the first user message and persists
    automatically via conversation history.
    """

    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self._first_turn = True

        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def send(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        full_messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.messages
            + [{"role": "user", "content": user_message}]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
        )
        assistant_reply = response.choices[0].message.content

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def reset(self):
        self.messages = []
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的咖啡馆老板，隐藏着一段不为人知的过去。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「我推开咖啡店的门」"你好，请问还有位置吗？""))
print(session.send("「我坐到窗边」"来一杯美式。""))
print(session.send("「我注意到你手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / App Usage (No Code)

Paste the full instruction block at the end of your **first message only**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

After that, send all subsequent messages normally — no marker needed again.

---

## Expected Think-Tag Output

**Character Immersion (`inner_os`) mode:**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

**Pure Analysis (`no_inner_os`) mode:**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` | Controls which marker (if any) is appended |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support thinking mode instructions |
| Injection position | End of **first** user message | Training-aligned position — most reliable |
| System prompt position | Standard `system` role | Do NOT put the marker here |

---

## Troubleshooting

**Marker didn't trigger the expected format:**
- Re-run: the trigger is probabilistic, not deterministic. Multiple attempts usually succeed.
- Ensure you're using **Expert Mode** on web (not Quick Mode).
- Confirm the marker is in the **first user turn**, not the system prompt.
- Confirm you're using `deepseek-v4-flash` or `deepseek-v4-pro` — other model variants are not supported.

**Marker affects visible reply text:**
- It shouldn't. The instructions target only `<think>` content. If bleed-through occurs, try rephrasing your system prompt to reinforce the character voice.

**Want to switch modes mid-conversation:**
- Open a new conversation/session. Inject the new marker in the first message of the fresh session.

**API returns no `<think>` block:**
- Thinking (reasoning) mode must be enabled on your API request. Check DeepSeek API docs for the `reasoning_effort` or equivalent parameter if available on your plan.

---

## Key Principles

1. **Inject once, works forever** — the marker sits in conversation history and is visible to the model on every subsequent turn.
2. **Marker position matters** — end of first user message mirrors training data injection point.
3. **Think ≠ Reply** — markers only reshape the `<think>` reasoning process; final reply quality improves indirectly through better reasoning alignment.
4. **Probabilistic, not deterministic** — build retry logic in production if consistent format adherence is required.
```
