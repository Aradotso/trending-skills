```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special control markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 roleplay instructions
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay marker injection
  - control deepseek think tag behavior
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes special control markers that can be injected into the **first user message** of a conversation to steer how the model thinks inside its `<think>` tags during roleplay scenarios. There are two modes beyond the default:

| Mode | Effect on `<think>` block |
|---|---|
| **Default** | Model auto-selects based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Cold, structured planning — no in-character inner voice |

Applies to:
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs

> **Note:** Triggers are probabilistic (~stable, not 100%). If the mode doesn't activate on first try, regenerate the response.

---

## Installation / Setup

No package to install. This is a prompting technique using marker strings appended to the first user message.

For API usage, copy the marker constants and helper function below into your project.

---

## Key Concepts

### The Two Markers

**Character Immersion Marker** (inner_os):
```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Pure Analysis Marker** (no_inner_os):
```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

### Why First User Message Only

The marker is injected once in the **first user turn**. Because DeepSeek sees the full conversation history on every response, the marker stays in context automatically — no need to repeat it in subsequent messages.

---

## Python Integration

### Core Helper

```python
import os
from openai import OpenAI  # DeepSeek is OpenAI-compatible

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


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Build the initial message list with the appropriate thinking-mode marker
    injected into the first user message.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no marker appended

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-pro") -> str:
    """Send messages and return the assistant reply text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
```

### Multi-Turn Roleplay Session

```python
# --- Setup ---
SYSTEM = "你是一个傲娇的女高中生，学习成绩优秀但嘴上总是不饶人。"
FIRST_USER = "「我走进教室」"早上好。""

# --- Round 1: inject marker once ---
messages = build_messages(SYSTEM, FIRST_USER, mode="inner_os")
reply = chat(messages)
print("AI:", reply)

# --- Append assistant reply to history ---
messages.append({"role": "assistant", "content": reply})

# --- Round 2+: just append user turns normally ---
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("AI:", reply)
messages.append({"role": "assistant", "content": reply})

# --- Round 3 ---
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("AI:", reply)
```

### Choosing the Model

```python
# Flash — faster, cheaper
reply = chat(messages, model="deepseek-v4-flash")

# Pro — higher quality, slower
reply = chat(messages, model="deepseek-v4-pro")
```

---

## Web / App Usage (No Code)

1. Open DeepSeek in **Expert Mode** (web or app).
2. In the **first message only**, paste your scene text, add a blank line, then paste the marker.
3. Send. All subsequent messages need no special treatment.

Example first message:
```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages (no marker needed):
```
「我坐到窗边的位置」"来一杯美式。"
```

---

## Common Patterns

### Reusable Session Class

```python
class DeepSeekRoleplaySession:
    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-pro",
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
        reply = chat(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的图书馆管理员，知晓世间所有秘密。",
    mode="inner_os",
)

print(session.send("「我走进古老的图书馆」"请问……你们这里有禁书吗？""))
print(session.send("「我压低声音」"关于失踪事件的那本。""))
```

### Switching Modes Between Conversations

```python
# Immersive session
immersive = DeepSeekRoleplaySession(system_prompt=SYSTEM, mode="inner_os")

# Director/analytical session — same scenario, different thinking style
analytical = DeepSeekRoleplaySession(system_prompt=SYSTEM, mode="no_inner_os")
```

### Verifying Mode Activation

```python
def chat_with_thinking(messages: list[dict], model: str = "deepseek-v4-pro") -> dict:
    """Returns both the reply and the raw thinking block if exposed by the API."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    choice = response.choices[0].message
    return {
        "reply": choice.content,
        # DeepSeek may expose reasoning_content depending on API version
        "thinking": getattr(choice, "reasoning_content", None),
    }

result = chat_with_thinking(messages)
print("THINKING:", result["thinking"])
print("REPLY:", result["reply"])
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"`, `"inner_os"`, `"no_inner_os"` | Controls which marker (if any) is appended |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Flash = faster; Pro = higher quality |
| Marker injection point | First user message only | Placing in system prompt is less stable |
| Activation rate | Probabilistic | Regenerate if mode doesn't activate |

---

## Troubleshooting

**Mode didn't activate (thinking still shows wrong style)**
- Regenerate the response — activation is probabilistic, not guaranteed on every attempt.
- Confirm the marker is in the **first user message**, not the system prompt.
- Confirm you are using **Expert Mode** on web, or the `deepseek-v4-flash` / `deepseek-v4-pro` API endpoint. Quick Mode on web does not support this.

**Marker text appears in the model's reply**
- The marker should only be appended to the user message content string, not sent as a separate message. Check that `build_messages` is concatenating correctly.

**Mode resets mid-conversation**
- The first user message (with marker) must remain in the message history. Do not truncate or summarize early messages in your history management — keep the first turn intact.

**Want to switch modes**
- Start a new conversation/session. Pass the new `mode` value to `build_messages` or `DeepSeekRoleplaySession`. You cannot change mode mid-conversation reliably.

**API key setup**
```bash
export DEEPSEEK_API_KEY="your-key-from-deepseek-platform"
```
```python
import os
api_key = os.environ["DEEPSEEK_API_KEY"]  # never hardcode
```
```
