```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 character immersion
  - switch deepseek thinking style
  - deepseek roleplay instruct markers
  - deepseek chain of thought roleplay
  - control deepseek think tags
  - deepseek v4 flash pro roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct is a prompt-engineering technique that injects special control markers into the **first user message** of a conversation to steer how the model's `<think>` block behaves during roleplay. Two modes exist beyond the default:

- **角色沉浸 (Character Immersion)** — the model thinks in first-person inner monologue inside `<think>`, like an actor in character.
- **纯分析 (Pure Analysis)** — the model thinks analytically as a director/writer, no in-character inner voice.

**Supported surfaces:**
- DeepSeek official APP / Web — **Expert Mode** only
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> Web **Quick Mode** is not supported.

---

## How It Works

The marker is appended (with a blank line separator) to the **first** user message only. Because the model always sees the full conversation history, the marker remains in context for every subsequent turn automatically — no need to repeat it.

---

## Marker Strings

### Character Immersion Mode (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode (`no_inner_os`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python Integration

### Constants and Builder

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

MARKERS = {
    "inner_os": INNER_OS_MARKER,
    "no_inner_os": NO_INNER_OS_MARKER,
    "default": "",
}


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking-mode marker
    injected into the first user turn.

    Args:
        system_prompt: The character/scenario system prompt.
        user_first_message: The player's opening action/dialogue.
        mode: One of "inner_os", "no_inner_os", or "default".

    Returns:
        A messages list ready to send to the DeepSeek API.
    """
    marker = MARKERS.get(mode, "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message + marker},
    ]
```

### Full Multi-Turn Example

```python
import os
from openai import OpenAI  # DeepSeek is OpenAI-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-flash"  # or "deepseek-v4-pro"

SYSTEM_PROMPT = (
    "你是一个傲娇的女高中生，名叫晴子。"
    "你表面冷淡、嘴硬，内心却很在意对方。"
    "请用符合角色性格的语气和动作描写回复。"
)

def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Turn 1: inject marker once ──────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好，晴子。"",
    mode="inner_os",          # character immersion
)
reply = chat(messages)
print("[晴子]", reply)

# ── Turn 2+: append normally, marker persists in history ────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("[晴子]", reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("[晴子]", reply)
```

### Switching Modes

To switch modes, start a **new conversation** with the other marker:

```python
# Pure analysis mode — director's perspective
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好，晴子。"",
    mode="no_inner_os",
)
reply = chat(messages)
```

---

## Web Usage (One-Step)

Paste the marker at the end of your **first** message in the input box, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no marker:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Verify by clicking **"查看思考过程"** (View Thinking Process).

---

## Reusable Session Class

```python
import os
from openai import OpenAI
from typing import Literal

ThinkingMode = Literal["inner_os", "no_inner_os", "default"]


class DeepSeekRoleplaySession:
    """Stateful multi-turn roleplay session with thinking-mode control."""

    def __init__(
        self,
        system_prompt: str,
        mode: ThinkingMode = "default",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._first_turn = True
        self._client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def send(self, user_message: str) -> str:
        content = user_message
        if self._first_turn and self.mode != "default":
            content += MARKERS[self.mode]
            self._first_turn = False

        self._messages.append({"role": "user", "content": content})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages,
        )
        assistant_content = response.choices[0].message.content
        self._messages.append({"role": "assistant", "content": assistant_content})
        return assistant_content

    def reset(self, mode: ThinkingMode | None = None) -> None:
        """Start a fresh conversation, optionally changing the mode."""
        if mode is not None:
            self.mode = mode
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个博学多才的魔法师，说话文雅而神秘。",
    mode="inner_os",
)

print(session.send("「我走进魔法塔」"师傅，我来了。""))
print(session.send("「我从包里取出一本残缺的古籍」"这本书……你认识吗？""))

# Switch to pure analysis for a new arc
session.reset(mode="no_inner_os")
print(session.send("「新场景开始」"有人闯入了魔法塔！""))
```

---

## Mode Comparison

| | Default | `inner_os` (Character Immersion) | `no_inner_os` (Pure Analysis) |
|---|---|---|---|
| `<think>` style | Auto-selected | First-person inner monologue in `（）` | Analytical, third-person planning |
| Best for | General use | Emotional depth, character consistency | Structural control, complex plots |
| Final reply affected? | — | More emotionally authentic | More structurally stable |
| Trigger reliability | — | Probabilistic (~stable) | Probabilistic (~stable) |

---

## Common Patterns

### Pattern 1 — Inject via system prompt wrapper (less reliable)

The recommended injection point is the **first user message**. Putting the marker in the system prompt is less effective because it doesn't match the training injection position:

```python
# ✅ Recommended — first user message
messages = build_messages(system_prompt, first_user_msg, mode="inner_os")

# ⚠️  Less reliable — system prompt
messages = [
    {"role": "system", "content": system_prompt + INNER_OS_MARKER},
    {"role": "user", "content": first_user_msg},
]
```

### Pattern 2 — Streaming with thinking mode

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    "你是一个冷酷的侦探。",
    "「我把一张照片推到桌上」"认识这个人吗？"",
    mode="no_inner_os",
)

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Pattern 3 — Multi-session manager

```python
sessions: dict[str, DeepSeekRoleplaySession] = {}

def get_or_create_session(
    session_id: str,
    system_prompt: str,
    mode: ThinkingMode = "inner_os",
) -> DeepSeekRoleplaySession:
    if session_id not in sessions:
        sessions[session_id] = DeepSeekRoleplaySession(system_prompt, mode)
    return sessions[session_id]
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Marker has no effect | Mode not triggered (probabilistic) | Re-roll (send again); the technique increases probability but isn't 100% |
| Marker works on turn 1 but fades | Context window too long | The first message is still in history; check if context was truncated |
| Used Quick Mode on web | Quick Mode unsupported | Switch to **Expert Mode** in the DeepSeek web interface |
| Placed marker in system prompt | Wrong injection position | Move marker to end of **first user message** |
| Want to change mode mid-conversation | Markers only configure once | Start a new conversation with the desired marker |
| API returns no `<think>` block | Model variant doesn't support thinking | Confirm you're using `deepseek-v4-flash` or `deepseek-v4-pro` |

---

## Environment Setup

```bash
# Required
export DEEPSEEK_API_KEY="your-api-key-here"

# Install the OpenAI-compatible client
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```
```
