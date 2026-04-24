```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue control
  - deepseek v4 roleplay instruct
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek think tag control
  - deepseek roleplay api markers
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control markers injected into the **first user message** to switch how the model reasons inside its `<think>` tags during roleplay. Two modes are available beyond the default: **character immersion** (first-person inner monologue) and **pure analysis** (cold director-style planning). This applies to `deepseek-v4-flash`, `deepseek-v4-pro` APIs, and the official DeepSeek app/web in Expert Mode.

---

## How It Works

- The model reads its full conversation history on every turn.
- A marker appended to the **first user message** persists in context for the entire session.
- The marker steers the `<think>` block style — it does **not** alter the final visible reply, only the reasoning process.
- Trigger probability is high but not 100% — retry if the first attempt doesn't produce the desired format.

---

## The Three Modes

| Mode | How to activate | `<think>` behavior |
|---|---|---|
| **Default** | Nothing added | Model picks automatically based on complexity |
| **Character Immersion** | Append `INNER_OS_MARKER` to first user message | Parenthesized first-person inner monologue |
| **Pure Analysis** | Append `NO_INNER_OS_MARKER` to first user message | Logical analysis only, no inner monologue |

---

## Marker Strings (Copy-Ready)

### Character Immersion Marker

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker

```
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
def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"  # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking-mode marker.
    Only call this for the FIRST user message. Subsequent turns are plain appends.
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

### Full Session Example (OpenAI-compatible client)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # adjust to actual DeepSeek endpoint
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，名叫小雪，坐在教室窗边。"

# --- Turn 1: inject marker ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",  # character immersion
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
reply = response.choices[0].message.content
print("Assistant:", reply)

# --- Turn 2+: plain append, marker stays in history automatically ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
reply2 = response.choices[0].message.content
print("Assistant:", reply2)
```

### Async Version

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_session(system: str, opening: str, mode: str = "inner_os"):
    messages = build_messages(system, opening, mode=mode)
    turns = []

    async def send(user_msg: str | None = None) -> str:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        resp = await client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
        )
        content = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    # First turn uses the pre-built messages (opening already inside)
    reply = await send()
    turns.append(reply)
    return turns, send  # return `send` to continue the session


async def main():
    _, send = await roleplay_session(
        system="你是一个傲娇的咖啡师。",
        opening="「我推开咖啡店的门」"请问还有位置吗？"",
        mode="inner_os",
    )
    print(await send("「我坐到窗边」"来一杯美式。""))

asyncio.run(main())
```

---

## Streaming Support

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    "你是一个神秘侦探。",
    "「我走进你的办公室」"我需要帮助。"",
    mode="no_inner_os",  # pure analysis mode
)

stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

---

## Web / App Usage

Paste the marker after a blank line at the end of your **first message only**:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are plain text — the marker persists in history automatically.

---

## Session Manager Class

```python
from dataclasses import dataclass, field
from typing import Literal
import os
from openai import OpenAI

ThinkingMode = Literal["default", "inner_os", "no_inner_os"]

@dataclass
class DeepSeekRoleplaySession:
    system_prompt: str
    mode: ThinkingMode = "default"
    model: str = "deepseek-v4-pro"
    messages: list[dict] = field(default_factory=list)
    _started: bool = field(default=False, init=False)

    def __post_init__(self):
        self._client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def send(self, user_message: str) -> str:
        if not self._started:
            # First turn: inject marker and system prompt
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self._started = True

        self.messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.messages = []
        self._started = False


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的图书馆员。",
    mode="inner_os",
)

print(session.send("「我走到借阅台」"你好，我想借这本书。""))
print(session.send("「我微笑」"谢谢你。""))

# Switch mode: reset and change
session.reset()
session.mode = "no_inner_os"
print(session.send("「我走进图书馆」"请问哪里有历史类书籍？""))
```

---

## Thinking Output Comparison

```
# Character Immersion Mode (inner_os)
<think>
（他走进来了……为什么心跳有点快。）
好，要装作若无其事地回答他。
（不能让他看出来我注意到他了！）
回复策略：语气冷淡，但给出正确指引，动作描写暗示紧张。
</think>

# Pure Analysis Mode (no_inner_os)
<think>
场景：用户进入图书馆询问历史书籍位置。角色设定：傲娇图书馆员。
回复策略：语气略带不耐烦，提供准确信息，加入轻微肢体语言描写。
字数控制：120字以内，动作+对话结构。
</think>
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key |

---

## Troubleshooting

**Marker didn't activate the expected mode**
- The trigger probability is not 100%. Retry the generation (regenerate/resend).
- Confirm the marker is in the **first user message**, not in `system` or a later turn.
- Ensure you're using Expert Mode on web, or `deepseek-v4-flash`/`deepseek-v4-pro` on API. Quick Mode on web is not supported.

**Mode active in turn 1 but faded later**
- This should not happen — the first message with the marker is always in context.
- If using a context-trimming setup, ensure the first message is never evicted from the window.

**Marker in system prompt doesn't work**
- By design, the marker must go in the first **user** message. The model was trained with markers in that position.

**Switching modes mid-session**
- Start a new conversation/session. The marker is tied to the first user message; you cannot swap modes in an existing session without resetting.

**API base URL**
- Confirm the correct endpoint with DeepSeek's official documentation, as it may differ from the OpenAI default (`https://api.openai.com/v1`). Use `base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")` for flexibility.
```
