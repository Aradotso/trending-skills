```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking chain style in roleplay scenarios using special instruction markers for immersive character mode or pure analysis mode.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue roleplay
  - deepseek v4 character immersion
  - switch deepseek thinking style
  - deepseek roleplay instruction marker
  - deepseek analysis mode roleplay
  - deepseek think tag roleplay control
  - deepseek v4 flash roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 supports special control instructions injected into the first user message to influence how the model thinks inside its `<think>` tags during roleplay. Two modes are available:

- **角色沉浸 (Character Immersion)**: The model's reasoning contains first-person inner monologue as the character, wrapped in parentheses.
- **纯分析 (Pure Analysis)**: The model's reasoning is cold, structured, director-style planning — no in-character inner voice.

These markers are appended to the **first user message only**. Because DeepSeek sees full conversation history each turn, the instruction persists automatically throughout the session.

**Supported surfaces:**
- DeepSeek official APP / web (Expert Mode / 专家模式)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

**Not supported:** Web Quick Mode (快速模式).

> Note: Trigger rate is probabilistic, not 100%. Re-roll if the first response doesn't reflect the expected format.

---

## Instruction Markers (Copy-Ready)

### Character Immersion Mode (角色沉浸)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode (纯分析)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Quick Visual Comparison

```
Character Immersion Mode               Pure Analysis Mode
──────────────────────────────         ──────────────────────────────
<think>                                <think>
（他跟我打招呼了……心跳加速。）            场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。                 回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）               控制 150 字，先动作描写再对话。
</think>                               </think>
```

---

## Web Usage (1-Step Setup)

Paste the marker at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no modification:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Click **"查看思考过程"** to verify the mode is active.

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
    mode: str = "default"  # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking mode marker.

    Args:
        system_prompt: Character/scene description for the system role.
        user_first_message: The opening user action or dialogue.
        mode: "inner_os" for character immersion, "no_inner_os" for pure analysis,
              "default" for model auto-selection.

    Returns:
        A messages list ready to send to the DeepSeek API.
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

### Full Multi-Turn Example

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-flash"  # or "deepseek-v4-pro"

SYSTEM_PROMPT = (
    "你是一个傲娇的女高中生，表面冷漠，内心其实很在意对方。"
    "回复时用动作描写+对话的格式，控制在150字以内。"
)

# Turn 1 — inject marker once
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print("Turn 1:", assistant_reply)

# Turn 2+ — append normally, marker stays in history
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print("Turn 2:", assistant_reply)
```

### Accessing the Think Block

```python
# If the API returns reasoning_content separately:
choice = response.choices[0]
if hasattr(choice.message, "reasoning_content"):
    print("=== THINK ===")
    print(choice.message.reasoning_content)
print("=== REPLY ===")
print(choice.message.content)
```

---

## Common Patterns

### Session Manager Class

```python
class DeepSeekRoleplaySession:
    """Manages a multi-turn roleplay session with a persistent thinking mode."""

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        model: str = "deepseek-v4-flash",
        mode: str = "inner_os",
    ):
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self._system_prompt = system_prompt
        self._mode = mode
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            self.messages = build_messages(
                self._system_prompt, user_message, mode=self._mode
            )
            self._first_turn = False
        else:
            self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, new_mode: str | None = None):
        """Start a fresh session, optionally switching mode."""
        self.messages = []
        self._first_turn = True
        if new_mode:
            self._mode = new_mode


# Usage
session = DeepSeekRoleplaySession(
    client=client,
    system_prompt="你是一个神秘的图书馆守护者，博学多识，说话喜欢引用典故。",
    mode="inner_os",
)

print(session.send("「我推开图书馆的大门」"请问这里有关于时间旅行的书吗？""))
print(session.send("「我跟着他走向书架深处」"你在这里工作多久了？""))
```

### Switching Modes Between Sessions

```python
# Start a new session for pure analysis mode
session.reset(new_mode="no_inner_os")
print(session.send("「新场景开始」"你好。""))
```

### Async Version

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_turn(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        "你是一个古代剑客，性格洒脱，重义轻利。",
        "「我拦住你的去路」"留步，此路不通。"",
        mode="inner_os",
    )
    reply = await roleplay_turn(messages)
    print(reply)

asyncio.run(main())
```

---

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mode` | `"inner_os"` | Character immersion — parenthesized inner monologue in `<think>` |
| `mode` | `"no_inner_os"` | Pure analysis — structured planning only in `<think>` |
| `mode` | `"default"` | No marker appended; model decides automatically |
| Marker position | End of **first user message** | Training-aligned injection point; most stable |
| `model` | `"deepseek-v4-flash"` | Faster, lower cost |
| `model` | `"deepseek-v4-pro"` | Higher quality reasoning |
| API base URL | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| Auth | `DEEPSEEK_API_KEY` env var | Never hardcode keys |

---

## Troubleshooting

### Mode didn't activate on first try
The trigger is probabilistic. Regenerate the response (re-roll). Success rate increases with retries.

### Marker placed in system prompt — not working well
Move the marker to the end of the first **user** message. This matches the model's training injection point and is significantly more stable.

### Inner monologue appearing in final reply (not just `<think>`)
This is a model behavior issue unrelated to the marker. Add explicit instruction to system prompt: `"最终回复中不要出现括号内心戏，只在思考过程中使用。"`

### Mode bleeds into wrong turns
It shouldn't — the marker in turn 1 persists in context. If behavior drifts in very long sessions, you can re-append the marker to a later user message as a reinforcement.

### Using Quick Mode on web — nothing happens
Quick Mode (快速模式) does not support `<think>` tag control. Switch to **Expert Mode (专家模式)** in settings.

### API returns no `reasoning_content` field
Some API tiers or versions may not expose the thinking block separately. Check the raw response object: `response.choices[0].model_dump()` to inspect all available fields.

---

## Environment Setup

```bash
# Set your API key
export DEEPSEEK_API_KEY="your-key-here"

# Install the OpenAI-compatible client
pip install openai
```

```python
# Minimal working setup
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```
```
