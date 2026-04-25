```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special control instructions.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek roleplay control instructions
  - deepseek think tag roleplay
  - deepseek v4 flash pro roleplay api
  - deepseek pure analysis mode roleplay
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides **special control instructions** that shift how DeepSeek-V4 models reason inside their `<think>` tags during roleplay conversations. You can force the model into:

- **Character Immersion Mode** — first-person inner monologue wrapped in parentheses, like an actor in character
- **Pure Analysis Mode** — cold, structured reasoning like a director planning a scene
- **Default** — model chooses automatically based on scene complexity

Supported surfaces: DeepSeek official APP / web **Expert Mode**, and API models `deepseek-v4-flash` and `deepseek-v4-pro`.

---

## How It Works

The control instruction is appended to the **first user message only**. Because DeepSeek sends full conversation history on every turn, the instruction remains in context for the entire session — no need to repeat it.

Triggering is probabilistic (~stable, not 100%). If a mode doesn't activate, re-roll the response.

---

## The Two Markers (copy-ready)

### Character Immersion Mode (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Think-tag output looks like:**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

### Pure Analysis Mode (`no_inner_os`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

**Think-tag output looks like:**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Python API Integration

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
    Constructs the initial message list with the appropriate thinking marker.
    Only call this for the FIRST user turn. Subsequent turns append normally.
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

### Multi-turn Conversation Loop

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，不擅长表达自己的感情，但内心对主角充满好感。"

# --- Round 1: build with marker ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",   # or "deepseek-v4-pro"
    messages=messages,
)
reply = response.choices[0].message.content
print("Round 1:", reply)

# Append assistant reply to history
messages.append({"role": "assistant", "content": reply})

# --- Round 2+: just append user message normally ---
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
)
reply = response.choices[0].message.content
print("Round 2:", reply)
messages.append({"role": "assistant", "content": reply})
```

### Streaming Version

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def chat_stream(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    """Stream a response and return the full content string."""
    full_content = ""
    with client.chat.completions.stream(
        model=model,
        messages=messages,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_content += delta
    print()  # newline after stream ends
    return full_content

# First turn with immersion mode
messages = build_messages(
    system_prompt="你是一个神秘的酒吧调酒师，话不多，但观察力极强。",
    user_first_message="「我推开酒吧的门，在吧台坐下」"来一杯你推荐的。"",
    mode="inner_os",
)

reply = chat_stream(messages)
messages.append({"role": "assistant", "content": reply})
```

### Helper: Detect If Mode Activated

```python
import re

def check_mode_activated(think_content: str, mode: str) -> bool:
    """
    Inspect the <think> block to verify mode triggered.
    Requires the API to expose reasoning_content or you parse raw output.
    """
    if mode == "inner_os":
        # Look for parenthesized inner monologue
        return bool(re.search(r'[（(].+?[）)]', think_content))
    elif mode == "no_inner_os":
        # Should have NO parenthesized inner monologue
        return not bool(re.search(r'[（(].{2,}?[）)]', think_content))
    return True  # default always "activated"
```

---

## Web / Chat UI Usage

Paste the marker at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages require **no modification**:

```
Round 2: 「我坐到窗边的位置」"来一杯美式。"
Round 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Verify activation: click **"查看思考过程"** (View Thinking Process) in the UI.

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` | Pass to `build_messages()` |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support these markers |
| Marker position | End of first user message | Trained injection point — most stable here |
| System prompt | Any roleplay persona | Marker goes in user turn, not system |

### Environment Variables

```bash
DEEPSEEK_API_KEY=your_api_key_here   # Required — get from platform.deepseek.com
```

---

## Common Patterns

### Pattern 1: Reusable Session Class

```python
import os
from openai import OpenAI
from typing import Literal

class DeepSeekRoleplaySession:
    def __init__(
        self,
        system_prompt: str,
        mode: Literal["default", "inner_os", "no_inner_os"] = "inner_os",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
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

    def reset(self):
        system = self.messages[0]
        self.messages = [system]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一位冷酷的剑客，行走江湖多年，不轻易动情。",
    mode="inner_os",
)

print(session.chat("「一名年轻女子拦住了你的去路」"大侠，请留步。""))
print(session.chat("「女子递上一封信」"这是我师父托我转交的。""))
```

### Pattern 2: Switch Mode Between Sessions

```python
# Session A — immersive drama
drama_session = DeepSeekRoleplaySession(
    system_prompt="你是一个深情的古风书生。",
    mode="inner_os",
)

# Session B — structured creative writing assistance
writing_session = DeepSeekRoleplaySession(
    system_prompt="你是一个专业的网文写作助手，帮助用户规划剧情。",
    mode="no_inner_os",
)
```

### Pattern 3: Async Version

```python
import asyncio
import os
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_roleplay_turn(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        system_prompt="你是一个温柔的图书馆管理员。",
        user_first_message="「我走进图书馆，发现只有你一人」"不好意思，请问有没有关于量子力学的书？"",
        mode="inner_os",
    )
    reply = await async_roleplay_turn(messages)
    print(reply)

asyncio.run(main())
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Mode didn't activate | Probabilistic triggering | Re-roll / regenerate the response |
| Marker placed in system prompt | Wrong injection point | Move marker to end of first **user** message |
| Works on turn 1, breaks later | Marker not in history | Never needed — history preserves it automatically |
| Using web Quick Mode | Not supported | Switch to **Expert Mode** in the DeepSeek web UI |
| `inner_os` mode shows no parentheses | Model chose not to comply | Retry; or make first message more emotionally complex |
| API auth error | Wrong key or base URL | Ensure `DEEPSEEK_API_KEY` is set and `base_url="https://api.deepseek.com/v1"` |

### Verify Think Content via API

If the API returns `reasoning_content` separately (model-dependent):

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
)
choice = response.choices[0]

# Some DeepSeek endpoints expose reasoning separately
think_block = getattr(choice.message, "reasoning_content", None)
final_reply = choice.message.content

if think_block:
    activated = check_mode_activated(think_block, mode="inner_os")
    print("Mode activated:", activated)
    print("Think:", think_block[:200])

print("Reply:", final_reply)
```

---

## Quick Reference

```python
# Minimal working example
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)

messages = [
    {"role": "system", "content": "你是一个傲娇的女高中生。"},
    {"role": "user",   "content": "「我走进教室」"早上好。"" + INNER_OS_MARKER},
]

r = client.chat.completions.create(model="deepseek-v4-flash", messages=messages)
print(r.choices[0].message.content)
```
```
