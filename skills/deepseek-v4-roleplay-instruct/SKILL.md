```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking-chain style in roleplay scenarios using special injected instructions for immersive character or pure-analysis modes
triggers:
  - "add deepseek roleplay thinking mode"
  - "switch deepseek to character immersion mode"
  - "use deepseek pure analysis mode for roleplay"
  - "inject deepseek v4 roleplay marker"
  - "control deepseek thinking chain style"
  - "deepseek inner monologue roleplay"
  - "set up deepseek roleplay api messages"
  - "deepseek v4 role play instruct setup"
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions injected into the **first user message** to steer the model's internal `<think>` block behavior during roleplay. Two modes are available beyond the default:

- **角色沉浸 (Character Immersion)**: The `<think>` block reads like an actor's in-character inner monologue.
- **纯分析 (Pure Analysis)**: The `<think>` block reads like a director's cold, structured planning — no inner monologue.

**Supported surfaces:**
- DeepSeek official app / web — **Expert Mode** only
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web **Quick Mode** is NOT supported

---

## How It Works

The marker is appended to the **first user message only**. Because the model always sees full conversation history, the instruction persists across all subsequent turns automatically. You never need to repeat it.

---

## The Markers (Copy-Ready)

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

## API Usage

### Full Working Example

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

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
    mode: str = "default"
) -> list[dict]:
    """
    Build the initial messages list with optional roleplay marker injected.

    Args:
        system_prompt: Character/scene setup for the system role.
        user_first_message: The first user turn content.
        mode: One of "default", "inner_os", "no_inner_os".

    Returns:
        List of message dicts ready for the chat API.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — no marker appended

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-pro") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Round 1: inject marker once ──────────────────────────────────────────────
system = "你是一个傲娇的女高中生，班级里最不善表达感情的那种。"
messages = build_messages(
    system_prompt=system,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # or "no_inner_os" / "default"
)
reply = chat(messages)
print(reply)

# ── Round 2+: append normally, marker persists in history ────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply2 = chat(messages)
print(reply2)
```

---

## Multi-Turn Session Helper

```python
class DeepSeekRoleplaySession:
    """Manages a stateful multi-turn DeepSeek roleplay conversation."""

    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
        )
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._mode = mode
        self._first_turn_done = False

    def send(self, user_message: str) -> str:
        if not self._first_turn_done:
            if self._mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self._mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn_done = True

        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的女高中生。",
    mode="inner_os",
    model="deepseek-v4-flash",   # cheaper/faster variant
)

print(session.send("「我推开咖啡店的门」"你好，请问还有位置吗？""))
print(session.send("「我坐到窗边」"来一杯美式。""))
print(session.send("「我注意到你手上有疤痕」"你的手……没事吧？""))
```

---

## Web / App Usage

Paste the marker at the **end of your very first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are written normally — no marker needed again.

---

## Mode Comparison

| | Default | inner_os (角色沉浸) | no_inner_os (纯分析) |
|---|---|---|---|
| `<think>` style | Auto-selected | First-person inner monologue in `（）` | Pure logical analysis, no brackets |
| Best for | General use | Emotional, immersive roleplay | Structured, plot-driven stories |
| Marker location | None | End of first user message | End of first user message |

**Thinking output examples:**

```
# inner_os mode
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>

# no_inner_os mode
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制150字，先动作描写再对话。
</think>
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `model` | `deepseek-v4-flash`, `deepseek-v4-pro` | Flash = faster/cheaper; Pro = higher quality |
| `mode` | `"default"`, `"inner_os"`, `"no_inner_os"` | Only affects `<think>` block behavior |
| Marker position | End of first user message | Placing in system prompt is less stable |
| `DEEPSEEK_API_KEY` | env var | Never hardcode |
| `base_url` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |

---

## Troubleshooting

**Marker didn't take effect (thinking style unchanged)**
- This is probabilistic — trigger rate is not 100%. Regenerate the response (roll again).
- Confirm you are using **Expert Mode** on web, not Quick Mode.
- Confirm the marker is in the **first user message**, not the system prompt.
- Confirm the model is `deepseek-v4-flash` or `deepseek-v4-pro`.

**Want to switch modes mid-conversation**
- Start a **new conversation**. The marker from round 1 is baked into history and cannot be overridden mid-session cleanly.

**inner_os marker affects the final reply content**
- Expected behavior: immersive thinking leads to more emotionally authentic replies. This is by design.

**Using async client**

```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages):
    response = await async_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    return response.choices[0].message.content
```

**Verifying mode on web**
- Click "查看思考过程" (View Thinking Process) after the model responds to inspect the `<think>` block style.
```
