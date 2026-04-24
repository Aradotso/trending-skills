```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning via special prompt markers
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion prompt
  - switch deepseek thinking style
  - deepseek roleplay instruct marker
  - deepseek think tag control
  - deepseek v4 roleplay api
  - deepseek inner os mode
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides **special control markers** that influence how DeepSeek-V4 models reason inside their `<think>` tags during roleplay. By appending a marker to the first user message, you steer the model between two styles:

| Mode | Thinking Style |
|---|---|
| **Default** | Model auto-selects based on scene complexity |
| **角色沉浸 (Character Immersion)** | First-person inner monologue wrapped in parentheses |
| **纯分析 (Pure Analysis)** | Cold, director-style logic — no inner monologue |

**Supported surfaces:**
- DeepSeek official APP / web **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- ⚠️ Web **Quick Mode** is NOT supported

---

## How It Works

The marker is appended **once** to the first user message. Because the model always receives full conversation history, the marker remains in context for all subsequent turns — no need to repeat it.

---

## The Two Markers (Copy-Ready)

### Character Immersion Marker (角色沉浸)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker (纯分析)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Expected Thinking Output

```
# Character Immersion Mode          # Pure Analysis Mode
<think>                             <think>
（他跟我打招呼了……心跳加速。）        场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。             回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）           控制 150 字，先动作描写再对话。
</think>                            </think>
```

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
    Construct the initial message list for a roleplay session.

    Args:
        system_prompt: Character/scenario description for the system role.
        user_first_message: The first user turn content.
        mode: Thinking style — "inner_os", "no_inner_os", or "default".

    Returns:
        List of message dicts ready to pass to the DeepSeek API.
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
```

### Full Session Example (openai-compatible client)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，外冷内热，嘴上不饶人，心里却很在乎对方。"

# ── Turn 1: build with marker ──────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # or "no_inner_os" / "default"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
reply = response.choices[0].message.content
print(reply)

# ── Turn 2+: append normally, marker stays in history ─────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,        # marker from turn 1 is still active
)
reply2 = response.choices[0].message.content
print(reply2)

# ── Continue appending for as many turns as needed ────────────────────────
messages.append({"role": "assistant", "content": reply2})
messages.append({"role": "user", "content": "「我注意到你手上有一道疤痕」"你的手……没事吧？""})
```

### Async Version

```python
import asyncio
import os
from openai import AsyncOpenAI

async def roleplay_session(system_prompt: str, turns: list[str], mode: str = "inner_os"):
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )

    messages = build_messages(system_prompt, turns[0], mode=mode)
    replies = []

    # First turn
    resp = await client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )
    reply = resp.choices[0].message.content
    replies.append(reply)
    messages.append({"role": "assistant", "content": reply})

    # Subsequent turns
    for user_msg in turns[1:]:
        messages.append({"role": "user", "content": user_msg})
        resp = await client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
        )
        reply = resp.choices[0].message.content
        replies.append(reply)
        messages.append({"role": "assistant", "content": reply})

    return replies

# Usage
replies = asyncio.run(roleplay_session(
    system_prompt="你是一个神秘的咖啡店老板……",
    turns=[
        "「我推开咖啡店的门」"你好，请问还有位置吗？"",
        "「我坐到窗边的位置」"来一杯美式。"",
    ],
    mode="inner_os",
))
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

After that, all subsequent messages are plain — no marker needed:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Click **"查看思考过程"** (View Thinking Process) to verify the mode is active.

---

## Mode Selection Guide

| Scenario | Recommended Mode |
|---|---|
| Emotional / romance roleplay | `inner_os` — richer character feelings |
| Structured narrative writing | `no_inner_os` — stable plot planning |
| Mystery / suspense scenes | `inner_os` — suspense through hidden thoughts |
| Debug / inspect model reasoning | `no_inner_os` — clean logical output |
| Unknown / general use | `default` — model decides |

---

## Common Patterns

### Pattern 1: Reusable Session Class

```python
import os
from openai import OpenAI

class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-pro"):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.system_prompt = system_prompt
        self.mode = mode
        self.messages: list[dict] = []
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self.messages = [{"role": "system", "content": self.system_prompt}]
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
        self.messages = []
        self._first_turn = True

# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一位冷静的侦探……",
    mode="no_inner_os",
)
print(session.chat("「一具尸体出现在密室中央」"开始调查。""))
print(session.chat("「我检查了窗锁」"这扇窗从里面锁上的。""))
```

### Pattern 2: Inject Marker into Existing History

```python
def inject_marker_into_history(
    messages: list[dict],
    mode: str,
) -> list[dict]:
    """
    Retroactively add a marker to the first user message in an existing history.
    Useful when upgrading a plain conversation to a controlled-mode session.
    """
    result = []
    first_user_patched = False
    for msg in messages:
        if msg["role"] == "user" and not first_user_patched:
            patched = dict(msg)
            if mode == "inner_os":
                patched["content"] = msg["content"] + INNER_OS_MARKER
            elif mode == "no_inner_os":
                patched["content"] = msg["content"] + NO_INNER_OS_MARKER
            result.append(patched)
            first_user_patched = True
        else:
            result.append(msg)
    return result
```

### Pattern 3: Streaming Response

```python
import os
from openai import OpenAI

def stream_roleplay(messages: list[dict], model: str = "deepseek-v4-pro"):
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_reply += delta
    print()
    return full_reply
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Controls which marker is appended |
| `model` | `"deepseek-v4-pro"` / `"deepseek-v4-flash"` | Both support the markers |
| Marker position | End of first user message | Training-time injection point — most stable |
| System prompt | Any | Place character description here; marker goes in user turn |

---

## Troubleshooting

**Marker not taking effect (thinking still looks wrong)**
- This is probabilistic — trigger rate is not 100%. Re-roll (send again) 1–3 times.
- Ensure you're using **Expert Mode** on web, not Quick Mode.
- Confirm the marker is in the **first user message**, not system prompt.
- Verify the model is `deepseek-v4-flash` or `deepseek-v4-pro`.

**Should I put the marker in the system prompt?**
- No. The training injection point is the first user message. System prompt placement is less stable.

**Does the marker affect the final reply text?**
- Only indirectly. The marker controls the `<think>` reasoning style. `inner_os` mode produces more emotionally authentic replies; `no_inner_os` mode produces more structurally consistent replies.

**How do I switch modes mid-conversation?**
- Open a **new conversation** and append the desired marker to its first message. You cannot switch modes within an existing session.

**How do I verify the mode is working?**
- Web: click **"查看思考过程"** to inspect the `<think>` block.
- API: check `response.choices[0].message` for a `reasoning_content` field if exposed, or parse `<think>...</think>` from the content.

```python
import re

def extract_thinking(raw_response: str) -> tuple[str, str]:
    """Extract <think> block and final reply from a raw response string."""
    think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    reply = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    return thinking, reply
```

**API rate limits / auth errors**
- Ensure `DEEPSEEK_API_KEY` environment variable is set correctly.
- Check your DeepSeek account quota for `deepseek-v4-pro` vs `deepseek-v4-flash`.
```
