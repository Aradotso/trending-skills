```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning via special prompt markers
triggers:
  - "deepseek roleplay thinking mode"
  - "deepseek inner monologue instruct"
  - "switch deepseek think style"
  - "deepseek roleplay immersion prompt"
  - "deepseek v4 roleplay control"
  - "deepseek character inner OS marker"
  - "deepseek pure analysis mode"
  - "deepseek roleplay special instructions"
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 supports special **control markers** injected into the first user message to change how the model thinks inside its `<think>` tags during roleplay sessions. Two modes are available beyond the default:

| Mode | Marker | Thinking Style |
|---|---|---|
| **Default** | Nothing added | Model chooses automatically |
| **Character Immersion** | `INNER_OS_MARKER` | First-person inner monologue in parentheses `（心想：…）` |
| **Pure Analysis** | `NO_INNER_OS_MARKER` | Cold analytical planning, no character inner voice |

**Supported surfaces:**
- DeepSeek official APP / web (Expert Mode)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> Note: Web Quick Mode does **not** support these markers. Triggers are probabilistic — retry if it doesn't fire on the first attempt.

---

## Core Concepts

### How Markers Work

Markers are appended to the **first user message only**. Because the model sees full conversation history on every turn, the marker persists automatically across all subsequent rounds — no need to repeat it.

### The Two Markers

**Character Immersion (`INNER_OS_MARKER`)** — model thinks like an actor in-character:
```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Pure Analysis (`NO_INNER_OS_MARKER`)** — model thinks like a director, structurally:
```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Installation / Setup

No package to install. This is a prompting technique. Set up your API client normally:

```python
pip install openai  # DeepSeek uses OpenAI-compatible API
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)
```

---

## Key API Usage

### Define the Markers

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

### Build Messages Helper

```python
def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"
) -> list[dict]:
    """
    mode options:
      "default"     — no marker injected
      "inner_os"    — character immersion mode
      "no_inner_os" — pure analysis mode
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

### Full Multi-Turn Roleplay Session

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


# --- Round 1: inject marker in first user message ---
system_prompt = "你是一个傲娇的女高中生，表面冷漠，内心其实很在意对方。"
first_user_msg = "「我走进教室」"早上好。""

messages = build_messages(system_prompt, first_user_msg, mode="inner_os")
reply = chat(messages)
print("Assistant:", reply)

# --- Round 2+: append normally, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

reply = chat(messages)
print("Assistant:", reply)

# --- Round 3 ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})

reply = chat(messages)
print("Assistant:", reply)
```

---

## Common Patterns

### Pattern 1: Reusable Session Class

```python
class DeepSeekRoleplaySession:
    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-pro",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.mode = mode
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
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

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Start a new conversation, keeping system prompt and mode."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的咖啡店店主，话不多，但观察力极强。",
    mode="inner_os",
)

print(session.send("「我推开咖啡店的门」"你好，还有位置吗？""))
print(session.send("「我坐到窗边」"来一杯美式。""))
print(session.send("「我注意到你的眼神」"你认识我？""))
```

### Pattern 2: Streaming with Marker

```python
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


messages = build_messages(
    "你是一位沉默寡言的剑客，只对真正的强者开口。",
    "「我挡住了你的去路」"在下想讨教几招。"",
    mode="no_inner_os",
)
reply = stream_roleplay(messages)
messages.append({"role": "assistant", "content": reply})
```

### Pattern 3: Mode Switching (New Conversation Per Mode)

```python
def create_session_with_mode(system: str, opening: str, mode: str):
    """Each mode requires a fresh conversation."""
    msgs = build_messages(system, opening, mode=mode)
    # ... continue chatting
    return msgs

# Immersive — for emotional, character-driven scenes
immersive_msgs = create_session_with_mode(
    system="你是一个暗恋主角很久的邻居。",
    opening="「我在楼道里碰到你」"哦，是你。"",
    mode="inner_os",
)

# Analytical — for complex plot or structured dialogue
analytical_msgs = create_session_with_mode(
    system="你是一个经验丰富的侦探，正在审问嫌疑人。",
    opening="「我把照片放在桌上」"解释一下这张照片。"",
    mode="no_inner_os",
)
```

---

## Web / App Usage (No Code)

**Step 1 only:** Paste the marker at the end of your **first message**, separated by a blank line.

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages are typed normally — the marker stays in context history automatically.

To verify: click **"查看思考过程"** (View Thinking Process) and check the `<think>` block.

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"default"`, `"inner_os"`, `"no_inner_os"` | Passed to `build_messages()` |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Flash = faster, Pro = higher quality |
| Marker position | End of **first** user message | Training-aligned injection point |
| System prompt placement | Standard `system` role | Do not put markers here |

---

## Troubleshooting

**Marker didn't fire (thinking mode didn't change)**
- Retry — activation is probabilistic, not guaranteed 100%
- Confirm you're using Expert Mode (web) or the correct model names via API
- Ensure the marker is in the **first user message**, not system prompt or later turns
- Do not translate or paraphrase the Chinese marker text — use it verbatim

**Mode not persisting after a few turns**
- The marker persists via conversation history automatically; no action needed
- If context gets truncated (very long sessions), the marker may fall out — inject it again in a new conversation

**Want to switch modes mid-session**
- You cannot switch modes in the same conversation
- Start a new conversation and inject the other marker in the first message

**Marker in system prompt doesn't work as well**
- By design — the model was trained to respond to this marker in the user turn, not the system role

**API key / auth errors**
```python
# Always use environment variable, never hardcode
api_key=os.environ["DEEPSEEK_API_KEY"]
```

---

## Think Block Output Examples

**Character Immersion (`inner_os`):**
```xml
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
回复策略：冷漠但不失礼，控制在100字以内。
</think>
```

**Pure Analysis (`no_inner_os`):**
```xml
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先表现嫌弃，通过肢体语言暴露真实情感。
控制150字，先动作描写再对话。
语气：表面冷淡，实际关心。
</think>
```
```
