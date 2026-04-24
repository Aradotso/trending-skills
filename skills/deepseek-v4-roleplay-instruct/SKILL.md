```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning in the <think> block
triggers:
  - "add inner monologue to DeepSeek roleplay"
  - "control DeepSeek thinking mode"
  - "DeepSeek character immersion mode"
  - "switch DeepSeek roleplay thinking style"
  - "DeepSeek v4 roleplay instructions"
  - "pure analysis mode DeepSeek"
  - "DeepSeek inner OS marker"
  - "DeepSeek think block roleplay control"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 (flash and pro variants) supports special control instructions injected into the **first user message** to steer how the model thinks inside its `<think>` block during roleplay. There are two modes beyond the default:

| Mode | Trigger Location | Think Block Behavior |
|---|---|---|
| **Default** | Nothing added | Model auto-selects based on complexity |
| **Character Immersion** (`inner_os`) | End of first user message | First-person inner monologue wrapped in parentheses |
| **Pure Analysis** (`no_inner_os`) | End of first user message | Logical planning only, no in-character inner voice |

**Supported surfaces:**
- DeepSeek official APP / web **Expert Mode**
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> ⚠️ Web Quick Mode is **not** supported. Instructions are probabilistic — retry if not triggered on first attempt.

---

## Core Marker Strings

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

No package to install. Copy the marker constants into your project. For API usage you need a DeepSeek API key:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

---

## Key API Pattern

### Message Builder

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


def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    mode options:
      "default"     — no marker appended
      "inner_os"    — character immersion marker appended
      "no_inner_os" — pure analysis marker appended
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

### Multi-Turn Conversation (Marker persists automatically)

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，性格表面冷淡，内心细腻。"

# Turn 1: inject marker into first user message only
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
messages.append({"role": "assistant", "content": reply})

# Turn 2+: append normally — first-turn marker stays in history, keeps working
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
)
reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": reply})
```

---

## Common Patterns

### Pattern 1: Wrapper Class for Long Roleplay Sessions

```python
import os
from openai import OpenAI


class DeepSeekRoleplay:
    def __init__(self, system_prompt: str, mode: str = "default", model: str = "deepseek-v4-flash"):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
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

    def reset(self, new_mode: str | None = None):
        """Start a new conversation, optionally switching mode."""
        if new_mode:
            self.mode = new_mode
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._first_turn = True


# Usage
rp = DeepSeekRoleplay(
    system_prompt="你是一个神秘的咖啡店店员，话不多，但观察力极强。",
    mode="inner_os",
)

print(rp.chat("「我推开咖啡店的门」"你好，还有位置吗？""))
print(rp.chat("「我坐到窗边」"来一杯美式。""))
print(rp.chat("「我注意到你手上有一道疤痕」"你的手……没事吧？""))
```

### Pattern 2: Streaming with Think Block Separation

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    system_prompt="你是一位冷静睿智的侦探。",
    user_first_message="「凶案现场，我递给你一张字条」"这是在死者手边发现的。"",
    mode="no_inner_os",
)

stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
)

think_buffer = []
reply_buffer = []
in_think = False

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    if "<think>" in delta:
        in_think = True
    elif "</think>" in delta:
        in_think = False
    elif in_think:
        think_buffer.append(delta)
        print(f"[THINK] {delta}", end="", flush=True)
    else:
        reply_buffer.append(delta)
        print(delta, end="", flush=True)

think_text = "".join(think_buffer)
reply_text = "".join(reply_buffer)
```

### Pattern 3: Web-Style Prompt Construction (copy-paste format)

```python
def format_web_prompt(scene: str, mode: str = "inner_os") -> str:
    """
    Returns a string ready to paste into DeepSeek Expert Mode web UI.
    """
    marker = ""
    if mode == "inner_os":
        marker = (
            "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
            "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
            "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
        )
    elif mode == "no_inner_os":
        marker = (
            "\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可\n"
            "2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代\n"
            "3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演"
        )
    return scene + marker

# Example
prompt = format_web_prompt(
    "「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"",
    mode="inner_os",
)
print(prompt)
```

---

## Configuration Reference

| Parameter | Value | Notes |
|---|---|---|
| `mode` | `"default"` / `"inner_os"` / `"no_inner_os"` | Pass to `build_messages()` |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support these markers |
| Marker injection point | End of **first user message only** | Do NOT inject into system prompt or later turns |
| API base URL | `https://api.deepseek.com/v1` | OpenAI-compatible |
| Auth | `DEEPSEEK_API_KEY` env var | Never hardcode |

---

## Troubleshooting

**Marker not taking effect (think block looks the same):**
- Confirm you're using **Expert Mode** on web, not Quick Mode
- Retry — triggering is probabilistic; 2–3 attempts usually works
- Ensure the marker is appended to the **first user turn**, not system prompt or later turns
- Check the raw `<think>` block (click "查看思考过程" on web) to verify

**Think block is empty or missing:**
- Model: `deepseek-v4-flash` and `deepseek-v4-pro` support thinking; verify your model name
- Some very short/simple prompts may produce minimal think content regardless of mode

**Want to switch modes mid-session:**
- Start a **new conversation** — markers are tied to the first turn in context
- In code: call `reset(new_mode="no_inner_os")` on the wrapper class, or rebuild `messages` from scratch

**Marker affects final reply content:**
- By design, the marker only targets `<think>` block behavior
- Indirectly: `inner_os` mode produces more emotionally authentic replies; `no_inner_os` produces more structurally consistent ones

**System prompt placement:**
- Do NOT put the marker in the system prompt — it was trained to respond to it in the user message position
- System prompt should contain only character description/persona

---

## Think Block Output Examples

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
```
