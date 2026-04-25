```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4's thinking mode during roleplay — switch between immersive character inner-monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue roleplay
  - deepseek v4 character immersion
  - switch deepseek thinking style
  - deepseek roleplay control instructions
  - deepseek pure analysis mode
  - deepseek roleplay prompt markers
  - deepseek think tag roleplay
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct documents a set of **special prompt-injection markers** that control how DeepSeek-V4 structures its `<think>` chain-of-thought during roleplay conversations. Two modes are provided:

| Mode | Effect on `<think>` block |
|---|---|
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Cold, director-style logical planning — no character voice, no parenthetical asides |
| **Default** | Model auto-selects based on scene complexity |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode only**
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> ⚠️ Web **Quick Mode** does NOT support these markers.

---

## The Marker Strings

Copy these verbatim. Append to the **first user message only**. All subsequent turns inherit the effect via conversation history.

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

No package to install. This is a **prompt-engineering pattern**. To use it programmatically:

1. Copy the marker constants into your project.
2. Use the `build_messages` helper (see below) or inline the logic.
3. Point your API client at `deepseek-v4-flash` or `deepseek-v4-pro`.

Store your API key in an environment variable:

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

---

## Core Usage Pattern (Python)

```python
import os
from openai import OpenAI  # DeepSeek API is OpenAI-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# --- Marker constants ---
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
    mode: str = "default",  # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Construct the initial message list with the appropriate marker
    injected at the end of the first user turn.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Example: Character Immersion ──────────────────────────────────────────────
messages = build_messages(
    system_prompt="你是一个傲娇的女高中生，暗恋班上的男主角，表面冷漠实则在意。",
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)

reply = chat(messages)
print(reply)

# Continue the conversation — marker stays in history automatically
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
    The marker is injected once (first user turn) and persists via history.
    """

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True
        self._client = OpenAI(
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

        response = self._client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    def reset(self, new_mode: str | None = None):
        """Start a fresh session, optionally changing mode."""
        system = self.messages[0]
        self.messages = [system]
        self._first_turn = True
        if new_mode:
            self.mode = new_mode


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是咖啡店里温柔的店员，对每位客人都格外用心。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「我推开咖啡店的门」"你好，请问还有位置吗？""))
print(session.send("「我坐到窗边的位置」"来一杯美式。""))
print(session.send("「我注意到你手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / App Usage

Paste the marker at the end of your **first** message, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages require no special formatting:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Verify the mode is active by clicking **"查看思考过程"** (View Thinking Process).

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Set once per session |
| `model` | `"deepseek-v4-flash"` / `"deepseek-v4-pro"` | Both support these markers |
| Marker injection point | First user message, end | Training-aligned position; do **not** use system prompt |
| Subsequent turns | No marker needed | History carries the instruction |

---

## Thinking Mode Comparison

```
# inner_os mode — actor "in character":
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>

# no_inner_os mode — director planning coldly:
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Common Patterns

### Pattern 1: Switch mode for a new conversation

```python
# Start fresh with a different mode — do NOT reuse the old session
session.reset(new_mode="no_inner_os")
print(session.send("「新场景开始」"你好。""))
```

### Pattern 2: Batch roleplay generation

```python
def generate_scene(system: str, opening: str, mode: str = "inner_os") -> str:
    msgs = build_messages(system, opening, mode=mode)
    return chat(msgs)

scenes = [
    ("你是中世纪骑士，忠诚而严肃。", "「女王走入议事厅」"骑士，战争已迫在眉睫。""),
    ("你是现代侦探，冷静而犀利。",   "「我进入嫌疑人的公寓」"这里有什么不对劲。""),
]

for system, opening in scenes:
    print(generate_scene(system, opening, mode="inner_os"))
    print("---")
```

### Pattern 3: Streaming with markers

```python
def chat_stream(messages: list[dict], model: str = "deepseek-v4-flash"):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()

messages = build_messages(
    "你是江湖中冷傲的剑客。",
    "「陌生人挡住了去路」"此路不通。"",
    mode="inner_os",
)
chat_stream(messages)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Marker has no effect | Wrong surface (Quick Mode) | Switch to **Expert Mode** in web app, or use API |
| Inner monologue doesn't appear | Probabilistic — not 100% | Re-generate (roll again); try `deepseek-v4-pro` for higher compliance |
| Mode bleeds between sessions | Reusing message history across sessions | Always start a new `messages` list for a new session |
| Marker in system prompt not working | Not the training-aligned injection point | Move marker to **first user message** end |
| Thinking block not visible in web | UI collapsed | Click "查看思考过程" to expand |
| API key error | Missing env var | `export DEEPSEEK_API_KEY="..."` before running |

---

## Key Rules (Summary for Agent)

1. **Inject marker once** — append to the last character of the first `user` message.
2. **Never put markers in system prompt** — they are less effective there.
3. **Never repeat markers** in later turns — history propagation handles it.
4. **To change mode** — open a new conversation / reset the message list.
5. **Probabilistic** — if the desired `<think>` style doesn't appear, regenerate.
6. **Both models supported** — `deepseek-v4-flash` (faster) and `deepseek-v4-pro` (higher compliance).
```
