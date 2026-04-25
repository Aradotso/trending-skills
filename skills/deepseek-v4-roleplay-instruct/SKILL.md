```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning in the <think> block.
triggers:
  - "deepseek roleplay thinking mode"
  - "deepseek inner monologue instruct"
  - "switch deepseek think style"
  - "deepseek character immersion mode"
  - "deepseek v4 roleplay control"
  - "deepseek pure analysis mode"
  - "deepseek think tag roleplay"
  - "deepseek roleplay instruct marker"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 exposes special **control markers** that can be injected into the first user message to steer *how the model thinks* inside its `<think>` block during roleplay sessions. There are two modes beyond the default:

| Mode | Marker Constant | `<think>` Behavior |
|---|---|---|
| **Default** | *(nothing)* | Model auto-decides |
| **Character Immersion** (`inner_os`) | `INNER_OS_MARKER` | First-person inner monologue wrapped in `（）` parentheses |
| **Pure Analysis** (`no_inner_os`) | `NO_INNER_OS_MARKER` | Cold, structured plot/reply planning — no inner-character voice |

**Applicable surfaces:**
- DeepSeek official APP / Web — **Expert Mode only**
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web **Quick Mode** is NOT supported

The markers are injected **once** at the end of the first user turn. Because the model always sees full conversation history, the instruction stays in context for every subsequent turn automatically.

---

## Installation / Setup

No package to install — this is a prompting pattern. Copy the marker strings into your project.

```python
# markers.py
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

---

## API Usage

### Minimal client wrapper

```python
# client.py
import os
from openai import OpenAI
from markers import INNER_OS_MARKER, NO_INNER_OS_MARKER

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # adjust to actual endpoint
)

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")  # or deepseek-v4-flash


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Append the appropriate marker to the first user message.
    Only call this for the FIRST turn — subsequent turns are plain dicts.
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


def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
```

### Full multi-turn roleplay session

```python
# session.py
from client import build_messages, chat

SYSTEM = "你是一个傲娇的女高中生，名叫雪乃。表面冷淡，内心其实很在意对方。"

# ── Turn 1: inject marker once ──────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # character-immersive inner monologue
)
reply1 = chat(messages)
print("雪乃:", reply1)

# ── Turn 2+: plain append, marker stays in history automatically ─────────────
messages.append({"role": "assistant", "content": reply1})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply2 = chat(messages)
print("雪乃:", reply2)

messages.append({"role": "assistant", "content": reply2})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply3 = chat(messages)
print("雪乃:", reply3)
```

### Pure-analysis mode example

```python
messages = build_messages(
    system_prompt="你是一个冷静睿智的侦探，正在调查一起谋杀案。",
    user_first_message="「尸体被发现在书房，书桌上有一封未写完的信」"从哪里开始调查？"",
    mode="no_inner_os",       # structured plot planning, no character voice
)
reply = chat(messages)
print("侦探:", reply)
```

---

## Web / App Usage (no code)

Paste the marker text directly at the end of your **first message**, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

All subsequent messages are plain — no marker needed.

---

## Configuration Reference

| Env Var | Purpose | Example |
|---|---|---|
| `DEEPSEEK_API_KEY` | Auth token for DeepSeek API | `sk-...` |
| `DEEPSEEK_MODEL` | Model to use | `deepseek-v4-pro` or `deepseek-v4-flash` |

---

## Common Patterns

### Helper: mode from string arg (CLI / config file friendly)

```python
import argparse
from client import build_messages, chat

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["default", "inner_os", "no_inner_os"], default="inner_os")
parser.add_argument("--system", default="你是一个温柔的角色。")
parser.add_argument("--first", required=True, help="First user message")
args = parser.parse_args()

messages = build_messages(args.system, args.first, mode=args.mode)
print(chat(messages))
```

```bash
python session_cli.py \
  --mode inner_os \
  --system "你是一个傲娇的女高中生。" \
  --first "「我走进教室」早上好。"
```

### Reusable session class

```python
class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os"):
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = []

    def send(self, user_message: str) -> str:
        if not self.messages:
            # First turn — inject marker
            self.messages = build_messages(self.system_prompt, user_message, self.mode)
        else:
            self.messages.append({"role": "user", "content": user_message})

        reply = chat(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.messages = []


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个骄傲的魔法师。",
    mode="inner_os",
)
print(session.send("「我走进魔法学院大厅」"请问图书馆在哪里？""))
print(session.send("「我跟上他的脚步」"你在这里学习多久了？""))
```

### Async version (for FastAPI / async apps)

```python
import asyncio
import os
from openai import AsyncOpenAI
from markers import INNER_OS_MARKER, NO_INNER_OS_MARKER

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages: list[dict]) -> str:
    response = await async_client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    from client import build_messages
    msgs = build_messages("你是一个侦探。", "「案发现场」"有什么线索？"", mode="no_inner_os")
    reply = await async_chat(msgs)
    print(reply)

asyncio.run(main())
```

---

## Expected `<think>` Output Comparison

**`inner_os` (character immersion):**
```
<think>
（他走进来了……心跳有点快。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
回复策略：先嫌弃，再给一个台阶。控制100字。
</think>
```

**`no_inner_os` (pure analysis):**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制150字，先动作描写再对话。
</think>
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Marker has no effect | Used Quick Mode on web | Switch to **Expert Mode** |
| Marker has no effect | Probabilistic — not 100% | Re-generate the response (roll again) |
| Wrong model | Using unsupported model | Use `deepseek-v4-flash` or `deepseek-v4-pro` only |
| Marker placed in system prompt | Wrong injection point | Move marker to end of **first user message** |
| Mode bleeds across sessions | Shared `messages` list | Call `session.reset()` or create a new session object |
| Inner monologue appears in final reply | Model confusion | Add explicit instruction in system prompt to keep `<think>` internal |

**Key rule:** The marker must be appended to the **first `user` turn**, not the `system` prompt. The system prompt placement was not used during training and is less effective.
```
