```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special instruction markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - switch deepseek thinking style
  - deepseek roleplay instruction marker
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - control deepseek think tag roleplay
  - deepseek v4 roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 supports special control instructions injected into the **first user message** to steer how the model reasons inside its `<think>` tag during roleplay sessions. There are two modes beyond the default:

| Mode | Effect on `<think>` |
|---|---|
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses, e.g. `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Cold, director-style logical planning — no character voice, no parenthetical inner monologue |
| **Default** | Model auto-selects based on scene complexity |

**Supported surfaces:**
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- ⚠️ Web **Quick Mode** is NOT supported

---

## Core Concept

The markers are appended to the **first user message only**. Because the model always sees full conversation history, the marker remains in context for all subsequent turns automatically. You never repeat it.

---

## Instruction Markers (Copy-Ready)

### Character Immersion Mode

```text
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode

```text
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python API Usage

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
    mode: str = "default"
) -> list[dict]:
    """
    Build the initial message list with the appropriate thinking mode marker.

    Args:
        system_prompt: Character/scene description for the system role.
        user_first_message: The player's opening action/dialogue.
        mode: One of "default", "inner_os", "no_inner_os".

    Returns:
        List of message dicts ready for the DeepSeek chat API.
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

### Full Multi-Turn Session

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "你是一个傲娇的女高中生，名叫晴，坐在教室靠窗的位置。"
    "表面上冷漠高冷，内心其实很在意对方。"
)

# --- Turn 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室，看到晴正望着窗外」"早上好。"",
    mode="inner_os",  # character immersion
)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply = response.choices[0].message.content
print("Turn 1:", reply)

# --- Turn 2+: append normally, marker stays in history automatically ---
messages.append({"role": "assistant", "content": reply})
messages.append({
    "role": "user",
    "content": "「我在她旁边坐下」"今天心情不好吗？""
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply2 = response.choices[0].message.content
print("Turn 2:", reply2)
```

### Streaming Version

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def roleplay_stream(messages: list[dict], model: str = "deepseek-v4-pro"):
    """Stream a roleplay response, printing think and reply separately."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    full_content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_content += delta
    print()
    return full_content

messages = build_messages(
    system_prompt="你是一个神秘的咖啡师，隐藏着一段过去。",
    user_first_message="「我推开咖啡店的门」"请问还有位置吗？"",
    mode="inner_os",
)

reply = roleplay_stream(messages)
```

---

## Web Usage (No Code)

1. Open DeepSeek APP or web in **Expert Mode** (专家模式).
2. In your **first message**, write your opening roleplay line, add a blank line, then paste the desired marker.
3. Send. From the second message onward, write normally — the marker stays active in context.

**Example first message:**

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Subsequent messages (no marker needed):**

```
「我坐到窗边的位置」"来一杯美式。"
```

---

## Configuration Patterns

### Selecting a Model

```python
# Faster, cheaper
MODEL = "deepseek-v4-flash"

# More capable, better adherence to markers
MODEL = "deepseek-v4-pro"
```

### Environment Variables

```bash
# Required
export DEEPSEEK_API_KEY="your-api-key-here"

# Optional: override base URL if using a proxy
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
```

### Reusable Session Class

```python
import os
from openai import OpenAI

class DeepSeekRoleplay:
    """Manages a stateful multi-turn DeepSeek roleplay session."""

    MARKERS = {
        "inner_os": (
            "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
            "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
            "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
        ),
        "no_inner_os": (
            "\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
            "1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可\n"
            "2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代\n"
            "3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演"
        ),
    }

    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-pro",
    ):
        self.model = model
        self.mode = mode
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn and self.mode in self.MARKERS:
            user_message += self.MARKERS[self.mode]
            self._first_turn = False

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt: str | None = None):
        """Start a new session, optionally with a new system prompt."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        self._first_turn = True


# Usage
session = DeepSeekRoleplay(
    system_prompt="你是一个傲娇的女高中生，名叫晴。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.chat("「我走进教室」"早上好。""))
print(session.chat("「我在她旁边坐下」"今天心情不好吗？""))
print(session.chat("「我递给她一块糖」"送你的。""))
```

---

## Common Patterns

### Pattern: Retry Until Marker Takes Effect

Since markers have probabilistic (not guaranteed) activation, retry if the `<think>` block does not show the expected format:

```python
import re

def chat_with_retry(session: DeepSeekRoleplay, user_message: str, max_retries: int = 3) -> str:
    """
    Retry the first turn until the marker appears to have taken effect.
    Detection heuristic: look for parenthetical inner monologue in the raw response.
    """
    for attempt in range(max_retries):
        # Save state before attempt
        saved_messages = [m.copy() for m in session.messages]
        saved_first_turn = session._first_turn

        reply = session.chat(user_message)

        # Check for inner_os activation (parenthetical pattern)
        if session.mode == "inner_os":
            # The <think> block is typically not exposed in content,
            # but some API configs return it; adjust check to your setup
            if re.search(r'[（(].+[）)]', reply):
                return reply  # looks good
        else:
            return reply  # no structural check needed for other modes

        if attempt < max_retries - 1:
            print(f"Marker may not have activated, retrying ({attempt + 2}/{max_retries})...")
            # Restore state and retry
            session.messages = saved_messages
            session._first_turn = saved_first_turn

    return reply  # return last attempt regardless
```

### Pattern: Mode Switching (New Session)

```python
def switch_mode(session: DeepSeekRoleplay, new_mode: str) -> DeepSeekRoleplay:
    """
    Switch thinking mode by creating a new session.
    Preserves system prompt from the existing session.
    """
    system_prompt = session.messages[0]["content"]
    return DeepSeekRoleplay(
        system_prompt=system_prompt,
        mode=new_mode,
        model=session.model,
    )

# Start with immersion mode
session = DeepSeekRoleplay("你是一个傲娇的女高中生。", mode="inner_os")
session.chat("「我走进教室」"早上好。"")

# Switch to pure analysis for debugging
session = switch_mode(session, "no_inner_os")
session.chat("「我坐下来」"我们今天要学什么？"")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Marker seems to have no effect | Probabilistic activation; not 100% guaranteed | Re-send (roll again); use `deepseek-v4-pro` which has better instruction following |
| Using Quick Mode on web | Quick Mode does not support think-tag control | Switch to **Expert Mode** (专家模式) |
| Marker placed in system prompt | Training injection point is user message turn 1 | Always append marker to first **user** message, not system prompt |
| Inner monologue not visible | `<think>` content may not be exposed by default | Click "查看思考过程" (View Thinking Process) in the web UI to verify |
| Mode active for only first reply | You removed/modified messages array incorrectly | Keep the full message history; never strip the first user message |
| API returns error on model name | Wrong model identifier | Use exactly `deepseek-v4-flash` or `deepseek-v4-pro` |

---

## Quick Reference

```python
# Minimal working example
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)

messages = [
    {"role": "system", "content": "你是一个傲娇的女高中生。"},
    {"role": "user",   "content": "「我走进教室」"早上好。"" + MARKER},
]

r = client.chat.completions.create(model="deepseek-v4-pro", messages=messages)
print(r.choices[0].message.content)
```
```
