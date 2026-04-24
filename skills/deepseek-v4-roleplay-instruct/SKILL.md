```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning using special marker instructions
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion mode
  - switch deepseek thinking style
  - deepseek roleplay instruct markers
  - deepseek inner OS mode
  - deepseek pure analysis mode
  - deepseek v4 roleplay control
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides special control markers that influence how the model thinks during roleplay scenarios. By appending specific instruction blocks to the **first user message**, you can steer the `<think>` chain-of-thought between two distinct styles:

- **Character Immersion Mode** — the model thinks in first-person inner monologue as the character (parenthesized internal thoughts)
- **Pure Analysis Mode** — the model thinks as a cold director/planner, no inner monologue, only structured reasoning

These markers work with:
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs
- ⚠️ Web **Quick Mode** is NOT supported

---

## How It Works

The model sees the full conversation history on every reply. By injecting the marker into the **first user message only**, it persists in context for the entire session — no need to repeat it on subsequent turns.

**Trigger probability is not 100%** — if the mode doesn't activate, retry (re-roll). Stable enough for production use with retry logic.

---

## The Two Markers

### Character Immersion Mode (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

**Result in `<think>`:**
```
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
```

### Pure Analysis Mode (`no_inner_os`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

**Result in `<think>`:**
```
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
```

---

## Python Integration

### Core Constants and Builder

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


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking mode marker.
    Only call this for the FIRST turn. Append subsequent turns normally.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" → no marker, model auto-selects

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message},
    ]
```

### Full Conversation Loop (OpenAI-compatible client)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # adjust to actual endpoint
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷淡但内心温柔。"

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
print(reply)

# Append assistant reply to history
messages.append({"role": "assistant", "content": reply})

# --- Turn 2+: just append user message, marker persists in history ---
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
reply = response.choices[0].message.content
print(reply)
messages.append({"role": "assistant", "content": reply})
```

### With Retry Logic (handling probabilistic activation)

```python
import time
from typing import Callable


def chat_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    validate_fn: Callable[[str], bool] | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Call the API with optional validation and retry.
    validate_fn receives the thinking block content if you can extract it,
    or the full response — return True if the mode activated correctly.
    """
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content

        if validate_fn is None or validate_fn(content):
            return content

        if attempt < max_retries - 1:
            print(f"Mode not activated (attempt {attempt + 1}), retrying...")
            time.sleep(retry_delay)

    return content  # return last attempt regardless


def is_inner_os_active(response_text: str) -> bool:
    """Simple heuristic: check if parenthesized inner monologue appeared."""
    return "（" in response_text or "(内心" in response_text or "心想" in response_text


# Usage
messages = build_messages(SYSTEM_PROMPT, "「我推开咖啡店的门」"请问有位置吗？"", mode="inner_os")
reply = chat_with_retry(
    client=client,
    model="deepseek-v4-pro",
    messages=messages,
    validate_fn=is_inner_os_active,
    max_retries=3,
)
```

### Conversation Manager Class

```python
class DeepSeekRoleplaySession:
    """
    Manages a multi-turn roleplay session with a fixed thinking mode.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        mode: str = "inner_os",
    ):
        self.client = client
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            # Inject marker only on the first user message
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
        """Start a new session (clears history, optionally changes mode)."""
        system = self.messages[0]
        self.messages = [system]
        self._first_turn = True
        if new_mode:
            self.mode = new_mode


# Usage
session = DeepSeekRoleplaySession(
    client=client,
    model="deepseek-v4-pro",
    system_prompt="你是一个神秘的图书馆管理员。",
    mode="inner_os",
)

print(session.send("「我走进图书馆」"请问有关于魔法的书吗？""))
print(session.send("「我接过书，翻开第一页」"))
print(session.send("「我抬头看向你」"这本书……是禁书吗？""))
```

---

## Web Usage (No Code)

1. Open DeepSeek official APP or web in **Expert Mode**
2. In the **first message** input box, write your roleplay opening, then paste the marker on a new line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

3. All subsequent messages: type normally, **no marker needed**
4. Click "查看思考过程" (View Thinking Process) to verify mode activated

**Switching modes:** Start a new conversation and paste the other marker in turn 1.

---

## Mode Comparison

| Feature | Character Immersion (`inner_os`) | Pure Analysis (`no_inner_os`) | Default |
|---|---|---|---|
| Inner monologue | ✅ Parenthesized `（…）` | ❌ Forbidden | Auto |
| First-person in `<think>` | ✅ Yes | ❌ No | Auto |
| Best for | Emotional depth, character authenticity | Structural consistency, plot planning | General use |
| Marker placement | First user message tail | First user message tail | None |

---

## Common Patterns

### Pattern 1: System Prompt + Mode Setup

```python
system = """
你是艾莉，一个在孤儿院长大的年轻魔法师。
性格：外表坚强，内心渴望被接受。
说话风格：简短、直接，偶尔流露脆弱。
"""

session = DeepSeekRoleplaySession(
    client=client,
    model="deepseek-v4-flash",  # faster, cheaper
    system_prompt=system,
    mode="inner_os",
)
```

### Pattern 2: Director Mode for Complex Plot Management

```python
# Use no_inner_os when you need the model to track plot consistency
# rather than emotional authenticity
session = DeepSeekRoleplaySession(
    client=client,
    model="deepseek-v4-pro",
    system_prompt="你是一个复杂推理游戏中的NPC，需要精确追踪线索和玩家行为。",
    mode="no_inner_os",
)
```

### Pattern 3: Dynamic Mode Per Session

```python
def start_session(system_prompt: str, emotional_depth: bool) -> DeepSeekRoleplaySession:
    return DeepSeekRoleplaySession(
        client=client,
        model="deepseek-v4-pro",
        system_prompt=system_prompt,
        mode="inner_os" if emotional_depth else "no_inner_os",
    )
```

---

## Environment Setup

```bash
# Required environment variable
export DEEPSEEK_API_KEY="your-api-key-here"

# Install OpenAI-compatible client
pip install openai
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

## Troubleshooting

| Problem | Solution |
|---|---|
| Mode didn't activate | Retry the request 1–3 times; activation is probabilistic |
| Marker not persisting after turn 1 | Ensure you're appending to the same `messages` list, not creating a new one |
| Works on web but not API | Confirm you're using Expert Mode on web; confirm API model is `deepseek-v4-flash` or `deepseek-v4-pro` |
| Quick Mode on web not working | Switch to **Expert Mode** — Quick Mode is unsupported |
| Inner monologue appearing in pure analysis mode | Increase `max_retries` in retry logic; the marker reduces but doesn't eliminate probability |
| Marker in system prompt not working | Move it to the **first user message tail** — that's where the model was trained to expect it |

---

## Key Rules Summary

1. **Marker goes in the first `user` message** — not system prompt, not later turns
2. **One mode per session** — to switch, start a new conversation/session
3. **Subsequent turns need no changes** — the marker stays in context automatically
4. **Probabilistic, not deterministic** — build retry logic for production use
5. **Only affects `<think>` content** — the final response style is influenced indirectly, not directly controlled
```
