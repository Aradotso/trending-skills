```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning inside <think> tags
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - switch deepseek think style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay control instructions
  - deepseek think tag roleplay
  - deepseek v4 flash pro roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions injected into the **first user message** to steer the style of its `<think>` reasoning block during roleplay. Two modes exist beyond the default:

| Mode | Effect inside `<think>` |
|---|---|
| **Default** | Model auto-selects based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses |
| **Pure Analysis** (`no_inner_os`) | Cold, director-style logical planning, no inner monologue |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only (not Quick Mode)
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`

---

## How It Works

The instruction is appended to the **first user turn only**. Because the model always sees full conversation history, the instruction remains in context for all subsequent turns automatically. No need to repeat it.

---

## Marker Constants

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

---

## Core Helper

```python
def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default"  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Construct the initial messages list with the appropriate thinking-mode marker
    appended to the first user message.

    Args:
        system_prompt: Character/scenario description for the system role.
        user_first_message: The player's opening action/dialogue.
        mode: Thinking mode selector.

    Returns:
        A messages list ready to send to the DeepSeek API.
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
```

---

## Full Working Example (OpenAI-compatible client)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # adjust if different
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

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


def build_messages(system_prompt, user_first_message, mode="default"):
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]


def chat(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Round 1: inject marker only here ──────────────────────────────────────────
system = "你是一个傲娇的女高中生，内心喜欢主角但表面上总是嫌弃他。"
opening = "「我走进教室」"早上好。""

messages = build_messages(system, opening, mode="inner_os")
reply = chat(messages)
print("Assistant:", reply)

# ── Round 2+: append normally, marker persists in history ─────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Assistant:", reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Assistant:", reply)
```

---

## Async Example

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def roleplay_session(system: str, turns: list[str], mode: str = "inner_os"):
    messages = build_messages(system, turns[0], mode=mode)

    for i, user_input in enumerate(turns):
        if i > 0:
            messages.append({"role": "user", "content": user_input})

        response = await client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"[Turn {i+1}] {reply}\n")

    return messages


asyncio.run(roleplay_session(
    system="你是一位神秘的咖啡店店主，隐藏着不为人知的过去。",
    turns=[
        "「我推开咖啡店的门」"你好，请问还有位置吗？"",
        "「我坐到窗边的位置」"来一杯美式。"",
        "「我注意到你手上有一道疤痕」"你的手……没事吧？"",
    ],
    mode="inner_os",
))
```

---

## Stateful Session Class

```python
class DeepSeekRoleplaySession:
    """
    Manages a stateful roleplay conversation with a fixed thinking mode.
    The mode marker is injected once into the first user turn.
    """

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True
        self.client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=base_url,
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

    def reset(self, system_prompt: str | None = None):
        """Start a new conversation, optionally with a new system prompt."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的女高中生。",
    mode="inner_os",
)
print(session.send("「我走进教室」"早上好。""))
print(session.send("「我在她旁边坐下」"今天心情不好吗？""))
```

---

## Web UI Usage (Step-by-Step)

1. Open DeepSeek web → enable **Expert Mode**
2. In the **first message only**, write your opening scene, leave a blank line, then paste the full marker:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

3. All subsequent messages need **no modification** — type naturally.
4. Click **"查看思考过程"** (View thinking process) to verify mode is active.

---

## Mode Comparison

```
# Character Immersion (inner_os)          # Pure Analysis (no_inner_os)
<think>                                   <think>
（他跟我打招呼了……心跳加速。）             场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。                  回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）                控制 150 字，先动作描写再对话。
</think>                                  </think>
```

---

## Common Patterns

### Pattern 1: Different modes for different scenarios

```python
# Emotional, character-driven scenes → inner_os
romance_session = DeepSeekRoleplaySession(system, mode="inner_os")

# Plot-heavy, structure-critical scenes → no_inner_os
mystery_session = DeepSeekRoleplaySession(system, mode="no_inner_os")

# Let the model decide → default
casual_session = DeepSeekRoleplaySession(system, mode="default")
```

### Pattern 2: Switching modes between sessions

```python
# New conversation = new mode. Never switch mid-session.
session.reset(system_prompt="新的角色设定……")
session.mode = "no_inner_os"
session._first_turn = True
```

### Pattern 3: Validating mode activation

```python
def verify_mode(reply_with_think: str, expected_mode: str) -> bool:
    """
    Check raw API response (if think block is exposed) for mode markers.
    Works when the API returns <think>...</think> in the raw content.
    """
    import re
    think_match = re.search(r"<think>(.*?)</think>", reply_with_think, re.DOTALL)
    if not think_match:
        return False  # No think block found
    think_content = think_match.group(1)

    if expected_mode == "inner_os":
        # Should contain parenthesized inner monologue
        return bool(re.search(r"[（(].+?[）)]", think_content))
    elif expected_mode == "no_inner_os":
        # Should NOT contain first-person inner monologue markers
        return not bool(re.search(r"[（(].+?[）)]", think_content))
    return True
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key |

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

---

## Troubleshooting

**Mode didn't activate (think block looks wrong)**
- The instruction is probabilistic, not guaranteed. Re-roll the response (regenerate).
- Ensure the marker was appended to the **first user message**, not system prompt.
- Confirm you are using Expert Mode on web, or `deepseek-v4-flash`/`deepseek-v4-pro` via API.
- Quick Mode on web does not support this feature.

**Marker in system prompt doesn't work as well**
- By design — the model was trained with markers in the user turn. Always inject into `user` role, first message only.

**Mode works for turn 1 but fades later**
- The instruction is still in context. This is normal model drift on long conversations. Start a new session or nudge with: `（请继续保持之前的思维风格）` at the end of a user message.

**Which model should I use?**
- `deepseek-v4-pro` — higher quality, slower
- `deepseek-v4-flash` — faster, slightly less stable mode adherence

**Can I use both markers at once?**
- No. Use one per session. They are mutually exclusive.
```
