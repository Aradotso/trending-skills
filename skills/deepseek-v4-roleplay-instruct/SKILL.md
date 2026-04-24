```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue roleplay
  - deepseek v4 roleplay instruct
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay control instructions
  - deepseek think tag roleplay
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides **special control markers** you append to the first user message in a conversation to steer how the model thinks inside its `<think>` tag during roleplay. Two modes are available:

- **Character Immersion** (`inner_os`): The model writes first-person inner monologue (e.g. `（心想：……）`) inside `<think>`, acting like an actor in-character.
- **Pure Analysis** (`no_inner_os`): The model writes cold, third-person analytical planning inside `<think>`, like a director scripting a scene.

The default (no marker) lets the model auto-select based on scene complexity.

> **Scope**: DeepSeek official APP/web **Expert Mode**, and the `deepseek-v4-flash` / `deepseek-v4-pro` APIs. Quick Mode on web is not supported.
> **Note**: Triggers are probabilistic (~stable, not 100%). Re-roll if a mode doesn't activate.

---

## How It Works

The markers are injected **once** into the first user message. Because the model always sees its full conversation history, the marker remains in context for every subsequent turn automatically — no need to repeat it.

```
Turn 1 (user): 「...scene...」"Hello." + [MARKER appended here]
Turn 2 (user): 「...action...」"Next line."   ← marker still active via history
Turn 3 (user): 「...action...」"Another line." ← still active
```

---

## Marker Strings

### Character Immersion Marker (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker (`no_inner_os`)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python Integration (API Developers)

### Installation

```bash
pip install openai  # DeepSeek uses an OpenAI-compatible API
```

### Core Constants and Builder

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # or your DeepSeek endpoint
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

MODE_MARKERS = {
    "inner_os": INNER_OS_MARKER,
    "no_inner_os": NO_INNER_OS_MARKER,
    "default": "",
}


def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",
) -> list[dict]:
    """
    Build the initial messages list with the appropriate thinking-mode marker
    appended to the first user message.

    Args:
        system_prompt: Character/scenario description for the system role.
        user_first_message: The player's opening action/dialogue.
        mode: One of "inner_os", "no_inner_os", or "default".

    Returns:
        A messages list ready to send to the DeepSeek API.
    """
    marker = MODE_MARKERS.get(mode, "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_first_message + marker},
    ]
```

### Full Multi-Turn Conversation Example

```python
def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Setup ────────────────────────────────────────────────────────────────────
system_prompt = "你是一个傲娇的女高中生，喜欢主角但嘴上不承认。"

# Turn 1: inject marker once
messages = build_messages(
    system_prompt=system_prompt,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # character immersion
)
reply = chat(messages)
print("Assistant:", reply)

# Turn 2+: just append normally — marker stays in history
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
reply = chat(messages)
print("Assistant:", reply)

# Turn 3
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})
reply = chat(messages)
print("Assistant:", reply)
```

### Switching Modes Between Sessions

```python
def new_session(system_prompt: str, opening: str, mode: str):
    """Start a fresh session with a specific thinking mode."""
    messages = build_messages(system_prompt, opening, mode=mode)
    reply = chat(messages)
    messages.append({"role": "assistant", "content": reply})
    return messages

# Character immersion session
session_a = new_session(
    "你是一个神秘的咖啡馆店主。",
    "「我推开咖啡店的门」"请问还有位置吗？"",
    mode="inner_os",
)

# Pure analysis session (new conversation)
session_b = new_session(
    "你是一个冷静的侦探。",
    "「我把证物袋放在桌上」"这是我们在现场找到的。"",
    mode="no_inner_os",
)
```

### Helper: Verify Mode Activation

```python
import re

def extract_think_block(raw_response: str) -> str | None:
    """Extract content inside <think>…</think> for mode verification."""
    match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    return match.group(1).strip() if match else None


def verify_mode(raw_response: str, expected_mode: str) -> bool:
    """
    Quick heuristic check that the model honoured the requested mode.
    Returns True if the think block looks consistent with the expected mode.
    """
    think = extract_think_block(raw_response)
    if think is None:
        return False  # no think block found

    has_inner_monologue = bool(re.search(r"[（(].*?(心想|内心|OS).*?[）)]", think))

    if expected_mode == "inner_os":
        return has_inner_monologue
    elif expected_mode == "no_inner_os":
        return not has_inner_monologue
    return True  # default: no constraint
```

---

## Web / App Usage (No Code)

1. Open a **new conversation** in the DeepSeek app or web (Expert Mode).
2. Type your opening scene/dialogue.
3. Leave a blank line, then paste the desired marker.
4. Send. All future messages in that conversation are mode-locked automatically.

**Example first message:**

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Click **查看思考过程** (View thinking process) to verify the mode activated.

---

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mode` | `"inner_os"` | Character immersion — first-person inner monologue in `<think>` |
| `mode` | `"no_inner_os"` | Pure analysis — no inner monologue, only cold reasoning |
| `mode` | `"default"` | No marker injected; model auto-selects |
| Marker position | End of first `user` message | Training-aligned position; more stable than `system` prompt |
| Model scope | `deepseek-v4-flash`, `deepseek-v4-pro` | Flash and Pro API + Expert Mode web/app |
| Activation rate | Probabilistic, not 100% | Re-roll if mode doesn't activate on first try |

---

## Common Patterns

### Pattern 1: Reusable Session Manager

```python
class DeepSeekRoleplaySession:
    def __init__(
        self,
        system_prompt: str,
        mode: str = "default",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._mode = mode
        self._first_turn = True

    def send(self, user_message: str) -> str:
        content = user_message
        if self._first_turn:
            content += MODE_MARKERS.get(self._mode, "")
            self._first_turn = False

        self.messages.append({"role": "user", "content": content})
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个沉默寡言的剑客。",
    mode="inner_os",
)
print(session.send("「陌生人走进酒馆，在你对面坐下」"听说你是最好的剑客。""))
print(session.send("「陌生人将一袋金币推到桌上」"我需要你保护我的女儿。""))
```

### Pattern 2: Async API Calls

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

async def async_chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    messages = build_messages(
        "你是一个活泼的精灵向导。",
        "「我迷路在森林里」"有人吗？"",
        mode="inner_os",
    )
    reply = await async_chat(messages)
    print(reply)

asyncio.run(main())
```

### Pattern 3: Streaming with Mode Marker

```python
def stream_chat(messages: list[dict], model: str = "deepseek-v4-flash"):
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
    "你是一个骄傲的龙族公主。",
    "「勇者跪在你面前」"伟大的公主，我来求你的帮助。"",
    mode="no_inner_os",
)
reply = stream_chat(messages)
messages.append({"role": "assistant", "content": reply})
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Mode doesn't activate | Probabilistic trigger didn't fire | Resend / start a new conversation and try again |
| No `<think>` block visible | Using Quick Mode on web | Switch to **Expert Mode** in DeepSeek web/app |
| Marker has no effect via API | Marker placed in `system` instead of first `user` message | Move marker to end of first `user` content |
| Inner monologue appears in `no_inner_os` mode | Model occasionally breaks the constraint | Re-roll; consider adding stronger negation in your system prompt |
| Mode bleeds across sessions | You reused `messages` list from a prior session | Always start a new `messages` list for a new conversation |
| `deepseek-v4-flash` not found | Wrong model name or endpoint | Check your DeepSeek API dashboard for current model identifiers |

### FAQ

**Q: Can I put the marker in the system prompt?**
A: The marker is trained to work at the end of the **first user message**. Placing it in the system prompt is less reliable.

**Q: Does the marker affect the final reply text?**
A: The marker only targets `<think>` content. However, thinking style indirectly shapes output — immersion mode tends to produce more emotionally authentic replies; analysis mode produces more structurally consistent ones.

**Q: Do I need to repeat the marker every turn?**
A: No. The full conversation history is always in context, so the marker in turn 1 remains active for all subsequent turns automatically.

**Q: Which models support this?**
A: `deepseek-v4-flash` and `deepseek-v4-pro` via API; DeepSeek official app and web in **Expert Mode**. Quick Mode on web is not supported.
```
