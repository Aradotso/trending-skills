```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — toggle between character-immersive inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue roleplay
  - deepseek v4 roleplay instructions
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay control markers
  - deepseek think tag roleplay
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control markers injected into the first user message to influence how the model reasons inside its `<think>` tags during roleplay. You can switch between **character-immersive** (actor-style inner monologue) and **pure analytical** (director-style cold planning) thinking without changing the final reply format.

---

## What This Project Does

DeepSeek-V4 (flash and pro variants) exposes its chain-of-thought inside `<think>...</think>` tags. By appending structured Chinese-language markers to the **first user message**, you probabilistically steer the thinking style:

| Mode | Thinking Behaviour |
|---|---|
| **Default** | Model auto-selects based on scene complexity |
| **Character Immersion** (`inner_os`) | First-person inner monologue wrapped in brackets — `（心想：……）` |
| **Pure Analysis** (`no_inner_os`) | Direct analytical statements only, no in-character inner voice |

> **Scope**: DeepSeek official APP / web **Expert Mode**, and the `deepseek-v4-flash` / `deepseek-v4-pro` APIs. Web Quick Mode is not supported. Triggering is probabilistic — retry if not activated on first attempt.

---

## Marker Strings

### Character Immersion Marker (`inner_os`)

```text
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker (`no_inner_os`)

```text
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Python Integration

### Constants and Message Builder

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
    mode: str = "default"  # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Build the initial message list with the appropriate thinking marker
    injected into the first user turn.
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

### Full Conversation Loop with OpenAI-Compatible Client

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # or your deployment URL
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

# --- Round 1: inject marker into first user message ---
system_prompt = "你是一个傲娇的女高中生，名叫晴子，暗恋主角但嘴上从不承认。"
first_user_msg = "「我走进教室，看到晴子正趴在桌上发呆」"早上好，晴子。""

messages = build_messages(system_prompt, first_user_msg, mode="inner_os")

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print("Assistant:", assistant_reply)

# Append assistant reply to history
messages.append({"role": "assistant", "content": assistant_reply})

# --- Round 2+: just append normally, marker stays in history ---
messages.append({
    "role": "user",
    "content": "「我在她旁边坐下，放下书包」"今天心情不好吗？脸色有点差。""
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
print("Assistant:", response.choices[0].message.content)
```

### Accessing the Think Tag Content

```python
def extract_think_and_reply(raw_content: str) -> tuple[str, str]:
    """
    Split DeepSeek's response into thinking chain and visible reply.
    Returns (think_content, reply_content).
    """
    import re
    think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    reply = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    return think, reply


# Usage
raw = response.choices[0].message.content
thinking, visible_reply = extract_think_and_reply(raw)
print("=== Think Process ===")
print(thinking)
print("=== Reply ===")
print(visible_reply)
```

---

## Web / App Usage

Append the marker directly to your **first message** in a new conversation, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need no modification — the marker persists in context automatically.

---

## Common Patterns

### Pattern 1: Reusable Roleplay Session Class

```python
class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-pro"):
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def chat(self, user_message: str) -> tuple[str, str]:
        """Send a message and return (thinking, reply)."""
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
        raw = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": raw})

        return extract_think_and_reply(raw)


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的图书馆司书，守护着无数故事的入口。",
    mode="inner_os",
)

think, reply = session.chat("「我第一次踏入图书馆，空气中弥漫着旧书的味道」"…这里真的有我要找的书吗？"")
print(reply)

think, reply = session.chat("「我顺着书架走，停在一本没有书名的黑皮书前」")
print(reply)
```

### Pattern 2: Switch Mode Mid-Story (New Session)

```python
def switch_mode(old_session: DeepSeekRoleplaySession, new_mode: str) -> DeepSeekRoleplaySession:
    """
    Modes cannot be changed mid-session via marker.
    Start a fresh session with the same system prompt but new mode.
    """
    system_msg = old_session.messages[0]["content"]
    return DeepSeekRoleplaySession(
        system_prompt=system_msg,
        mode=new_mode,
        model=old_session.model,
    )

# Switch from immersive to analytical
new_session = switch_mode(session, new_mode="no_inner_os")
```

### Pattern 3: Validate Marker Activation

```python
def verify_mode_activated(thinking: str, mode: str) -> bool:
    """
    Heuristically check whether the marker took effect.
    """
    has_brackets = bool(re.search(r"[（(].{2,50}[）)]", thinking))
    has_first_person = any(kw in thinking for kw in ["我心想", "我觉得", "我暗自", "我感到"])

    if mode == "inner_os":
        return has_brackets or has_first_person
    elif mode == "no_inner_os":
        return not has_brackets and not has_first_person
    return True  # default mode — no validation needed


think, reply = session.chat("「我再次出现在图书馆门口」")
if not verify_mode_activated(think, mode="inner_os"):
    print("Warning: marker may not have activated — consider retrying.")
```

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"`, `"no_inner_os"`, `"default"` | Set once per session on first user turn |
| `model` | `"deepseek-v4-flash"`, `"deepseek-v4-pro"` | Flash is faster; Pro is more capable |
| Marker injection point | First user message only | Training position — most stable here |
| System prompt location | Standard `system` role | Marker in `user` role, not `system` |

---

## Troubleshooting

**Marker didn't activate (wrong thinking style appeared)**
- Retry the same first message — activation is probabilistic, not guaranteed
- Ensure the marker is appended to the **first user message**, not the system prompt
- Verify you're using Expert Mode on web (not Quick Mode)
- Check the model is `deepseek-v4-flash` or `deepseek-v4-pro`

**`<think>` tags not visible in API response**
- Confirm your client is not stripping raw content; access `response.choices[0].message.content` directly
- Some SDK versions may parse think tags separately — check your SDK docs

**Mode changed mid-conversation**
- Markers only steer; the model may drift over long contexts
- Start a new session with the marker re-injected if drift is severe

**Chinese encoding issues in Python**
- Ensure your source file is saved as UTF-8
- Use `# -*- coding: utf-8 -*-` at the top of scripts targeting older Python environments

**System prompt vs. user message placement**
- Do **not** put the marker in the system prompt — the model was trained to respond to it in the user turn
- Placing it in `system` role reduces effectiveness significantly
```
