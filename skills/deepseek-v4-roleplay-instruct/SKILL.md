```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical reasoning inside <think> tags
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion mode
  - switch deepseek think style
  - deepseek roleplay control instructions
  - deepseek pure analysis mode
  - deepseek think tag roleplay
  - deepseek v4 flash pro roleplay api
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct documents **special control instructions** for DeepSeek-V4's thinking mode during roleplay sessions. By appending specific markers to the **first user message**, you can steer what appears inside the model's `<think>` tags:

- **Character Immersion Mode**: The model thinks in first-person inner monologue (like an actor "in character")
- **Pure Analysis Mode**: The model thinks in cold, analytical director-style reasoning (no inner monologue)
- **Default**: Model auto-selects based on scene complexity

**Supported surfaces:**
- DeepSeek official APP / web in **Expert Mode**
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> ⚠️ Web Quick Mode is **not** supported. Triggers are probabilistic — retry if not activated on first attempt.

---

## Core Concepts

### How It Works

The control instruction is injected **once** at the end of the first user message. Because the model always sees full conversation history, the instruction persists automatically for all subsequent turns — no need to repeat it.

```
Turn 1: [roleplay scene] + [CONTROL INSTRUCTION]   ← inject here
Turn 2: [normal message]                            ← instruction still active via history
Turn 3: [normal message]                            ← still active
```

### Thinking Mode Comparison

```
Character Immersion Mode              Pure Analysis Mode
──────────────────────────────        ──────────────────────────────
<think>                               <think>
（He greeted me… heart racing.）      Scene: user greets, character is tsundere.
I should act indifferent.             Strategy: feign disinterest, body language
（Don't let him see I'm happy!）      reveals true feelings. ~150 chars, action
</think>                              then dialogue.
                                      </think>
```

---

## Control Instruction Text

### Character Immersion Mode (角色沉浸模式)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode (纯分析模式)

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
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
    Build the initial message list with the appropriate thinking mode marker.
    Only call this for the FIRST turn. Subsequent turns append normally.
    """
    if mode == "inner_os":
        user_first_message += INNER_OS_MARKER
    elif mode == "no_inner_os":
        user_first_message += NO_INNER_OS_MARKER
    # "default" — no modification

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_first_message},
    ]
```

### Full Multi-Turn Conversation Example

```python
import os
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

MODEL = "deepseek-v4-pro"  # or "deepseek-v4-flash"

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷淡但内心温柔。"

# ── Turn 1: inject marker ──────────────────────────────────────────────────
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",  # Character Immersion Mode
)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply = response.choices[0].message.content
print("[Turn 1 reply]", reply)

# ── Turn 2+: append normally, NO marker needed ─────────────────────────────
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
reply2 = response.choices[0].message.content
print("[Turn 2 reply]", reply2)
```

### Streaming with Thinking Visibility

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

def stream_roleplay(messages: list[dict], model: str = "deepseek-v4-pro"):
    """Stream response and print thinking + reply separately."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    full_content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            full_content += delta.content
    print()  # newline
    return full_content

# Usage
messages = build_messages(
    system_prompt="你是一个神秘的图书馆馆员。",
    user_first_message="「推开图书馆的门」"请问……这里有关于禁忌魔法的书吗？"",
    mode="no_inner_os",  # Pure Analysis Mode
)
stream_roleplay(messages)
```

### Conversation Manager Class

```python
import os
from openai import OpenAI
from typing import Literal

class DeepSeekRoleplay:
    """Manages a multi-turn DeepSeek-V4 roleplay conversation."""

    MODES = {
        "inner_os": INNER_OS_MARKER,
        "no_inner_os": NO_INNER_OS_MARKER,
        "default": "",
    }

    def __init__(
        self,
        system_prompt: str,
        mode: Literal["inner_os", "no_inner_os", "default"] = "inner_os",
        model: str = "deepseek-v4-pro",
    ):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        if self._first_turn and self.mode != "default":
            user_message += self.MODES[self.mode]
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
        """Start a fresh conversation, optionally switching mode."""
        system = self.messages[0]
        self.messages = [system]
        self._first_turn = True
        if new_mode:
            self.mode = new_mode


# ── Example usage ─────────────────────────────────────────────────────────
rp = DeepSeekRoleplay(
    system_prompt="你是一个温柔的咖啡师，有着不为人知的过去。",
    mode="inner_os",
)

print(rp.chat("「推开咖啡店的门」"你好，还有位置吗？""))
print(rp.chat("「坐到窗边」"来一杯美式。""))
print(rp.chat("「注意到你手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web UI Usage

Paste the instruction at the end of your **first message** (blank line between scene and instruction):

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

After that, chat normally — no further action needed.

---

## Configuration Reference

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `"inner_os"` | Character Immersion — brackets around inner thoughts |
| `mode` | `"no_inner_os"` | Pure Analysis — no brackets, analytical language only |
| `mode` | `"default"` | No marker injected, model decides |
| `model` | `"deepseek-v4-pro"` | Higher quality, slower |
| `model` | `"deepseek-v4-flash"` | Faster, lighter |
| Injection point | First user message (end) | Do NOT put in system prompt |

---

## Common Patterns

### Pattern 1: Always-on Inner Monologue Helper

```python
def inject_mode(messages: list[dict], mode: str) -> list[dict]:
    """
    Given an existing messages list, inject the marker into the
    first user message if not already present.
    """
    marker = INNER_OS_MARKER if mode == "inner_os" else NO_INNER_OS_MARKER
    result = list(messages)
    for i, msg in enumerate(result):
        if msg["role"] == "user":
            if marker not in msg["content"]:
                result[i] = {**msg, "content": msg["content"] + marker}
            break
    return result
```

### Pattern 2: Retry Logic for Probabilistic Triggers

```python
import re

def chat_with_retry(
    client, model: str, messages: list[dict],
    mode: str, max_retries: int = 3
) -> str:
    """
    Retry if the expected thinking pattern is not detected.
    inner_os expects bracketed content: （...） or (...)
    no_inner_os expects no bracketed inner monologue.
    """
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model, messages=messages
        )
        content = response.choices[0].message.content

        if mode == "inner_os":
            # Check think block contains brackets
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match and re.search(r"[（(].+?[）)]", think_match.group(1)):
                return content
        elif mode == "no_inner_os":
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match and not re.search(r"[（(].+?[）)]", think_match.group(1)):
                return content
        else:
            return content  # default: no check needed

        print(f"[Attempt {attempt + 1}] Mode not triggered, retrying…")

    return content  # return last attempt regardless
```

### Pattern 3: Switch Mode Mid-Session (New Conversation)

```python
# To switch modes, start a fresh conversation — don't append to old history
def switch_mode(
    system_prompt: str,
    new_mode: str,
    carry_over_context: str | None = None
) -> list[dict]:
    """
    Start a new conversation with a different thinking mode.
    Optionally summarize previous context into the first message.
    """
    first_message = carry_over_context or "（继续之前的故事）"
    return build_messages(system_prompt, first_message, mode=new_mode)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Instruction has no effect | Placed in system prompt | Move to **first user message** end |
| Instruction has no effect | Quick Mode on web | Switch to **Expert Mode** |
| Brackets still appear in `no_inner_os` | Probabilistic trigger | Retry — use `chat_with_retry()` |
| Mode stops working after many turns | Instruction pushed out of context window | Start new conversation with instruction |
| Wrong model | Using non-V4 model | Ensure model is `deepseek-v4-flash` or `deepseek-v4-pro` |
| `AuthenticationError` | Missing API key | Set `DEEPSEEK_API_KEY` env var |

### Verify Mode Activation

On web: click **"查看思考过程"** (View Thinking Process) to inspect `<think>` content.

Via API: parse the `<think>` block from the response:

```python
import re

def extract_thinking(content: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1).strip() if match else None

def verify_mode(content: str, mode: str) -> bool:
    thinking = extract_thinking(content)
    if not thinking:
        return False
    has_brackets = bool(re.search(r"[（(].+?[）)]", thinking))
    if mode == "inner_os":
        return has_brackets
    if mode == "no_inner_os":
        return not has_brackets
    return True
```

---

## FAQ

**Q: Can I put the instruction in the system prompt?**  
A: Not recommended. The marker is trained to be injected at the first **user** message end. System prompt placement is less reliable.

**Q: Does the instruction affect the final reply (outside `<think>`)?**  
A: Only indirectly. The instruction controls thinking style. `inner_os` tends to produce more emotionally authentic replies; `no_inner_os` tends to produce more structurally consistent replies.

**Q: Do I need to repeat the instruction every turn?**  
A: No. It stays in the conversation history automatically.

**Q: Can I use both markers in the same conversation?**  
A: No — use one per conversation. To switch, start a new conversation.
```
