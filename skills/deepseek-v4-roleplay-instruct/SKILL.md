```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersion inner monologue and pure analytical reasoning via special instruction markers
triggers:
  - "add deepseek roleplay thinking mode"
  - "switch deepseek inner monologue mode"
  - "deepseek v4 roleplay instructions"
  - "control deepseek thinking process in roleplay"
  - "deepseek character immersion vs pure analysis"
  - "inject deepseek roleplay marker into messages"
  - "deepseek think tag inner os mode"
  - "configure deepseek roleplay chat api"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions that steer the **content inside `<think>` tags** during roleplay conversations. By injecting a marker at the end of the **first user message**, you can choose between:

- **角色沉浸 (Character Immersion)** — The model thinks in first-person inner monologue wrapped in parentheses, like an actor inhabiting a role.
- **纯分析 (Pure Analysis)** — The model thinks in clean, third-person analytical language with no character inner dialogue.
- **Default** — The model decides automatically based on scene complexity.

The instruction persists throughout the conversation because the full history is always in context — you only inject once.

**Supported surfaces:**
- DeepSeek official APP / web **Expert Mode**
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

---

## How It Works

The marker is appended (separated by a blank line) to the **first user turn**. Every subsequent model call sees the full conversation history, so the marker remains active with no extra work.

```
[Normal first message]

[Marker block]
```

Subsequent turns are plain text — no special handling needed.

---

## The Marker Strings

### Character Immersion Mode (`inner_os`)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Expected `<think>` output style:
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

Expected `<think>` output style:
```
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
```

---

## Python Integration

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
from typing import Literal

Mode = Literal["default", "inner_os", "no_inner_os"]

def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: Mode = "default",
) -> list[dict]:
    """
    Construct the initial messages list with the appropriate thinking marker.

    Args:
        system_prompt: The character/scenario system prompt.
        user_first_message: The opening user turn (scene action + dialogue).
        mode: "inner_os" for character immersion, "no_inner_os" for pure analysis,
              "default" for model-chosen behaviour.

    Returns:
        A messages list ready to pass to the DeepSeek API.
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

### Full Multi-Turn Example (OpenAI-compatible SDK)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，名叫雪音。表面冷淡，内心其实很在意对方。"

# --- Round 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",   # or "deepseek-v4-flash"
    messages=messages,
)
reply = response.choices[0].message.content
print("Round 1 reply:", reply)

# --- Round 2+: append normally, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
reply2 = response.choices[0].message.content
print("Round 2 reply:", reply2)
```

### Streaming Version

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    system_prompt="你是一个侦探，冷静睿智。",
    user_first_message="「线人神色慌张地走进办公室」"侦探，出事了！"",
    mode="no_inner_os",
)

with client.chat.completions.stream(
    model="deepseek-v4-flash",
    messages=messages,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## TypeScript / Node.js Integration

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/v1",
});

const INNER_OS_MARKER = `

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复`;

const NO_INNER_OS_MARKER = `

【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演`;

type Mode = "default" | "inner_os" | "no_inner_os";

function buildMessages(
  systemPrompt: string,
  userFirstMessage: string,
  mode: Mode = "default"
): OpenAI.Chat.ChatCompletionMessageParam[] {
  let content = userFirstMessage;
  if (mode === "inner_os") content += INNER_OS_MARKER;
  if (mode === "no_inner_os") content += NO_INNER_OS_MARKER;

  return [
    { role: "system", content: systemPrompt },
    { role: "user", content },
  ];
}

// Usage
const messages = buildMessages(
  "你是一个傲娇的女高中生，名叫雪音。",
  "「我走进教室」"早上好。"",
  "inner_os"
);

const response = await client.chat.completions.create({
  model: "deepseek-v4-pro",
  messages,
});

console.log(response.choices[0].message.content);
```

---

## Conversation Manager Class (Python)

For longer sessions, use a stateful manager:

```python
import os
from openai import OpenAI
from typing import Literal

class DeepSeekRoleplaySession:
    """Manages a multi-turn DeepSeek roleplay conversation with thinking mode control."""

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
        model: str = "deepseek-v4-pro",
        mode: Literal["default", "inner_os", "no_inner_os"] = "inner_os",
    ):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True

    def chat(self, user_message: str) -> str:
        content = user_message
        if self._first_turn and self.mode in self.MARKERS:
            content += self.MARKERS[self.mode]
            self._first_turn = False

        self.messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, system_prompt: str | None = None):
        """Start a fresh conversation, optionally with a new system prompt."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        self._first_turn = True


# Example usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的女高中生，名叫雪音。",
    mode="inner_os",
)

print(session.chat("「我走进教室」"早上好。""))
print(session.chat("「我在她旁边坐下」"今天心情不好吗？""))
print(session.chat("「我注意到她手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / Manual Usage

Paste the marker after a blank line at the end of your **first message** in the chat box:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages are plain — no marker needed:

```
第二轮：「我坐到窗边的位置」"来一杯美式。"
第三轮：「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Click **"查看思考过程"** (View Thinking Process) to verify the mode is active.

---

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` | `deepseek-v4-flash` / `deepseek-v4-pro` | Flash = faster/cheaper; Pro = higher quality thinking |
| `mode` | `inner_os` / `no_inner_os` / `default` | Only affects `<think>` tag content |
| Marker position | End of **first user message** | Matches training injection position — most stable |
| System prompt | Any string | Place character description here, not the marker |
| Subsequent turns | No changes needed | History keeps marker active automatically |

---

## Common Patterns

### Pattern 1: Retry on Mode Failure

The marker has probabilistic (not guaranteed) effect. Retry if the thinking style is wrong:

```python
def chat_with_retry(session: DeepSeekRoleplaySession, message: str, max_retries: int = 3) -> str:
    """Retry if the model doesn't adopt the expected thinking style."""
    for attempt in range(max_retries):
        # For retries, reset and re-inject the first message
        if attempt > 0:
            first_user = session.messages[1]["content"]  # Already has marker
            system = session.messages[0]
            session.messages = [system, {"role": "user", "content": first_user}]

        reply = session.chat(message)
        return reply

    return reply  # Return last attempt regardless
```

### Pattern 2: Switch Mode in New Session

```python
# Start fresh with a different mode — do NOT append marker to existing session
session_immersive = DeepSeekRoleplaySession(system_prompt, mode="inner_os")
session_analytical = DeepSeekRoleplaySession(system_prompt, mode="no_inner_os")
```

### Pattern 3: Extract Thinking Content

```python
import re

def extract_thinking(raw_response: str) -> tuple[str, str]:
    """Split <think>...</think> from the final reply."""
    match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    thinking = match.group(1).strip() if match else ""
    reply = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    return thinking, reply

thinking, reply = extract_thinking(raw_model_output)
print("Inner monologue:", thinking)
print("Final reply:", reply)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Marker has no effect | Wrong surface (Quick Mode on web) | Use Expert Mode or API only |
| Thinking still uses inner monologue after `no_inner_os` | Probabilistic trigger missed | Re-roll / retry the first message |
| Marker active only on round 1 | Messages list rebuilt without history | Keep full `messages` list across turns |
| `no_inner_os` still shows parentheses | Model drift on long contexts | Start a new session; marker works best on fresh context |
| API auth error | Missing or wrong API key | Set `DEEPSEEK_API_KEY` env var |
| Marker in system prompt less effective | Training position mismatch | Move marker to end of first **user** message |

---

## Environment Variables

```bash
# Required for API usage
export DEEPSEEK_API_KEY="your-key-here"

# Optional: set default model
export DEEPSEEK_MODEL="deepseek-v4-pro"   # or deepseek-v4-flash
```

```python
import os

api_key = os.environ["DEEPSEEK_API_KEY"]
model   = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
```
```
