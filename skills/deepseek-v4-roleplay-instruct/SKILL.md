```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning in the <think> block
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - deepseek character immersion mode
  - control deepseek think block
  - deepseek roleplay instruct
  - deepseek pure analysis mode
  - deepseek v4 roleplay api
  - switch deepseek thinking style
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 supports special control instructions that change how the model *thinks* during roleplay. By appending a marker to the **first user message**, you can steer the `<think>` block between two styles:

| Mode | Think Block Style |
|---|---|
| **Default** | Model chooses automatically based on complexity |
| **Role Immersion** (`inner_os`) | First-person inner monologue wrapped in parentheses `（心想：…）` |
| **Pure Analysis** (`no_inner_os`) | Detached director-style planning, no inner monologue |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode only** (深度思考/专家模式)
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web Quick Mode is **not** supported

---

## How It Works

The marker is appended to the **first user turn only**. Because the model always sees full conversation history, the instruction stays in context for every subsequent turn automatically. No changes needed to later messages.

---

## The Two Markers (copy-ready)

### Role Immersion Marker
```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker
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
    mode: str = "default"  # "default" | "inner_os" | "no_inner_os"
) -> list[dict]:
    """
    Build the initial message list with the appropriate thinking-mode marker
    injected into the first user turn.
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

### Full Multi-Turn Example

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，性格表面冷淡但内心温柔。"

# --- Turn 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",   # or "no_inner_os" or "default"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",   # or "deepseek-v4-flash"
    messages=messages,
)
reply = response.choices[0].message.content
print(reply)

# --- Turn 2+: append normally, marker stays in history ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response2 = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
print(response2.choices[0].message.content)
```

### Streaming Variant

```python
stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
)

think_buf, reply_buf = [], []
in_think = False

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    # DeepSeek wraps reasoning in <think>...</think>
    if "<think>" in delta:
        in_think = True
    if in_think:
        think_buf.append(delta)
    else:
        reply_buf.append(delta)
    if "</think>" in delta:
        in_think = False

print("THINK:", "".join(think_buf))
print("REPLY:", "".join(reply_buf))
```

---

## Web Usage (1-step)

Paste the marker at the end of your **first** message, separated by a blank line:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent turns need no marker — just write normally.

---

## TypeScript / JavaScript

```typescript
const INNER_OS_MARKER = `\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复`;

const NO_INNER_OS_MARKER = `\n\n【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演`;

type ThinkMode = "default" | "inner_os" | "no_inner_os";

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

function buildMessages(
  systemPrompt: string,
  userFirstMessage: string,
  mode: ThinkMode = "default"
): Message[] {
  let firstMsg = userFirstMessage;
  if (mode === "inner_os") firstMsg += INNER_OS_MARKER;
  if (mode === "no_inner_os") firstMsg += NO_INNER_OS_MARKER;

  return [
    { role: "system", content: systemPrompt },
    { role: "user",   content: firstMsg },
  ];
}

// Usage
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/v1",
});

const messages = buildMessages(
  "你是一个傲娇的女高中生。",
  "「我走进教室」"早上好。"",
  "inner_os"
);

const res = await client.chat.completions.create({
  model: "deepseek-v4-pro",
  messages,
});

console.log(res.choices[0].message.content);
```

---

## Common Patterns

### Reusable Roleplay Session Class (Python)

```python
class DeepSeekRoleplaySession:
    """Manages a stateful multi-turn roleplay conversation."""

    def __init__(
        self,
        system_prompt: str,
        model: str = "deepseek-v4-pro",
        mode: str = "inner_os",
    ):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self._first_turn = True
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def chat(self, user_message: str) -> str:
        if self._first_turn:
            if self.mode == "inner_os":
                user_message += INNER_OS_MARKER
            elif self.mode == "no_inner_os":
                user_message += NO_INNER_OS_MARKER
            self._first_turn = False

        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个在咖啡店打工的神秘少年。",
    mode="inner_os",
)

print(session.chat("「我推开门走进来」"今天有什么特调吗？""))
print(session.chat("「我在吧台旁坐下」"你平时喜欢喝什么？""))
```

### Switch Modes Between Conversations

```python
# Start a new session with a different mode — never reuse old session
analysis_session = DeepSeekRoleplaySession(
    system_prompt="你是一个冷静的侦探。",
    mode="no_inner_os",
)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Marker didn't trigger (think block still wrong style) | Re-roll — triggers are probabilistic, not guaranteed. Try 2–3 times. |
| Using web Quick Mode | Switch to **Expert Mode** (专家模式). Quick Mode is unsupported. |
| Putting marker in system prompt | Move it to the **first user message** — that's where the model was trained to read it. |
| Marker fires on turn 2+ | Only inject on turn 1; the history carries it forward automatically. |
| Think block not visible in web UI | Click "查看思考过程" to expand the reasoning panel. |
| `deepseek-v4-flash` vs `deepseek-v4-pro` | Both supported. `flash` is faster/cheaper; `pro` produces richer inner monologue. |

---

## Key Rules to Remember

1. **First user message only** — inject the marker once, never repeat it.
2. **Blank line separator** — when typing manually in web UI, leave a blank line between your message and the marker.
3. **Think block only** — markers control `<think>` content, not the final reply. Final reply style is only indirectly affected.
4. **New conversation = new mode** — to switch modes, start a fresh conversation.
5. **Probabilistic** — if the first generation doesn't show the expected think style, retry.
```
