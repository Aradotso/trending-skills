```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4's thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning inside <think> tags.
triggers:
  - "add inner monologue to DeepSeek roleplay"
  - "control DeepSeek thinking mode"
  - "DeepSeek roleplay immersion mode"
  - "switch DeepSeek think tag style"
  - "DeepSeek character inner OS monologue"
  - "pure analysis mode DeepSeek"
  - "DeepSeek V4 roleplay instruct"
  - "append roleplay marker to DeepSeek message"
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 (models: `deepseek-v4-flash`, `deepseek-v4-pro`) supports extended thinking via `<think>` tags. By default the model decides how to use that thinking space. This project documents **two special control instructions** you append to the **first user message** of a conversation to steer the thinking style:

| Mode | Effect inside `<think>` |
|---|---|
| **Default** | Model chooses automatically |
| **角色沉浸 (Inner OS)** | First-person character inner monologue wrapped in `（…）` brackets |
| **纯分析 (Pure Analysis)** | Pure logical planning, no inner monologue, no brackets |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only
- API: `deepseek-v4-flash`, `deepseek-v4-pro`
- ⚠️ Web **Quick Mode** is NOT supported

---

## Core Concept: Where to Inject

The marker goes at the **end of the first `user` message**, not in `system`. This matches the training-time injection position and gives the most reliable results.

The instruction persists because every subsequent API call includes the full conversation history — the marker in turn 1 stays in context for all future turns automatically.

---

## The Two Marker Strings

### Inner OS Mode (角色沉浸)

```python
INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"\n"
    "2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等\n"
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)
```

### Pure Analysis Mode (纯分析)

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

No package to install. This is a prompting pattern. To use with the DeepSeek API:

```bash
# Set your API key
export DEEPSEEK_API_KEY=your_key_here
```

```python
pip install openai  # DeepSeek API is OpenAI-compatible
```

---

## Full Python Integration

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# ── Marker constants ──────────────────────────────────────────────────────────

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

# ── Message builder ───────────────────────────────────────────────────────────

def build_first_message(user_text: str, mode: str = "default") -> str:
    """Append the appropriate marker to the first user message."""
    if mode == "inner_os":
        return user_text + INNER_OS_MARKER
    elif mode == "no_inner_os":
        return user_text + NO_INNER_OS_MARKER
    return user_text  # default: no modification


def build_messages(
    system_prompt: str,
    first_user_message: str,
    mode: str = "default",
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": build_first_message(first_user_message, mode)},
    ]


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

# ── Multi-turn roleplay session ───────────────────────────────────────────────

def roleplay_session(system_prompt: str, mode: str = "inner_os"):
    """
    Run an interactive multi-turn roleplay session.
    The marker is injected only once (turn 1); subsequent turns are plain text.
    """
    messages = []
    first_turn = True

    print(f"[Mode: {mode}] Type your message. Ctrl+C to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if first_turn:
            # Inject system + first user message with marker
            messages = build_messages(system_prompt, user_input, mode=mode)
            first_turn = False
        else:
            # Subsequent turns: just append normally
            messages.append({"role": "user", "content": user_input})

        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYSTEM = "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。"

    # --- Inner OS mode (immersive) ---
    messages = build_messages(
        system_prompt=SYSTEM,
        first_user_message="「我走进教室」"早上好。"",
        mode="inner_os",
    )
    reply = chat(messages)
    print("Turn 1:", reply)

    # Turn 2 onward: no special handling needed
    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})
    reply2 = chat(messages)
    print("Turn 2:", reply2)
```

---

## TypeScript / Node.js Integration

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY!,
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

function buildFirstMessage(text: string, mode: Mode): string {
  if (mode === "inner_os") return text + INNER_OS_MARKER;
  if (mode === "no_inner_os") return text + NO_INNER_OS_MARKER;
  return text;
}

async function startRoleplay(
  systemPrompt: string,
  firstUserMessage: string,
  mode: Mode = "inner_os",
  model = "deepseek-v4-flash"
) {
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user",   content: buildFirstMessage(firstUserMessage, mode) },
  ];

  const response = await client.chat.completions.create({ model, messages });
  const reply = response.choices[0].message.content ?? "";
  messages.push({ role: "assistant", content: reply });
  return { reply, messages };
}

// Usage
const { reply, messages } = await startRoleplay(
  "你是一个傲娇的女高中生，表面冷淡，内心其实很在意对方。",
  "「我走进教室」"早上好。"",
  "inner_os"
);
console.log(reply);

// Continue conversation
messages.push({ role: "user", content: "「我在她旁边坐下」"今天心情不好吗？"" });
const turn2 = await client.chat.completions.create({
  model: "deepseek-v4-flash",
  messages,
});
console.log(turn2.choices[0].message.content);
```

---

## Web / Manual Usage

Paste your message + marker directly into the DeepSeek web UI (Expert Mode):

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

After that, type normally — no need to repeat the marker.

---

## Expected Think Tag Output

**Inner OS mode:**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

**Pure Analysis mode:**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Common Patterns

### Pattern 1: Reusable session class

```python
class DeepSeekRoleplay:
    def __init__(self, system_prompt: str, mode: str = "inner_os",
                 model: str = "deepseek-v4-flash"):
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.mode = mode
        self._first_turn = True

    def send(self, user_message: str) -> str:
        if self._first_turn:
            user_message = build_first_message(user_message, self.mode)
            self._first_turn = False
        self.messages.append({"role": "user", "content": user_message})
        reply = chat(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Start a new conversation (keep system prompt & mode)."""
        system = self.messages[0]
        self.messages = [system]
        self._first_turn = True


# Usage
rp = DeepSeekRoleplay("你是一个冷漠的图书馆员...", mode="inner_os")
print(rp.send("「我走进图书馆，向你走近」"请问有没有……""))
print(rp.send("「我注意到你的眼神变了一下」"))
```

### Pattern 2: Switch modes across sessions

```python
# Session A — immersive
session_a = DeepSeekRoleplay(SYSTEM, mode="inner_os")

# Session B — analytical (e.g. for debugging / quality review)
session_b = DeepSeekRoleplay(SYSTEM, mode="no_inner_os")
```

### Pattern 3: Validate marker was respected

```python
import re

def verify_think_mode(reply_with_think: str, mode: str) -> bool:
    """
    Crude heuristic check on raw streamed content that includes <think> block.
    """
    think_match = re.search(r"<think>(.*?)</think>", reply_with_think, re.DOTALL)
    if not think_match:
        return False
    think_content = think_match.group(1)
    has_brackets = bool(re.search(r"[（(].+?[）)]", think_content))
    if mode == "inner_os":
        return has_brackets
    elif mode == "no_inner_os":
        return not has_brackets
    return True
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Marker seems ignored | Model didn't follow instruction this turn | Re-roll (send again); compliance is probabilistic |
| Inner monologue appears in `no_inner_os` mode | Model slipped | Re-roll; or add stronger negation in system prompt |
| `<think>` tags not visible | Wrong model or mode (Quick Mode on web) | Use Expert Mode on web or `deepseek-v4-flash`/`deepseek-v4-pro` via API |
| Marker placed in `system` prompt | Wrong injection position | Move marker to **end of first `user` message** |
| Marker not persisting after turn 1 | Not including full conversation history | Always pass the full `messages` list to each API call |
| Want to change mode mid-conversation | Modes are per-conversation | Start a new conversation with the new marker |

---

## Key Facts for AI Coding Agents

- **Inject location**: end of `messages[1]["content"]` (first user turn), not system prompt
- **One-time injection**: marker in turn 1 persists via conversation history; never re-inject
- **Models**: `deepseek-v4-flash` (faster), `deepseek-v4-pro` (more capable)
- **Base URL**: `https://api.deepseek.com/v1` (OpenAI-compatible)
- **Auth**: `Authorization: Bearer $DEEPSEEK_API_KEY`
- **Compliance**: probabilistic (~not 100%); retry on failure
- **Effect scope**: thinking process only (`<think>` tags); final reply style is indirectly affected
```
