```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning in the <think> block.
triggers:
  - deepseek roleplay thinking mode
  - switch deepseek inner monologue
  - deepseek v4 roleplay instruct
  - control deepseek think block
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay marker injection
  - deepseek think tag roleplay
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct is a prompt-engineering technique for controlling the **thinking style** inside DeepSeek-V4's `<think>` block during roleplay conversations. By appending a special marker to the **first user message**, you can switch between:

- **Character Immersion Mode** — the model thinks in first-person inner monologue (like an actor in-character)
- **Pure Analysis Mode** — the model thinks in cold, structured, third-person analysis (like a director planning the scene)
- **Default** — the model picks automatically based on scene complexity

Supported surfaces:
- DeepSeek official APP / web (Expert Mode only)
- `deepseek-v4-flash` and `deepseek-v4-pro` APIs

---

## Core Concept

The marker is appended **once**, to the **first user message only**. Because the model sees full conversation history on every turn, the marker stays in context for the entire session — no need to repeat it.

---

## Marker Strings

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

## Installation / Setup

No package to install. This is a pure prompt-engineering pattern. You need:

1. A DeepSeek API key set as an environment variable:
   ```bash
   export DEEPSEEK_API_KEY=your_key_here
   ```

2. The DeepSeek Python SDK or any OpenAI-compatible client (DeepSeek uses an OpenAI-compatible API):
   ```bash
   pip install openai
   ```

---

## Complete Working Example

```python
import os
from openai import OpenAI

# DeepSeek uses an OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# ── Marker definitions ──────────────────────────────────────────────────────

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

# ── Helper ──────────────────────────────────────────────────────────────────

def build_messages(
    system_prompt: str,
    user_first_message: str,
    mode: str = "default",  # "inner_os" | "no_inner_os" | "default"
) -> list[dict]:
    """
    Inject the thinking-mode marker into the first user message.
    Only call this for the FIRST turn; subsequent turns append normally.
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


def chat(messages: list[dict], model: str = "deepseek-v4-flash") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Multi-turn roleplay session ─────────────────────────────────────────────

SYSTEM = "你是一个傲娇的女高中生，表面上冷漠但内心其实很在意对方。"

# Turn 1 — inject marker here and only here
messages = build_messages(
    system_prompt=SYSTEM,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",   # character immersion
)
reply = chat(messages)
print("Assistant:", reply)

# Turn 2+ — just append, marker stays in history automatically
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

## Mode Comparison

| Mode | Marker constant | `<think>` style |
|---|---|---|
| `"default"` | *(none)* | Model decides automatically |
| `"inner_os"` | `INNER_OS_MARKER` | First-person inner monologue in `（）` brackets |
| `"no_inner_os"` | `NO_INNER_OS_MARKER` | Third-person analytical planning, no brackets |

**Inner OS example think block:**
```
<think>
（他跟我打招呼了……心跳加速。）
我要装作不在意的样子回应。
（不能让他看出来我很高兴！）
</think>
```

**Pure analysis example think block:**
```
<think>
场景：用户打招呼，角色是傲娇属性。
回复策略：先嫌弃，身体语言暴露真情。
控制 150 字，先动作描写再对话。
</think>
```

---

## Reusable Session Class

```python
class DeepSeekRoleplaySession:
    """Manages a stateful multi-turn roleplay session with DeepSeek-V4."""

    def __init__(
        self,
        system_prompt: str,
        mode: str = "inner_os",
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self._first_turn = True

        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def send(self, user_message: str) -> str:
        if self._first_turn:
            # Initialise with system prompt + marker-injected first message
            self.messages = build_messages(
                self.system_prompt, user_message, mode=self.mode
            )
            self._first_turn = False
        else:
            self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.messages = []
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个傲娇的女高中生。",
    mode="inner_os",
    model="deepseek-v4-pro",
)

print(session.send("「我走进教室」"早上好。""))
print(session.send("「我在她旁边坐下」"今天心情不好吗？""))
print(session.send("「我注意到她手上有一道疤痕」"你的手……没事吧？""))
```

---

## Web / Manual Usage

Paste the marker directly after your first message in the chat input (separate with a blank line):

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages are sent normally — no marker needed.

---

## Common Patterns

### Pattern 1 — Start with analysis, switch to immersion mid-story
```python
# Not supported mid-session (marker must be in turn 1).
# Start a NEW session with the desired mode instead:
session.reset()
# or
session = DeepSeekRoleplaySession(system_prompt=SYSTEM, mode="inner_os")
```

### Pattern 2 — Default mode (no marker)
```python
messages = build_messages(SYSTEM, first_message, mode="default")
# Equivalent to a plain API call with no special instructions
```

### Pattern 3 — Batch experiment across modes
```python
for mode in ("default", "inner_os", "no_inner_os"):
    msgs = build_messages(SYSTEM, "「我走进教室」"早上好。"", mode=mode)
    reply = chat(msgs)
    print(f"[{mode}]\n{reply}\n")
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Marker has no effect | Placed in system prompt instead of first user message | Move marker to end of first **user** message |
| Marker works on turn 1 but fades later | Marker was appended to turn 2+ instead of turn 1 | Inject once into turn 1 only; history carries it forward |
| Mode triggers inconsistently | Probabilistic by design | Re-roll (send again); ~stable but not 100% guaranteed |
| Using web Quick Mode | Quick Mode is unsupported | Switch to **Expert Mode** in DeepSeek web UI |
| `AuthenticationError` | Missing or invalid API key | Ensure `DEEPSEEK_API_KEY` env var is set correctly |
| Model name not found | Wrong model identifier | Use `deepseek-v4-flash` or `deepseek-v4-pro` |

---

## Key Rules Summary

1. **Inject once** — marker goes in the first user message, never repeated.
2. **User message, not system prompt** — training was done with the marker in user position.
3. **Full conversation history** — DeepSeek sees all prior turns, so the marker persists automatically.
4. **Probabilistic** — if a mode doesn't trigger, retry; it raises probability, not certainty.
5. **Think block only** — the marker affects `<think>` content, not the final reply text (though thinking style indirectly shapes reply quality).
```
