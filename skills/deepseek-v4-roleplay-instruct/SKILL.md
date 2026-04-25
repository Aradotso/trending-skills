```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between immersive character inner monologue and pure analytical planning within <think> tags
triggers:
  - deepseek roleplay thinking mode
  - deepseek v4 inner monologue
  - control deepseek think tags roleplay
  - deepseek character immersion mode
  - deepseek pure analysis mode roleplay
  - switch deepseek thinking style
  - deepseek roleplay instruct marker
  - deepseek v4 flash pro roleplay api
---

# DeepSeek V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

DeepSeek-V4 Roleplay Instruct provides **special control instructions** that steer how DeepSeek-V4 reasons inside its `<think>` tags during roleplay conversations. By appending a marker to the **first user message**, you can toggle between:

| Mode | Effect on `<think>` |
|------|---------------------|
| **Default** | Model auto-selects reasoning style |
| **角色沉浸 (Character Immersion)** | First-person inner monologue wrapped in parentheses — the model "acts" the character while planning |
| **纯分析 (Pure Analysis)** | Cold, structured planning only — no in-character inner dialogue |

**Supported surfaces:**
- DeepSeek official APP / web — **Expert Mode** only
- API models: `deepseek-v4-flash`, `deepseek-v4-pro`
- Web "Quick Mode" is **not** supported

---

## Core Concept: The Marker Pattern

Append a mode marker to the **first user message only**. Because the model always sees full conversation history, that first message keeps influencing every subsequent turn automatically — no need to repeat it.

```
[Normal roleplay message]

[Mode marker appended here]
```

---

## The Two Markers (Copy-Paste Ready)

### Character Immersion Marker (角色沉浸)

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Marker (纯分析)

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
    Only the first user message needs the marker — subsequent turns are appended normally.
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

### Full Multi-Turn Example (OpenAI-compatible client)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",  # adjust to actual endpoint
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷淡，内心其实在意对方的一举一动。"

# --- Turn 1: inject marker into first user message ---
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
print("Turn 1 reply:", reply)

# --- Turn 2+: append normally, marker stays in history automatically ---
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
reply2 = response.choices[0].message.content
print("Turn 2 reply:", reply2)
```

### Streaming Example

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

messages = build_messages(
    system_prompt="你是一个神秘的酒馆老板，见过世面，话不多但每句都有深意。",
    user_first_message="「一个风尘仆仆的旅人推开门走进来」"来一杯最烈的。"",
    mode="no_inner_os",  # pure analysis mode
)

with client.chat.completions.stream(
    model="deepseek-v4-flash",
    messages=messages,
) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
```

---

## Web / App Usage

Paste your roleplay message, add a blank line, then paste the marker:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

Subsequent messages need **no modification**:

```
Turn 2: 「我坐到窗边的位置」"来一杯美式。"
Turn 3: 「我注意到你手上有一道疤痕」"你的手……没事吧？"
```

Verify the mode is active by clicking **"查看思考过程"** (View Thinking Process).

---

## Mode Comparison

```
Character Immersion (inner_os):        Pure Analysis (no_inner_os):
──────────────────────────────         ──────────────────────────────
<think>                                <think>
（他跟我打招呼了……心跳加速。）            场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。                 回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）               控制 150 字，先动作描写再对话。
</think>                               </think>

→ Emotionally authentic replies        → Structurally consistent replies
→ Better for emotional/romance RP      → Better for complex plot management
```

---

## Reusable Roleplay Client Class

```python
import os
from openai import OpenAI
from typing import Literal

class DeepSeekRoleplayClient:
    """
    Stateful multi-turn roleplay client with thinking mode control.
    """

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
        mode: Literal["default", "inner_os", "no_inner_os"] = "default",
        model: str = "deepseek-v4-pro",
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
        """Start a new conversation, optionally with a new system prompt."""
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [self.messages[0]]
        self._first_turn = True


# Usage
rp = DeepSeekRoleplayClient(
    system_prompt="你是一个傲娇的女高中生。",
    mode="inner_os",
    model="deepseek-v4-flash",
)

print(rp.chat("「我走进教室」"早上好。""))
print(rp.chat("「我在她旁边坐下」"今天心情不好吗？""))

# Switch mode → open new conversation
rp.reset(system_prompt="你是一个冷酷的侦探。")
rp.mode = "no_inner_os"
rp._first_turn = True
print(rp.chat("「案发现场，我递给你一份报告」"这是今晨的尸检结果。""))
```

---

## Common Patterns

### Pattern 1 — Mode per conversation session

```python
# Emotional/romance roleplay → inner_os
romance_session = DeepSeekRoleplayClient(
    system_prompt="你是一个温柔体贴的咖啡师...",
    mode="inner_os",
)

# Plot-heavy / mystery roleplay → no_inner_os
mystery_session = DeepSeekRoleplayClient(
    system_prompt="你是一个老练的侦探...",
    mode="no_inner_os",
)
```

### Pattern 2 — Retry on mode non-compliance

The markers increase probability but don't guarantee 100% compliance. Retry on failure:

```python
import re

def chat_with_retry(client: DeepSeekRoleplayClient, message: str, max_retries: int = 3) -> str:
    """
    For inner_os mode: retry if no parenthesized inner monologue appears in <think>.
    """
    for attempt in range(max_retries):
        reply = client.chat(message)
        if client.mode != "inner_os":
            return reply
        # Heuristic: check think block contains （...）
        think_match = re.search(r"<think>(.*?)</think>", reply, re.DOTALL)
        if think_match and re.search(r"[（(].+?[）)]", think_match.group(1)):
            return reply
        if attempt < max_retries - 1:
            # Remove the last assistant message to retry same user turn
            client.messages.pop()
    return reply  # return last attempt regardless
```

### Pattern 3 — System prompt placement (FAQ)

```python
# ✅ Recommended: marker in first USER message (matches training injection point)
messages = [
    {"role": "system", "content": "你是..."},
    {"role": "user",   "content": "「场景」" + INNER_OS_MARKER},
]

# ⚠️ Less reliable: marker in system prompt
messages = [
    {"role": "system", "content": "你是..." + INNER_OS_MARKER},  # works but less stable
    {"role": "user",   "content": "「场景」"},
]
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Marker has no effect | Using web Quick Mode | Switch to **Expert Mode** |
| Marker has no effect | Wrong model | Use `deepseek-v4-flash` or `deepseek-v4-pro` |
| Mode stops working mid-conversation | First message was edited/deleted | Start a new conversation with the marker in turn 1 |
| Inner monologue not appearing consistently | Probabilistic — not 100% guaranteed | Re-roll / retry the same message |
| Want to switch modes | Can't change mid-conversation | Open a new conversation, paste the other marker in turn 1 |
| `<think>` tags not visible | Platform hides reasoning | Click "查看思考过程" on web; check API response `reasoning_content` field if available |

---

## Key Rules to Remember

1. **Marker goes in turn 1 user message** — not system prompt, not later turns
2. **One blank line** between roleplay content and the marker (web UI)
3. **Subsequent turns need nothing** — history carries the marker forward
4. **Not 100% deterministic** — retry if a turn doesn't comply
5. **Only affects `<think>` content** — the final reply format is unchanged, but thinking style influences reply quality
6. **New mode = new conversation** — you cannot switch mid-session
```
