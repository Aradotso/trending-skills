```markdown
---
name: deepseek-v4-roleplay-instruct
description: Control DeepSeek-V4 thinking mode during roleplay — switch between character-immersive inner monologue and pure analytical reasoning using special prompt markers.
triggers:
  - deepseek roleplay thinking mode
  - deepseek inner monologue thinking
  - deepseek v4 roleplay instructions
  - switch deepseek thinking style
  - deepseek character immersion mode
  - deepseek pure analysis mode
  - deepseek roleplay control markers
  - deepseek think tag roleplay
---

# DeepSeek-V4 Roleplay Instruct

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

DeepSeek-V4 Roleplay Instruct provides **special control markers** that steer how DeepSeek-V4 reasons inside its `<think>` tags during roleplay conversations. By appending a marker to the **first user message**, you can choose between:

- **Character Immersion Mode** — the model thinks in first-person inner monologue, like an actor in character.
- **Pure Analysis Mode** — the model thinks in cold, structured reasoning, like a director planning a scene.
- **Default** — the model auto-selects based on scene complexity.

**Supported surfaces:**
- DeepSeek official APP / Web (Expert Mode)
- `deepseek-v4-flash` API
- `deepseek-v4-pro` API

> ⚠️ Web Quick Mode is **not** supported. Trigger rate is probabilistic — retry if the marker doesn't take effect on the first attempt.

---

## How It Works

The marker is injected once into the **first user message**. Because DeepSeek includes the full conversation history in every inference call, the marker stays in context for all subsequent turns automatically — no need to repeat it.

---

## Marker Reference

### Character Immersion Mode (`inner_os`)

Causes `<think>` content to use bracketed first-person inner monologue: `（心想：……）` / `(内心OS：……)`

```
【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

### Pure Analysis Mode (`no_inner_os`)

Causes `<think>` content to use only structured, analytical language — no inner monologue, no parenthetical emotions.

```
【思维模式要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 禁止使用圆括号包裹内心独白，例如"（心想：……）"或"(内心OS：……)"，所有分析内容直接陈述即可
2. 禁止以角色第一人称描写内心活动，例如"我心想""我觉得""我暗自"等，请用分析性语言替代
3. 思考内容应聚焦于剧情走向分析和回复内容规划，不要在思考中进行角色扮演式的内心戏表演
```

---

## Thinking Output Comparison

```
# Character Immersion Mode (inner_os)        # Pure Analysis Mode (no_inner_os)
<think>                                       <think>
（他跟我打招呼了……心跳加速。）                场景：用户打招呼，角色是傲娇属性。
我要装作不在意的样子回应。                     回复策略：先嫌弃，身体语言暴露真情。
（不能让他看出来我很高兴！）                   控制 150 字，先动作描写再对话。
</think>                                      </think>
```

---

## Python API Integration

### Marker Constants

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
def build_messages(system_prompt: str, user_first_message: str, mode: str = "default") -> list[dict]:
    """
    Build the messages array for a DeepSeek-V4 roleplay conversation.

    Args:
        system_prompt: Character/scene setup for the model.
        user_first_message: The opening user turn (action + dialogue).
        mode: "inner_os" | "no_inner_os" | "default"

    Returns:
        List of message dicts ready to pass to the API.
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

### Full Multi-Turn Example

```python
import os
from openai import OpenAI  # DeepSeek is OpenAI-compatible

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

SYSTEM_PROMPT = "你是一个傲娇的女高中生，表面冷淡，内心在意对方的一举一动。"

# --- Turn 1: inject marker once ---
messages = build_messages(
    system_prompt=SYSTEM_PROMPT,
    user_first_message="「我走进教室」"早上好。"",
    mode="inner_os",          # or "no_inner_os" or "default"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # or "deepseek-v4-flash"
    messages=messages,
)
assistant_reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_reply})

# --- Turn 2+: append normally, marker stays in history ---
messages.append({"role": "user", "content": "「我在她旁边坐下」"今天心情不好吗？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
assistant_reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_reply})

# --- Turn 3 ---
messages.append({"role": "user", "content": "「我注意到她手上有一道疤痕」"你的手……没事吧？""})

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)
print(response.choices[0].message.content)
```

### Accessing the Think Block

```python
# If the API surfaces reasoning_content separately:
choice = response.choices[0]
if hasattr(choice.message, "reasoning_content"):
    print("=== THINK ===")
    print(choice.message.reasoning_content)

print("=== REPLY ===")
print(choice.message.content)
```

---

## Web / App Usage

1. Open a **new conversation** in DeepSeek APP or web in **Expert Mode**.
2. Write your opening message, leave a blank line, then paste the desired marker:

```
「我推开咖啡店的门，看到你正在擦吧台。」"你好，请问还有位置吗？"

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复
```

3. Send. All subsequent messages in this conversation need **no marker** — just chat normally.
4. Click **"查看思考过程"** (View thinking process) to verify the mode is active.

To switch modes: start a **new conversation** and paste the other marker in the first message.

---

## Common Patterns

### Pattern: Reusable Roleplay Session Class

```python
import os
from openai import OpenAI

class DeepSeekRoleplaySession:
    def __init__(self, system_prompt: str, mode: str = "inner_os", model: str = "deepseek-v4-pro"):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.mode = mode
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._first_turn = True

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

    def reset(self, mode: str | None = None):
        """Start a fresh conversation, optionally switching mode."""
        if mode:
            self.mode = mode
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._first_turn = True


# Usage
session = DeepSeekRoleplaySession(
    system_prompt="你是一个神秘的图书管理员，知道每本书背后的秘密。",
    mode="inner_os",
)

print(session.send("「我走进昏暗的图书馆」"请问……有关于消失的人的书吗？""))
print(session.send("「我在书架间穿行，注意到一本没有书名的书」"这本是什么？""))

# Switch to analysis mode for a new scene
session.reset(mode="no_inner_os")
print(session.send("「新场景开始」"你好。""))
```

### Pattern: Batch Testing Both Modes

```python
def compare_modes(system_prompt: str, opening_message: str) -> dict:
    """Run the same opening in both modes and return thinking + reply for each."""
    results = {}
    for mode in ("inner_os", "no_inner_os"):
        messages = build_messages(system_prompt, opening_message, mode=mode)
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
        )
        msg = response.choices[0].message
        results[mode] = {
            "thinking": getattr(msg, "reasoning_content", None),
            "reply": msg.content,
        }
    return results
```

---

## Configuration Reference

| Parameter | Values | Notes |
|-----------|--------|-------|
| `mode` | `"inner_os"` / `"no_inner_os"` / `"default"` | Controls which marker (if any) is appended |
| `model` | `"deepseek-v4-pro"` / `"deepseek-v4-flash"` | Both support the markers via API |
| Marker injection point | First user message only | Injecting in system prompt is less stable |
| Subsequent turns | No marker needed | Full history keeps marker in context |

---

## Troubleshooting

**Marker didn't take effect (thinking style unchanged)**
- The trigger is probabilistic, not guaranteed. Re-generate the response (retry the API call or click regenerate in the web UI).
- Confirm you're using **Expert Mode** on web, not Quick Mode.
- Confirm the marker is appended to the **first user message**, not the system prompt.

**Mode keeps reverting after several turns**
- The marker should persist as long as the full conversation history is included in each API call. Ensure your `messages` list accumulates all turns and is passed completely each time.

**`reasoning_content` field is missing**
- Some API versions surface thinking inside `<think>...</think>` tags within `content` instead. Parse it with:
  ```python
  import re
  match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
  thinking = match.group(1).strip() if match else None
  ```

**Want to disable the marker mid-conversation**
- You cannot remove the marker from history mid-conversation. Start a new conversation without any marker (default mode).

**System prompt vs. user message injection**
- Always inject into the **first user message**. The model was trained with the marker at that position; system prompt injection has lower reliability.
```
