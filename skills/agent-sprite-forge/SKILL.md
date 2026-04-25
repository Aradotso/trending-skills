```markdown
---
name: agent-sprite-forge
description: Generate game-ready 2D sprite sheets, transparent PNG frames, and animated GIFs from natural-language prompts using the Codex agent skill $generate2dsprite.
triggers:
  - generate a 2D sprite
  - create a sprite sheet
  - make a pixel art animation
  - use generate2dsprite to create
  - create an animated GIF from a prompt
  - generate a sprite bundle with cast projectile impact
  - make a four-direction walk sprite sheet
  - create game assets from a text prompt
---

# Agent Sprite Forge

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Agent Sprite Forge is a Codex agent skill (`$generate2dsprite`) that turns natural-language prompts into game-ready 2D pixel-art assets. The agent plans the asset, calls Codex's built-in image generation, then runs a deterministic local Python post-processor to clean up backgrounds, split frames, align cells, and export transparent PNGs and animated GIFs.

---

## How It Works

1. **Agent plans** the sheet layout, frame count, view angle, and action type.
2. **Codex generates** the raw sprite sheet image (built-in image generation — no external API needed).
3. **Local post-processor** (`generate2dsprite.py`) performs:
   - Magenta (`#FF00FF`) chroma-key background removal
   - Frame splitting based on grid layout
   - Bounding-box extraction and alignment
   - Rescaling
   - Transparent PNG and animated GIF export

---

## Installation

### macOS / Linux

```bash
git clone https://github.com/0x0funky/agent-sprite-forge.git
cd ./agent-sprite-forge
python3 -m pip install -r ./requirements.txt
mkdir -p ~/.codex/skills
cp -R ./skills/generate2dsprite ~/.codex/skills/generate2dsprite
```

### Windows PowerShell

```powershell
git clone https://github.com/0x0funky/agent-sprite-forge.git
cd .\agent-sprite-forge
python -m pip install -r .\requirements.txt
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.codex\skills" | Out-Null
Copy-Item -Recurse -Force `
  ".\skills\generate2dsprite" `
  "$env:USERPROFILE\.codex\skills\generate2dsprite"
```

Start a **new Codex session** after installation so the skill is loaded cleanly.

### Python Dependencies

```text
Pillow
numpy
```

Install manually if needed:

```bash
pip install Pillow numpy
```

---

## Repository Layout

```text
agent-sprite-forge/
  requirements.txt
  skills/
    generate2dsprite/
      SKILL.md
      agents/
        openai.yaml          # Codex agent config
      references/
        modes.md             # Asset type / mode reference
        prompt-rules.md      # Prompt authoring rules
      scripts/
        generate2dsprite.py  # Local post-processor
```

---

## Invoking the Skill

In a Codex session, use `$generate2dsprite` in your prompt. The agent reads `SKILL.md`, plans the asset, generates the sheet, then calls the local script.

### Basic invocation pattern

```text
Use $generate2dsprite to create <description>.
```

---

## Prompt Examples by Asset Type

### Idle / Creature

```text
Use $generate2dsprite to create a 3x3 idle for an ultimate earth titan.
```

```text
Use $generate2dsprite to create a golden divine boar 2x2 idle animation.
```

### Character Attack

```text
Use $generate2dsprite to create a side-view lightning knight attack animation.
```

```text
Use $generate2dsprite to create Omegamon attack and right-move animation assets.
```

### Spell Bundle (Cast + Projectile + Impact)

```text
Use $generate2dsprite to create a fire mage cast animation with projectile and impact.
```

```text
Use $generate2dsprite to create a wizard spell bundle with cast, projectile, and impact sprites.
```

```text
Use $generate2dsprite to create a fireball projectile loop and a matching explosion impact.
```

### Four-Direction Walk Sheet (top-down RPG)

```text
Use $generate2dsprite to create a top-down 4x4 player_sheet for a wandering young samurai
with a red scarf and short katana. Make a four-direction walk sprite sheet with 4 frames
per direction. Row order: down, left, right, up. Same character, same outfit, same
proportions, same pixel scale in every frame. Solid #FF00FF background. Each frame must
fit fully inside its cell, with clear margin on all sides. Retro JRPG pixel-art style.
```

### FX / Summon Effect

```text
Use $generate2dsprite to create a side-view summon entrance effect for a thunder wolf spirit.
```

### Reference Image → Sprite

Attach or reference an image in your Codex session:

```text
Use $generate2dsprite to create this male character teaching animation.
```

```text
Use $generate2dsprite — make an animation of this crocodile playing with the stone in its hand.
```

### Full Playable Game (One-Shot)

```text
Use $generate2dsprite to create a 2D side-scrolling game similar to Mega Man. It should
include attack mechanics, map elements, and all essential features. Design everything and
create all assets using this skill. It must be an actually playable game with a cyberpunk
story setting.
```

---

## What the Post-Processor Produces

For a typical sprite sheet, the output directory contains:

| File | Description |
|---|---|
| `raw-sheet.png` | Original image from Codex generation |
| `raw-sheet-clean.png` | After chroma-key removal |
| `sheet-transparent.png` | Full sheet with transparency |
| `frame-00.png`, `frame-01.png`, … | Individual extracted frames |
| `animation.gif` | Assembled animated GIF |
| `prompt-used.txt` | The exact prompt sent to generation |
| `pipeline-meta.json` | Grid layout, frame count, frame size metadata |

For **player walk sheets**, additional per-direction outputs are generated:

| File | Description |
|---|---|
| `strip-down.png` | Horizontal strip for down direction |
| `strip-left.png` | Horizontal strip for left direction |
| `strip-right.png` | Horizontal strip for right direction |
| `strip-up.png` | Horizontal strip for up direction |
| `down.gif` | Animated GIF for down walk |
| `left.gif` | Animated GIF for left walk |
| `right.gif` | Animated GIF for right walk |
| `up.gif` | Animated GIF for up walk |

---

## Running the Post-Processor Directly

If you have a raw sheet image and want to run post-processing manually:

```python
# Example: split a 3x3 sheet and export transparent frames + GIF
from PIL import Image
import numpy as np

def remove_magenta(img: Image.Image, tolerance: int = 30) -> Image.Image:
    """Replace #FF00FF background with transparency."""
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.int32)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    mask = (
        (r > 255 - tolerance) &
        (g < tolerance) &
        (b > 255 - tolerance)
    )
    data[..., 3] = np.where(mask, 0, a)
    return Image.fromarray(data.astype(np.uint8), "RGBA")

def split_sheet(sheet_path: str, cols: int, rows: int, output_dir: str):
    """Split a sprite sheet into individual transparent frame PNGs and a GIF."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    sheet = Image.open(sheet_path).convert("RGBA")
    total_w, total_h = sheet.size
    frame_w = total_w // cols
    frame_h = total_h // rows

    frames = []
    for row in range(rows):
        for col in range(cols):
            left = col * frame_w
            top = row * frame_h
            cell = sheet.crop((left, top, left + frame_w, top + frame_h))
            clean = remove_magenta(cell)
            idx = row * cols + col
            out_path = os.path.join(output_dir, f"frame-{idx:02d}.png")
            clean.save(out_path)
            frames.append(clean)
            print(f"Saved {out_path}")

    # Export animated GIF
    gif_path = os.path.join(output_dir, "animation.gif")
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=100,  # ms per frame
            disposal=2,
        )
        print(f"Saved {gif_path}")

# Usage
split_sheet("raw-sheet.png", cols=3, rows=3, output_dir="./output")
```

---

## Asset Types and Modes

Reference `skills/generate2dsprite/references/modes.md` for the full list. Common modes:

| Mode | Description |
|---|---|
| `idle` | Looping idle animation (e.g., `3x3 idle`) |
| `attack` | Attack action frames |
| `walk` / `player_sheet` | Four-direction walk sheet (top-down RPG) |
| `spell_bundle` | Cast + projectile + impact as separate sheets |
| `unit_bundle` | Character unit with multiple action strips |
| `combat_bundle` | Full combat set: idle, attack, hit, death |
| `projectile` | Single-action projectile loop |
| `impact` / `explosion` | Impact / explosion FX sheet |
| `fx_sheet` | General visual effects sheet |

---

## Prompt Authoring Rules

From `skills/generate2dsprite/references/prompt-rules.md`:

- **Specify view angle**: `side-view`, `top-down`, `isometric`, `front-facing`
- **Specify grid layout**: e.g., `3x3`, `2x4`, `1x6` (columns × rows)
- **Name the action**: `idle`, `attack`, `walk`, `cast`, `death`
- **Use magenta background**: always specify `solid #FF00FF background` for clean chroma-key removal
- **Describe style**: `retro JRPG pixel-art`, `16-bit`, `32x32`, `64x64`
- **Keep character consistent**: add `Same character, same outfit, same proportions, same pixel scale in every frame`
- **Set margins**: `Each frame must fit fully inside its cell, with clear margin on all sides`

### Good prompt structure

```text
Use $generate2dsprite to create a [view] [grid] [action] for [character description].
[Style notes.] [Consistency notes.] Solid #FF00FF background.
Each frame must fit fully inside its cell, with clear margin on all sides.
```

---

## Tips for Best Results

| Situation | Recommendation |
|---|---|
| Large creature / boss | Use `3x3 idle` or `3x3 attack` |
| Small spell or projectile | Use `1x4`, `2x2`, or `2x3` |
| Top-down RPG player | Use `player_sheet` mode, specify row order explicitly |
| Spell with phases | Use `spell_bundle` — agent generates cast, projectile, impact as separate assets |
| Reference image → sprite | Attach the image in Codex, describe the desired action |
| Inconsistent frames | Add "Same character, same outfit, same proportions, same pixel scale in every frame" to prompt |
| Clipped sprites | Add "Each frame must fit fully inside its cell, with clear margin on all sides" |
| Commercial use | Use original characters or IP you control |

---

## Troubleshooting

### Skill not found after install

Restart the Codex session. The skill directory must exist at `~/.codex/skills/generate2dsprite/` before the session starts.

### `ModuleNotFoundError: No module named 'PIL'`

```bash
pip install Pillow
```

### Magenta not fully removed / fringing

The default tolerance is `30`. If pink fringing remains around edges, increase it in the post-processor call, or add `anti-aliasing: none, hard pixel edges` to your generation prompt.

### Frames misaligned or wrong count

Explicitly state the grid in the prompt (e.g., `3x4 sheet, 3 columns, 4 rows`). Ambiguous layouts cause the agent to guess.

### GIF has no transparency

Ensure the chroma-key step runs before GIF assembly. The transparent frames must be RGBA mode. If using `disposal=2`, confirm Pillow version ≥ 9.0.

### Generated image not game-proportional

Add explicit pixel dimensions to the prompt: `each frame 64x64 pixels, total sheet 192x192`.

---

## License

MIT — see [LICENSE](https://github.com/0x0funky/agent-sprite-forge/blob/main/LICENSE).
```
