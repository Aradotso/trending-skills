```markdown
---
name: awesome-gpt-image-2-prompts
description: Curated GPT-Image-2 prompt patterns and image examples for portraits, posters, UI mockups, character sheets, and community experiments via Evolink.
triggers:
  - "generate a GPT-Image-2 prompt"
  - "write a prompt for gpt image 2"
  - "create an image prompt for portraits"
  - "help me write a poster prompt"
  - "character sheet prompt for ai image"
  - "ui mockup prompt generation"
  - "gpt-image-2 prompt examples"
  - "evolink image generation prompt"
---

# awesome-gpt-image-2-prompts

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A curated collection of high-quality prompts and image examples for **GPT-Image-2**, organized by category: portraits, posters, character design, UI mockups, and community experiments. All examples are sourced from X/Twitter, creator communities, and public demos. Images can be generated via [Evolink](https://evolink.ai/models).

---

## What This Project Is

This is an **awesome-list repository** — it contains no executable code or installable package. It is a structured Markdown reference of:

- **Prompt text** you can copy and use directly with GPT-Image-2
- **Output image examples** for visual reference
- **Category-organized cases** with community attribution

The primary artifact is the `README.md` (and localized variants) containing prompt–image pairs.

---

## How to Use These Prompts

### Step 1: Browse the Repository

Clone or browse the repo to find prompts by category:

```bash
git clone https://github.com/EvoLinkAI/awesome-gpt-image-2-prompts.git
cd awesome-gpt-image-2-prompts
```

Categories in `README.md`:
- `## Portrait & Photography Cases`
- `## Poster & Illustration Cases`
- `## Character Design Cases`
- `## UI & Social Media Mockup Cases`
- `## Comparison & Community Examples`

### Step 2: Use a Prompt via Evolink API

Copy a prompt from the README and send it to GPT-Image-2 via the [Evolink](https://evolink.ai/models) platform or API.

```python
import os
import requests

EVOLINK_API_KEY = os.environ["EVOLINK_API_KEY"]

prompt = """
35mm film photography with harsh convenience store fluorescent lighting mixed 
with colorful neon signs from outside, authentic film grain, high contrast, 
slight color cast, cinematic street editorial style, intimate medium shot...
"""

response = requests.post(
    "https://api.evolink.ai/v1/images/generations",
    headers={
        "Authorization": f"Bearer {EVOLINK_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-image-2",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    },
)

data = response.json()
image_url = data["data"][0]["url"]
print(f"Generated image: {image_url}")
```

### Step 3: Use via OpenAI-Compatible API

GPT-Image-2 on Evolink is OpenAI API-compatible:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

response = client.images.generate(
    model="gpt-image-2",
    prompt="Vintage Amalfi travel poster, retro style, warm Mediterranean palette, hand-lettered typography, sun and sea motifs, aged paper texture",
    n=1,
    size="1024x1792",  # portrait
)

print(response.data[0].url)
```

---

## Prompt Anatomy & Patterns

### Portrait Prompt Structure

Effective portrait prompts from this collection follow this pattern:

```
[camera/film style] + [lighting setup] + [subject description] +
[clothing] + [pose/expression] + [background] + [post-processing/color grade]
```

**Example (condensed):**
```
35mm film photography, harsh fluorescent + neon lighting, early 20s Chinese female,
oversized white shirt + black mini skirt, leaning on convenience store door at night,
seductive gaze, pink/blue neon color cast, authentic film grain
```

### Poster Prompt Structure

```
[medium/art style] + [subject/content] + [color palette] +
[typography style] + [layout hints] + [era/mood]
```

**Example:**
```
Vintage travel poster, Amalfi Coast Italy, warm terracotta and azure palette,
hand-lettered serif typography, panoramic cliffside view, 1950s retro illustration style,
aged paper texture, decorative border
```

### Character Sheet Prompt Structure

```
[art style reference] + [character description] + [sheet layout type] +
[multiple views: front/side/back] + [expression sheet] + [color palette swatches]
```

**Example (Persona5 style):**
```
Persona 5 official character reference card, anime art style, teenage male protagonist,
black school uniform with red accents, front view / side view / back view arranged horizontally,
4 expression panels below, color palette swatches in corner, white background, clean lineart
```

### UI Mockup Prompt Structure

```
[device/platform type] + [screenshot style] + [app/content description] +
[UI elements] + [realistic photo framing]
```

**Example:**
```
Realistic iPhone 16 Pro screenshot, social media feed UI, Song Dynasty aesthetic theme,
ink wash painting posts, traditional Chinese typography, likes and comment counts visible,
photographed at slight angle on wooden desk, ambient warm light
```

---

## Category Reference

### Portrait & Photography

| Case | Style | Key Techniques |
|------|-------|----------------|
| Convenience Store Neon | 35mm film, editorial | Mixed fluorescent + neon, film grain |
| Cinematic Minimal | Cinematic | Minimal background, mood lighting |
| Japanese Onsen Ryokan | Lifestyle | Soft natural light, warm tones |
| 35mm Flash Editorial | Flash photography | Harsh flash, editorial framing |
| Mirror Selfie Bedroom | Casual/authentic | DIY aesthetic, natural clutter |
| Luxury Glam Beauty | Studio beauty | Dramatic rim light, glam makeup |

### Poster & Illustration

| Case | Style | Key Techniques |
|------|-------|----------------|
| Boston Spring City Poster | Modern city poster | Architecture silhouette, seasonal palette |
| Vintage Amalfi Travel | Retro travel | Hand-lettered, aged texture |
| Chengdu Food Map | Illustrated map | Isometric food icons, warm palette |
| Chinese Minimalist S-Shaped | Minimalist | Negative space, flowing composition |
| Super Famicom Poster | Pixel/retro game | 16-bit palette, box art layout |

### Character Design

| Case | Style | Key Techniques |
|------|-------|----------------|
| Anime Snapshot Conversion | Anime | Photo-to-anime conversion |
| Persona5 Reference Card | Game art | Multi-view layout, expression sheet |
| Gal Game Introduction Page | Visual novel | Decorative border, stat panels |
| Chibi Reference Sheet | SD/chibi | Deformed proportions, cute palette |
| Official Character Sheet (JP) | Anime official | Front/side/back, color swatches |

---

## Prompt Tuning Tips (from Case 13)

From `@kotsu_kotsu_san`'s iterative tuning case:

1. **Start broad, then narrow** — Begin with style + subject, add lighting/mood details in revisions
2. **Specify film stock or lens** — `35mm`, `85mm f/1.4`, `Kodak Portra 400` anchor the aesthetic
3. **Name the lighting setup explicitly** — `three-point studio`, `harsh single flash`, `diffused window light`
4. **Reference real art styles** — `Persona 5 official art`, `Studio Ghibli background`, `1950s travel poster`
5. **Describe negative space** — Tell the model what NOT to include: `no digital oversharpening`, `no plastic skin`
6. **Use aspect ratio hints in prompt** — `vertical composition`, `9:16 mobile format`, `landscape widescreen`
7. **Layer specificity** — Global style → subject → details → post-processing

---

## Multi-Language READMEs

The repository provides localized versions of the prompt collection:

| File | Language |
|------|----------|
| `README.md` | English (default) |
| `README_es.md` | Español |
| `README_pt.md` | Português |
| `README_ja.md` | 日本語 |
| `README_ko.md` | 한국어 |
| `README_de.md` | Deutsch |
| `README_fr.md` | Français |
| `README_tr.md` | Türkçe |
| `README_zh-TW.md` | 繁體中文 |
| `README_zh-CN.md` | 简体中文 |
| `README_ru.md` | Русский |

---

## Batch Prompt Testing Script

To iterate across multiple prompts from the collection:

```python
import os
import json
import time
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

PROMPTS = {
    "neon_portrait": "35mm film photography, convenience store fluorescent + neon lighting, cinematic editorial, film grain, high contrast",
    "vintage_poster": "Vintage 1950s Amalfi Coast travel poster, hand-lettered typography, warm Mediterranean palette, aged paper texture",
    "character_sheet": "Anime character reference sheet, front/side/back views, expression grid below, color palette swatches, white background, clean lineart",
    "ui_mockup": "Realistic iPhone screenshot, social media app, modern clean UI, dark mode, photographed on marble desk",
}

output_dir = Path("generated")
output_dir.mkdir(exist_ok=True)

results = {}

for name, prompt in PROMPTS.items():
    print(f"Generating: {name}")
    try:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        url = response.data[0].url
        results[name] = {"prompt": prompt, "url": url, "status": "ok"}
        print(f"  ✓ {url}")
    except Exception as e:
        results[name] = {"prompt": prompt, "error": str(e), "status": "error"}
        print(f"  ✗ {e}")
    time.sleep(1)  # rate limit buffer

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir}/results.json")
```

---

## Contributing a New Prompt Case

To add a case to the repository:

1. Fork the repo and create a branch
2. Add your output image to `./images/[category]_case[N]/output.jpg`
3. Add a case entry to `README.md` following this template:

```markdown
### Case N: [Case Title](https://x.com/your_post_url) (by [@YourHandle](https://x.com/YourHandle))

| Output |
| :----: |
| <a href="https://evolink.ai/models"><img src="./images/[category]_caseN/output.jpg" width="300" alt="Output image"></a> |

**Prompt:**

```
Your full prompt text here
```
```

4. Open a pull request with the category tag in the title

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Prompt truncated mid-generation | Prompt too long (>4000 chars) | Split into style + detail prompts; use iterative refinement |
| Wrong aspect ratio output | Size not specified | Add `size` parameter: `"1024x1792"` (portrait), `"1792x1024"` (landscape) |
| Plastic/over-smooth skin | Default model smoothing | Add to prompt: `no plastic skin, visible micro pores, natural skin texture` |
| Wrong art style | Style reference too vague | Name specific references: `Persona 5 official art`, `Kodak Portra film` |
| API 401 error | Missing/wrong API key | Set `EVOLINK_API_KEY` env var correctly |
| Rate limit hit | Too many requests | Add `time.sleep(1)` between calls; check Evolink plan limits |

---

## Environment Variables

```bash
# Required for Evolink API access
export EVOLINK_API_KEY="your-evolink-api-key-here"

# Optional: OpenAI SDK base URL override
export OPENAI_BASE_URL="https://api.evolink.ai/v1"
```

---

## License

Content in this repository is licensed under [CC BY 4.0](LICENSE). Individual prompts and images are attributed to their original creators.
```
