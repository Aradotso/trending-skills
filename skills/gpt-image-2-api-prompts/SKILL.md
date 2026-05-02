```markdown
---
name: gpt-image-2-api-prompts
description: Expert skill for using GPT-Image-2 API via Evolink, including prompt engineering patterns, image generation workflows, and curated prompt templates for e-commerce, portraits, posters, UI mockups, and more.
triggers:
  - generate an image with GPT Image 2
  - use the GPT Image 2 API
  - create a prompt for image generation
  - how do I call the Evolink image API
  - text to image with GPT Image 2
  - image to image generation prompt
  - gpt-image-2 prompt examples
  - set up GPT Image 2 in my project
---

# GPT-Image-2 API and Prompts

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

**awesome-gpt-image-2-API-and-Prompts** is a curated collection of 330+ prompt patterns, reference cases, and reusable workflows for the GPT-Image-2 model served via the [Evolink API](https://evolink.ai). It covers:

- **Text-to-image** generation with detailed, production-ready prompts
- **Image-to-image** transformation workflows
- Prompt categories: e-commerce product shots, ad creatives, portraits, posters, character design, UI/social mockups, and community comparisons
- A callable skill (`gpt-image-2-gen-skill`) for one-line integration
- A `gpt_image_2_prompt.json` file tracking prompt-only updates

---

## Installation

### Option 1 — Callable Skill (Node.js, quickest)

```bash
npx evolink-gpt-image -y
```

### Option 2 — Direct API (no install required)

Set your API key as an environment variable:

```bash
export EVOLINK_API_KEY="your_api_key_here"
```

Get a key at: https://evolink.ai/dashboard

### Option 3 — Python SDK pattern (manual)

```bash
pip install requests python-dotenv
```

Create a `.env` file:

```
EVOLINK_API_KEY=your_api_key_here
```

---

## API Reference

**Base URL:** `https://api.evolink.ai/v1`

**Endpoint:** `POST /images/generations`

### Request Schema

| Field    | Type   | Required | Description                                      |
|----------|--------|----------|--------------------------------------------------|
| model    | string | Yes      | `"gpt-image-2"`                                  |
| prompt   | string | Yes      | Natural language description of the image        |
| size     | string | No       | `"1024x1024"`, `"1792x1024"`, `"1024x1792"`      |
| quality  | string | No       | `"standard"` or `"hd"`                           |
| n        | int    | No       | Number of images (1–4)                           |
| response_format | string | No | `"url"` or `"b64_json"`                    |

### Response Schema

```json
{
  "created": 1714000000,
  "data": [
    {
      "url": "https://...",
      "revised_prompt": "..."
    }
  ]
}
```

---

## Quick Start — cURL

```bash
curl --request POST \
  --url https://api.evolink.ai/v1/images/generations \
  --header "Authorization: Bearer $EVOLINK_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-image-2",
    "prompt": "A beautiful colorful sunset over the ocean",
    "size": "1024x1024",
    "quality": "hd",
    "n": 1
  }'
```

---

## Python Examples

### Basic Text-to-Image

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]
BASE_URL = "https://api.evolink.ai/v1"

def generate_image(prompt: str, size: str = "1024x1024", quality: str = "hd") -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": 1,
    }
    response = requests.post(f"{BASE_URL}/images/generations", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

result = generate_image("A hyper-realistic miniature diorama of a luxury skincare product")
print(result["data"][0]["url"])
```

### Save Image to Disk

```python
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]

def generate_and_save(prompt: str, output_path: str = "output.png") -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "hd",
        "n": 1,
        "response_format": "url",
    }
    response = requests.post(
        "https://api.evolink.ai/v1/images/generations",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    image_url = response.json()["data"][0]["url"]

    image_data = requests.get(image_url).content
    Path(output_path).write_bytes(image_data)
    print(f"Saved to {output_path}")
    return output_path

generate_and_save(
    "Cinematic hero shot of a gourmet cheeseburger, warm side light, shallow depth of field",
    "burger_hero.png",
)
```

### Batch Generation (Multiple Prompts)

```python
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]

PROMPTS = [
    "Luxury watch advertisement, dark background, dramatic lighting",
    "Miniature diorama skincare product, tilt-shift aesthetic, studio lighting",
    "Flat lay UI mockup for a mobile app, clean white background",
]

def batch_generate(prompts: list[str], delay_seconds: float = 1.0) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    results = []
    for i, prompt in enumerate(prompts):
        payload = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "standard",
        }
        resp = requests.post(
            "https://api.evolink.ai/v1/images/generations",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        results.append({"prompt": prompt, "url": resp.json()["data"][0]["url"]})
        print(f"[{i+1}/{len(prompts)}] Done: {prompt[:60]}...")
        if i < len(prompts) - 1:
            time.sleep(delay_seconds)
    return results

results = batch_generate(PROMPTS)
for r in results:
    print(r["url"])
```

### Base64 Response (for in-memory processing)

```python
import os
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]

def generate_image_base64(prompt: str) -> Image.Image:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    resp = requests.post(
        "https://api.evolink.ai/v1/images/generations",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    return Image.open(BytesIO(base64.b64decode(b64)))

img = generate_image_base64("A serene Japanese garden at sunrise, ultra-detailed")
img.show()
img.save("garden.png")
```

---

## Prompt Engineering Patterns

### Pattern 1: Product Photography (E-commerce)

Use tilt-shift miniature aesthetics, warm color palettes, and explicit studio settings:

```python
prompt = """
A hyper-realistic miniature diorama product advertisement featuring an oversized 
luxury skincare pump bottle labeled "LUXEVEIL Skin Science" in cream/beige with 
a polished gold pump top, placed on a circular platform. Tiny figurine construction 
workers in yellow coveralls swarm the bottle on scaffolding. Warm beige, cream, 
gold palette. Studio photography, soft diffused lighting, clean beige background. 
Tilt-shift miniature aesthetic, ultra-detailed, 8K resolution, photorealistic CGI.
"""
```

### Pattern 2: Cinematic Food Photography

```python
prompt = """
Cinematic hero image of a gourmet cheeseburger on a dark stone surface with 
glossy brioche bun, melted cheese, crisp lettuce, tomato, grilled patty, sauce, 
realistic texture, appetizing steam, warm side light, shallow depth of field, 
premium food commercial style. No text, no logos, no watermark.
"""
```

### Pattern 3: 9-Panel Storyboard Layout

```python
prompt = """
Create a 9-cell hybrid keyframe-to-transition storyboard sheet for a 15-second 
gourmet burger ad. Use large scene cells and smaller transition cells, motion 
arrows, ghosted ingredient positions, steam, sauce trails, camera push-in icons. 
Style: premium food commercial, warm lighting, rich texture, cinematic. 
Minimal labels only. No logos, no watermark.
"""
```

### Pattern 4: UI / Social Media Mockup

```python
prompt = """
Flat lay mockup of a mobile app home screen on an iPhone 15 Pro, clean white 
marble surface, minimal design, pastel color scheme, soft shadows, studio lighting, 
top-down perspective, ultra-sharp, commercial product photography style.
"""
```

### Pattern 5: Portrait / Character Design

```python
prompt = """
Hyper-realistic portrait of a young woman with warm studio lighting, shallow 
depth of field, skin detail, natural makeup, soft bokeh background in warm tones. 
High-end fashion editorial style, 85mm lens simulation, photorealistic, 8K.
"""
```

---

## Loading Prompts from the JSON Catalog

The repository ships `gpt_image_2_prompt.json` with all curated prompts. Use it to pick prompts programmatically:

```python
import json
import random
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]

with open("gpt_image_2_prompt.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Pick a random prompt from the catalog
entry = random.choice(catalog)
prompt = entry["prompt"]
category = entry.get("category", "unknown")
print(f"Category: {category}")
print(f"Prompt: {prompt[:120]}...")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
payload = {"model": "gpt-image-2", "prompt": prompt, "size": "1024x1024"}
resp = requests.post(
    "https://api.evolink.ai/v1/images/generations",
    json=payload,
    headers=headers,
)
resp.raise_for_status()
print("Image URL:", resp.json()["data"][0]["url"])
```

---

## Callable Skill Integration (Node.js / npx)

After installing with `npx evolink-gpt-image -y`, the skill is callable from agent pipelines:

```bash
# Generate a single image
evolink-gpt-image generate --prompt "A cinematic sunset over Tokyo" --size 1024x1024

# List available prompt templates
evolink-gpt-image templates --category poster

# Use a template by case ID
evolink-gpt-image generate --case 151
```

Set your key before calling:

```bash
export EVOLINK_API_KEY="your_api_key_here"
```

---

## Cinematic Workflow: GPT-Image-2 × Seedance 2.0

For video generation from still images:

```bash
# Clone the workflow repo
git clone https://github.com/EvoLinkAI/GPT-Image-2-Seedance2-Workflow
cd GPT-Image-2-Seedance2-Workflow
```

Workflow: `text prompt → GPT-Image-2 still → Seedance 2.0 video animation`

See the workflow README for full pipeline configuration.

---

## Common Patterns and Tips

### Always Specify Output Style

```python
# Weak prompt
prompt = "a bottle of perfume"

# Strong prompt — specifies lighting, background, style, resolution
prompt = """
Luxury perfume bottle on a black marble surface, dramatic rim lighting, 
deep shadows, reflections on surface, cinematic product photography, 
8K ultra-detailed, no text, no watermark.
"""
```

### Use Explicit Negative Instructions

GPT-Image-2 respects explicit exclusions inline:

```python
prompt = "... No text, no logos, no watermarks, no UI overlays, clean background."
```

### Aspect Ratio Selection

```python
# Portrait (tall) — ideal for mobile, fashion, portrait
size = "1024x1792"

# Landscape — ideal for banners, cinematic shots
size = "1792x1024"

# Square — ideal for product shots, social media
size = "1024x1024"
```

### Quality vs Speed Trade-off

```python
# Faster, lower cost
payload["quality"] = "standard"

# Slower, more detailed — recommended for final assets
payload["quality"] = "hd"
```

---

## Error Handling

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["EVOLINK_API_KEY"]

def safe_generate(prompt: str) -> str | None:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": "gpt-image-2", "prompt": prompt}
        resp = requests.post(
            "https://api.evolink.ai/v1/images/generations",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["url"]
    except requests.exceptions.Timeout:
        print("Request timed out — the model may be under heavy load, retry.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 401:
            print("Invalid API key — check EVOLINK_API_KEY env var.")
        elif status == 429:
            print("Rate limit hit — add delay between requests.")
        elif status == 400:
            print(f"Bad request: {e.response.json()}")
        else:
            print(f"HTTP error {status}: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
```

---

## Project Structure

```
awesome-gpt-image-2-API-and-Prompts/
├── README.md                    # Main English docs with all cases
├── README_zh-CN.md              # Simplified Chinese
├── README_ja.md                 # Japanese
├── README_es.md                 # Spanish
├── README_fr.md                 # French
├── README_de.md                 # German
├── gpt_image_2_prompt.json      # Machine-readable prompt catalog
├── images/                      # Output images by case ID
│   └── poster_case151/
│       └── output.jpg
├── cases/
│   ├── ecommerce.md             # E-commerce prompt cases
│   ├── ad-creative.md           # Ad creative cases
│   ├── portrait.md              # Portrait & photography cases
│   ├── poster.md                # Poster & illustration cases
│   ├── character.md             # Character design cases
│   ├── ui.md                    # UI & social media mockup cases
│   └── comparison.md            # Comparison & community examples
```

---

## Resources

- [Evolink Dashboard (API Keys)](https://evolink.ai/dashboard)
- [GPT-Image-2 API Docs](https://docs.evolink.ai/en/api-manual/image-series/gpt-image-2/gpt-image-2-image-generation)
- [Try Prompts Online](https://evolink.ai/gpt-image-2-prompts)
- [Callable Skill Repo](https://github.com/EvoLinkAI/gpt-image-2-gen-skill)
- [Cinematic Workflow (GPT-Image-2 × Seedance 2.0)](https://github.com/EvoLinkAI/GPT-Image-2-Seedance2-Workflow)
```
