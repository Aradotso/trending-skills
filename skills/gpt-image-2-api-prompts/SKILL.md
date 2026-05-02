```markdown
---
name: gpt-image-2-api-prompts
description: Expertise in using the GPT-Image-2 API via Evolink, prompt engineering patterns, and curated visual workflows for image generation
triggers:
  - generate an image with GPT Image 2
  - use gpt-image-2 api
  - create image generation prompts
  - text to image with openai gpt image
  - gpt image 2 api integration
  - image generation prompt patterns
  - evolink image api
  - awesome gpt image prompts
---

# GPT-Image-2 API and Prompts

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

**awesome-gpt-image-2-API-and-Prompts** is a curated collection of 330+ prompt patterns, reference cases, and reusable visual workflows for the GPT-Image-2 model served via the [Evolink API](https://evolink.ai). It covers:

- Text-to-image and image-to-image generation
- Domain-specific prompt patterns: e-commerce, ad creative, portrait, poster, character design, UI mockups
- A callable skill (`gpt-image-2-gen-skill`) for agent integration
- A cinematic workflow combining GPT-Image-2 with Seedance 2.0

---

## Installation & Setup

### 1. Install the Callable Skill (Node.js)

```bash
npx evolink-gpt-image -y
```

### 2. Get Your API Key

1. Go to [https://evolink.ai/dashboard](https://evolink.ai/dashboard)
2. Create an account and generate an API key
3. Store it as an environment variable:

```bash
export EVOLINK_API_KEY="your_api_key_here"
```

### 3. Python Dependencies

```bash
pip install openai requests python-dotenv
```

---

## API Reference

**Base URL:** `https://api.evolink.ai/v1`

**Image Generation Endpoint:** `POST /images/generations`

**Headers:**
- `Authorization: Bearer YOUR_API_KEY`
- `Content-Type: application/json`

**Key Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Always `"gpt-image-2"` |
| `prompt` | string | Text description of desired image |
| `n` | integer | Number of images (default: 1) |
| `size` | string | `"1024x1024"`, `"1792x1024"`, `"1024x1792"` |
| `quality` | string | `"standard"` or `"hd"` |
| `response_format` | string | `"url"` or `"b64_json"` |

---

## Code Examples

### Basic Text-to-Image (Python, requests)

```python
import os
import requests

api_key = os.environ["EVOLINK_API_KEY"]

response = requests.post(
    "https://api.evolink.ai/v1/images/generations",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-image-2",
        "prompt": "A beautiful colorful sunset over the ocean",
        "n": 1,
        "size": "1024x1024",
    },
)

data = response.json()
image_url = data["data"][0]["url"]
print(image_url)
```

### Using the OpenAI Python Client (Compatible)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

response = client.images.generate(
    model="gpt-image-2",
    prompt="Hyper-realistic miniature diorama product advertisement, luxury skincare bottle, tiny figurine workers on scaffolding, warm beige palette, studio photography, 8K",
    n=1,
    size="1024x1024",
    quality="hd",
)

print(response.data[0].url)
```

### Download and Save Image

```python
import os
import requests
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

def generate_and_save(prompt: str, output_path: str, size: str = "1024x1024") -> str:
    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        n=1,
        size=size,
        quality="hd",
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    Path(output_path).write_bytes(img_data)
    print(f"Saved to {output_path}")
    return image_url

generate_and_save(
    prompt="Cinematic hero image of a gourmet cheeseburger on dark stone surface, glossy brioche bun, melted cheese, warm side light, shallow depth of field, premium food commercial style",
    output_path="burger_hero.jpg",
    size="1792x1024",
)
```

### Batch Generation from Prompt List

```python
import os
import json
import time
import requests
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

def batch_generate(prompts: list[dict], delay: float = 1.0) -> list[dict]:
    """
    prompts: list of {"id": str, "prompt": str, "size": str}
    """
    results = []
    for item in prompts:
        try:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=item["prompt"],
                n=1,
                size=item.get("size", "1024x1024"),
                quality=item.get("quality", "standard"),
            )
            results.append({
                "id": item["id"],
                "url": response.data[0].url,
                "status": "success",
            })
        except Exception as e:
            results.append({"id": item["id"], "status": "error", "error": str(e)})
        time.sleep(delay)
    return results

# Load prompts from the repo's JSON file
with open("gpt_image_2_prompt.json") as f:
    prompt_data = json.load(f)

results = batch_generate(prompt_data[:5])
for r in results:
    print(r)
```

### cURL Quick Test

```bash
curl --request POST \
  --url https://api.evolink.ai/v1/images/generations \
  --header "Authorization: Bearer ${EVOLINK_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-image-2",
    "prompt": "A beautiful colorful sunset over the ocean",
    "n": 1,
    "size": "1024x1024"
  }'
```

---

## Prompt Patterns by Category

### E-commerce Product Shots

```python
ECOMMERCE_TEMPLATE = """
A hyper-realistic {style} product advertisement featuring {product_description}.
{scene_elements}.
Color palette: {palette}.
Studio photography style with soft diffused lighting, clean {background} background.
Tilt-shift miniature aesthetic, ultra-detailed, commercial product photography, 8K resolution, photorealistic CGI render.
"""

prompt = ECOMMERCE_TEMPLATE.format(
    style="miniature diorama",
    product_description="oversized luxury skincare pump bottle labeled 'LUXEVEIL Skin Science' in cream/beige with polished gold pump top",
    scene_elements="Tiny figurine construction workers in yellow coveralls and white hard hats, scaffolding, tower crane, industrial tanks",
    palette="warm beige, cream, gold, and mustard yellow",
    background="beige",
)
```

### Portrait & Photography

```python
PORTRAIT_TEMPLATE = """
{subject_description}, {lighting_style} lighting, {lens_style} lens,
{mood} mood, {color_grade} color grade, {camera_style} photography,
ultra-detailed, photorealistic, professional headshot composition.
"""

prompt = PORTRAIT_TEMPLATE.format(
    subject_description="close-up portrait of a woman in her 30s with natural makeup",
    lighting_style="soft golden hour",
    lens_style="85mm f/1.4 bokeh",
    mood="serene and confident",
    color_grade="warm film",
    camera_style="editorial",
)
```

### Poster & Illustration

```python
POSTER_TEMPLATE = """
{art_style} poster design for {subject},
{color_scheme} color scheme, {typography_notes},
{visual_elements}, {composition_style} composition,
high detail, print-ready quality, {resolution}.
"""

prompt = POSTER_TEMPLATE.format(
    art_style="vintage Art Deco",
    subject="an international jazz festival",
    color_scheme="deep navy, gold, and ivory",
    typography_notes="bold serif headline with geometric decorative borders",
    visual_elements="silhouette of saxophone player, geometric patterns, starbursts",
    composition_style="symmetrical centered",
    resolution="8K",
)
```

### UI & Social Media Mockups

```python
UI_MOCKUP_TEMPLATE = """
Clean {platform} UI mockup for {app_type} app,
{design_system} design language, {color_palette} color palette,
showing {screen_content}, {device_frame} device frame,
professional app store screenshot style, high fidelity, pixel-perfect.
"""

prompt = UI_MOCKUP_TEMPLATE.format(
    platform="iOS",
    app_type="fitness tracking",
    design_system="Material You",
    color_palette="vibrant coral and white",
    screen_content="dashboard with weekly activity rings, step count, and heart rate chart",
    device_frame="iPhone 15 Pro",
)
```

### 9-Panel Storyboard (Advanced)

```python
STORYBOARD_TEMPLATE = """
Create a 9-cell hybrid keyframe-to-transition storyboard sheet for a {duration}-second {product_type} ad.
Panels should progress: {story_arc}.
Use large scene cells and smaller transition cells, motion arrows, camera push-in icons.
Style: {visual_style}, {lighting}, {aesthetic}.
Include panel labels with timestamps. No logos, no watermarks.
"""

prompt = STORYBOARD_TEMPLATE.format(
    duration="15",
    product_type="gourmet burger",
    story_arc="empty surface → ingredient assembly → final macro hero shot",
    visual_style="premium food commercial",
    lighting="warm cinematic lighting",
    aesthetic="rich texture, appetizing, minimal labels only",
)
```

---

## Working with `gpt_image_2_prompt.json`

The repo ships a JSON file tracking all curated prompts. Use it programmatically:

```python
import json

with open("gpt_image_2_prompt.json") as f:
    prompts = json.load(f)

# Filter by category
portrait_prompts = [p for p in prompts if p.get("category") == "portrait"]
poster_prompts = [p for p in prompts if p.get("category") == "poster"]

# Get a specific case
case_151 = next((p for p in prompts if p.get("id") == "case151"), None)
if case_151:
    print(case_151["prompt"])
```

---

## Environment Configuration

```bash
# .env file
EVOLINK_API_KEY=your_api_key_here
EVOLINK_BASE_URL=https://api.evolink.ai/v1
IMAGE_OUTPUT_DIR=./generated_images
DEFAULT_IMAGE_SIZE=1024x1024
DEFAULT_IMAGE_QUALITY=hd
```

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

EVOLINK_API_KEY = os.environ["EVOLINK_API_KEY"]
EVOLINK_BASE_URL = os.getenv("EVOLINK_BASE_URL", "https://api.evolink.ai/v1")
IMAGE_OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "./generated_images")
DEFAULT_SIZE = os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024")
DEFAULT_QUALITY = os.getenv("DEFAULT_IMAGE_QUALITY", "hd")
```

---

## GPT-Image-2 × Seedance 2.0 Cinematic Workflow

For video generation from GPT-Image-2 outputs:

```bash
# Install the cinematic workflow
git clone https://github.com/EvoLinkAI/GPT-Image-2-Seedance2-Workflow
cd GPT-Image-2-Seedance2-Workflow
```

```python
# Pattern: generate keyframe image, then animate with Seedance 2.0
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EVOLINK_API_KEY"],
    base_url="https://api.evolink.ai/v1",
)

# Step 1: Generate cinematic keyframe
keyframe = client.images.generate(
    model="gpt-image-2",
    prompt="Cinematic establishing shot, golden hour, lone figure on misty mountain ridge, dramatic sky, photorealistic, 8K",
    size="1792x1024",
    quality="hd",
)
keyframe_url = keyframe.data[0].url

# Step 2: Pass keyframe_url to Seedance 2.0 for animation
# (see GPT-Image-2-Seedance2-Workflow repo for full integration)
print(f"Keyframe ready for animation: {keyframe_url}")
```

---

## Troubleshooting

### Authentication Errors (401)

```python
# Verify your key is set
import os
key = os.environ.get("EVOLINK_API_KEY")
if not key:
    raise ValueError("EVOLINK_API_KEY environment variable not set")
if len(key) < 20:
    raise ValueError("API key looks malformed")
```

### Rate Limiting (429)

```python
import time

def generate_with_retry(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.images.generate(
                model="gpt-image-2",
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential backoff
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
```

### Prompt Too Long

GPT-Image-2 works best with prompts under ~1000 characters. For complex storyboard prompts, break them into separate calls:

```python
def chunk_prompt(full_prompt: str, max_chars: int = 900) -> list[str]:
    """Split a long prompt into chunks for sequential generation."""
    if len(full_prompt) <= max_chars:
        return [full_prompt]
    sentences = full_prompt.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks
```

### Validate Response

```python
def safe_get_url(response) -> str | None:
    try:
        return response.data[0].url
    except (AttributeError, IndexError, KeyError):
        return None
```

---

## Key Links

- [GPT-Image-2 Prompts Gallery](https://evolink.ai/gpt-image-2-prompts)
- [API Documentation](https://docs.evolink.ai/en/api-manual/image-series/gpt-image-2/gpt-image-2-image-generation)
- [Evolink Dashboard (API Keys)](https://evolink.ai/dashboard)
- [gpt-image-2-gen-skill (callable skill)](https://github.com/EvoLinkAI/gpt-image-2-gen-skill)
- [GPT-Image-2 × Seedance 2.0 Workflow](https://github.com/EvoLinkAI/GPT-Image-2-Seedance2-Workflow)
- [Prompt JSON file](https://github.com/EvoLinkAI/awesome-gpt-image-2-API-and-Prompts/blob/main/gpt_image_2_prompt.json)
```
