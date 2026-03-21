```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - run netryx search
  - geolocation from photo without landmarks
  - open source geoguessr pipeline
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas into a searchable index, then matches query images against that index using a three-stage CV pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and spatial refinement. Sub-50m accuracy. No internet image database — it searches the physical world.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # Required
pip install kornia                                        # Optional: Ultra Mode (LoFTR)
```

### Environment Variables

```bash
# Optional — only needed for AI Coarse blind geolocation mode
export GEMINI_API_KEY="your_key_here"
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

GPU backends: CUDA (NVIDIA), MPS (Apple Silicon M1+), or CPU (slow).

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### Step 1 — Create an Index (crawl an area)

In the GUI:
1. Select **Create** mode
2. Enter center lat/lon of the target area
3. Set search radius (start with 0.5–1 km for testing)
4. Set grid resolution (default: 300 — do not change)
5. Click **Create Index**

Index is saved incrementally — safe to interrupt and resume.

**Indexing time estimates:**

| Radius  | ~Panoramas | Time (M2 Max) | Index Size |
|---------|-----------|---------------|------------|
| 0.5 km  | ~500      | 30 min        | ~60 MB     |
| 1 km    | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km    | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km   | ~100,000  | 24–48 hours   | ~7 GB      |

**File output:**
```
cosplace_parts/     ← raw embedding chunks (.npz per batch)
index/
  cosplace_descriptors.npy   ← all 512-dim descriptors
  metadata.npz               ← coordinates, headings, panoid IDs
```

### Step 2 — Search

In the GUI:
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual** — provide center lat/lon + radius if region is known
   - **AI Coarse** — Gemini analyzes the image for region clues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index cosine similarity search (radius-filtered via haversine)
    │
    └── Top 500–1000 candidates  [< 1 second]
    │
    ▼
Download panorama tiles (8 tiles, stitched) → rectilinear crops
    │
    ├── Multi-FOV crops: 70°, 90°, 110° (handles zoom mismatch)
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├── LightGlue deep feature matching
    └── RANSAC geometric verification  [2–5 min for 300–500 candidates]
    │
    ▼
Heading Refinement: ±45° at 15° steps, 3 FOVs, top 15 candidates
    │
    ├── Spatial consensus clustering (50 m cells)
    └── Confidence scoring (clustering + uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode

Enable the **Ultra Mode** checkbox in the GUI for difficult images (night, blur, low texture).

Adds three extra steps:
- **LoFTR** — detector-free dense matching (handles blur/low-contrast)
- **Descriptor hopping** — re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion** — searches all panoramas within 100 m of best match

```bash
pip install kornia  # required for LoFTR
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main application — GUI, indexing, search
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy
    └── metadata.npz
```

---

## Code Examples

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape → (512,)
print("Descriptor shape:", descriptor.shape)
```

### Search the index programmatically

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]
panoids = meta["panoids"]
headings = meta["headings"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_index(query_descriptor, center_lat, center_lon, radius_km=1.0, top_k=500):
    """
    Returns top-k candidates within radius_km of center, sorted by cosine similarity.
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lat, lon)
        for lat, lon in zip(lats, lons)
    ])
    mask = distances <= radius_km
    filtered_idx = np.where(mask)[0]

    if len(filtered_idx) == 0:
        return []

    filtered_desc = descriptors[filtered_idx]                        # (M, 512)
    sims = cosine_similarity(query_descriptor.reshape(1, -1),
                             filtered_desc)[0]                        # (M,)

    ranked = np.argsort(sims)[::-1][:top_k]
    results = []
    for r in ranked:
        orig_idx = filtered_idx[r]
        results.append({
            "panoid": panoids[orig_idx],
            "lat": float(lats[orig_idx]),
            "lon": float(lons[orig_idx]),
            "heading": float(headings[orig_idx]),
            "similarity": float(sims[r]),
        })
    return results

# Usage
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)
img = Image.open("query.jpg").convert("RGB")
desc = extract_descriptor(model, img, device=device)

# Also search flipped version
desc_flip = extract_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT), device=device)
combined = (desc + desc_flip) / 2   # simple average fusion

candidates = search_index(combined, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
print(f"Found {len(candidates)} candidates")
print("Top match:", candidates[0])
```

### Build/rebuild the searchable index from parts

```python
# Standalone rebuild — run after indexing is complete or interrupted
python build_index.py
```

Or trigger it programmatically:

```python
import subprocess
subprocess.run(["python", "build_index.py"], check=True)
```

### Multi-area index — no city selection needed

```python
# Index Paris
search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Index London (same index file, different radius filter)
search_index(desc, center_lat=51.5074, center_lon=-0.1278, radius_km=5.0)

# All embeddings coexist — radius filter isolates regions automatically
```

---

## Models Reference

| Model | Role | Backend |
|-------|------|---------|
| CosPlace | Global 512-dim visual fingerprint | CUDA / MPS / CPU |
| ALIKED | Local keypoints (1024 kp) | CUDA only |
| DISK | Local keypoints (768 kp) | MPS / CPU |
| LightGlue | Deep feature matching | CUDA / MPS / CPU |
| LoFTR | Dense detector-free matching (Ultra) | CUDA / MPS |

---

## Common Patterns

### Pattern: Batch index multiple cities

```python
cities = [
    ("Paris",   48.8566,  2.3522,  5.0),
    ("London",  51.5074, -0.1278,  5.0),
    ("Berlin",  52.5200, 13.4050,  5.0),
]
# Run test_super.py in Create mode for each entry.
# All output lands in the same cosplace_parts/ and index/ — 
# radius filtering at search time keeps them separate.
```

### Pattern: Check GPU backend in use

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    extractor = "ALIKED"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    extractor = "DISK"
else:
    device = torch.device("cpu")
    extractor = "DISK"

print(f"Using device: {device}, feature extractor: {extractor}")
```

### Pattern: Confidence threshold for automated pipelines

```python
TOP_K = 10

def is_high_confidence(candidates, min_inliers=50, cluster_radius_km=0.05):
    """
    Returns True if top candidates cluster tightly — indicates reliable match.
    """
    if not candidates:
        return False
    top = candidates[:TOP_K]
    lats = [c["lat"] for c in top]
    lons = [c["lon"] for c in top]
    spread_lat = max(lats) - min(lats)
    spread_lon = max(lons) - min(lons)
    # ~0.001 degree ≈ 111m — tight cluster = confident result
    return spread_lat < 0.001 and spread_lon < 0.001
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GUI appears blank on macOS | Bundled tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ModuleNotFoundError: kornia` | Ultra Mode dependency missing | `pip install kornia` |
| CUDA OOM during matching | Too many candidates | Reduce top_k in retrieval (500 → 200) |
| Index search returns 0 results | Radius too small or area not indexed | Increase radius or re-index with larger coverage |
| Low inlier count (<20) on correct match | FOV mismatch or query too degraded | Enable Ultra Mode; try multiple FOVs |
| Indexing stalled / no progress | API rate limit from Street View | Wait and re-run — index resumes from checkpoint |
| `cosplace_descriptors.npy` missing | Parts built but index not compiled | Run `python build_index.py` |
| Wrong region matched | Visually similar but geographically distant | Tighten radius filter; use Manual mode instead of AI Coarse |

---

## Key Files to Modify

- **`test_super.py`** — Full pipeline logic + GUI. Start here for any customization.
- **`cosplace_utils.py`** — Swap CosPlace model weights or change descriptor dimension here.
- **`build_index.py`** — Tune batch size for large-scale indexing on high-RAM machines.

---

## Quick Reference

```bash
# Launch GUI (indexing + search)
python test_super.py

# Rebuild searchable index from raw parts
python build_index.py

# Set Gemini key for AI Coarse mode
export GEMINI_API_KEY="your_key_here"

# Check index size
ls -lh index/cosplace_descriptors.npy index/metadata.npz

# Count indexed panoramas
python -c "import numpy as np; m=np.load('index/metadata.npz'); print(len(m['lats']), 'panoramas indexed')"
```
```
