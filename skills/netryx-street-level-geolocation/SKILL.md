---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a locally-hosted open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - identify location from photo
  - osint geolocation tool
  - reverse geolocate image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls Street View panoramas into a local index, then matches query images using a three-stage CV pipeline: **CosPlace** (global retrieval) → **ALIKED/DISK + LightGlue** (geometric verification) → **heading refinement + spatial consensus**.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

**macOS tkinter fix** (blank GUI on recent macOS):
```bash
brew install python-tk@3.11   # match your Python version
```

**Optional — Gemini API for AI Coarse mode:**
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 50 GB+ |
| Python | 3.9+ | 3.10+ |

GPU backend auto-detected: **CUDA** (NVIDIA) → **MPS** (Apple Silicon) → **CPU** fallback.

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface for all operations — indexing and searching are both driven from it.

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-perf index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # lat/lon, heading, panorama IDs
```

---

## Core Workflow

### Step 1 — Create an Index

In the GUI: select **Create** mode, enter center lat/lon, set radius and grid resolution (default 300), click **Create Index**.

**Time and size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — safe to interrupt and restart.

Multiple cities can share one unified index. Search radius + center coordinates filter results automatically.

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual** — enter approximate center coordinates + radius
   - **AI Coarse** — Gemini analyzes visual clues to estimate the region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## The Three-Stage Pipeline

### Stage 1: Global Retrieval (CosPlace)
- Extracts a 512-dim descriptor from the query image (and its horizontal flip)
- Cosine similarity search against the full index
- Haversine radius filter restricts to search area
- Returns top 500–1000 candidates
- Runs in **< 1 second** (single matrix multiply)

### Stage 2: Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads Street View panoramas for each candidate (8 tiles, stitched)
- Crops at the indexed heading across **3 FOVs** (70°, 90°, 110°)
- Extracts keypoints: **ALIKED** on CUDA, **DISK** on MPS/CPU
- **LightGlue** matches query keypoints to candidate keypoints
- **RANSAC** filters to geometrically consistent inliers
- Best candidate = highest verified inlier count
- Runs in **2–5 minutes** for 300–500 candidates

### Stage 3: Refinement
- **Heading refinement**: ±45° at 15° steps, 3 FOVs, for top 15 candidates
- **Spatial consensus**: clusters matches into 50m cells; clusters beat isolated outliers
- **Confidence scoring**: geographic clustering + uniqueness ratio vs. runner-up

---

## Ultra Mode

Enable the **Ultra Mode** checkbox in the GUI for difficult images (night, blur, low texture).

Adds three techniques:
1. **LoFTR** — detector-free dense matching (handles blur/low contrast where keypoints fail)
2. **Descriptor hopping** — if best match has < 50 inliers, re-extracts CosPlace from the *matched high-quality panorama* and re-searches the index
3. **Neighborhood expansion** — searches all panoramas within 100m of the best match

Significantly slower; use only when standard pipeline fails.

---

## Using CosPlace Utilities Directly

```python
# cosplace_utils.py exposes descriptor extraction
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape == (512,)
```

---

## Building a Large Index Programmatically

For large areas, use `build_index.py` directly instead of the GUI:

```python
# build_index.py is a standalone high-performance indexer
# Run from the command line:
# python build_index.py --lat 48.8566 --lon 2.3522 --radius 5000 --grid 300
```

Check `build_index.py` for accepted arguments — it writes incrementally to `cosplace_parts/` and can be resumed after interruption.

---

## Searching the Index Programmatically

The search pipeline lives in `test_super.py`. Key internal functions you can call if scripting:

```python
import numpy as np

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # shape (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # shape (N,)
lons = meta["lons"]      # shape (N,)
headings = meta["headings"]
panoids = meta["panoids"]

# Cosine similarity search (index must be L2-normalized)
from numpy.linalg import norm

def cosine_search(query_descriptor, descriptors, top_k=500):
    """Returns indices of top_k most similar descriptors."""
    q = query_descriptor / norm(query_descriptor)
    d = descriptors / norm(descriptors, axis=1, keepdims=True)
    scores = d @ q
    return np.argsort(-scores)[:top_k]

# Haversine radius filter
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def radius_filter(lats, lons, center_lat, center_lon, radius_km):
    """Returns boolean mask of entries within radius."""
    return np.array([
        haversine_km(center_lat, center_lon, lat, lon) <= radius_km
        for lat, lon in zip(lats, lons)
    ])
```

---

## Index Data Layout

```python
# After indexing, the compiled index contains:
# index/cosplace_descriptors.npy  — float32, shape (N, 512)
# index/metadata.npz              — arrays of equal length N:
#   lats      float64  — latitude of each panorama viewpoint
#   lons      float64  — longitude of each panorama viewpoint
#   headings  float64  — camera heading (degrees) of the indexed crop
#   panoids   object   — Google Street View panorama ID strings

# Raw chunks before compilation live in:
# cosplace_parts/part_XXXXX.npz  — written incrementally during Create mode
```

---

## Common Patterns

### Pattern: Index a small test area, then search it

```python
# 1. In GUI: Create mode
#    lat=48.8584, lon=2.2945, radius=0.5km, grid=300
#    → indexes ~500 panoramas around Eiffel Tower in ~30 min

# 2. In GUI: Search mode
#    Upload: any street photo from that area
#    Manual: lat=48.8584, lon=2.2945, radius=1.0km
#    → returns GPS coordinates + confidence
```

### Pattern: Multi-city unified index

```python
# Index Paris:
#   Create: lat=48.8566, lon=2.3522, radius=5km → writes to cosplace_parts/

# Index London (same index, no conflict):
#   Create: lat=51.5074, lon=-0.1278, radius=5km → appends to cosplace_parts/

# Search Paris only:
#   Search: lat=48.8566, lon=2.3522, radius=6km  (London results excluded by radius)

# Search London only:
#   Search: lat=51.5074, lon=-0.1278, radius=6km
```

### Pattern: Scripted descriptor extraction for batch processing

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
model = load_cosplace_model(device=device)

image_dir = Path("query_images/")
results = {}

for img_path in image_dir.glob("*.jpg"):
    img = Image.open(img_path).convert("RGB")
    desc = extract_descriptor(model, img, device=device)
    results[img_path.name] = desc
    print(f"{img_path.name}: descriptor shape {desc.shape}")

np.save("batch_descriptors.npy", np.stack(list(results.values())))
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GUI appears blank/white | macOS tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | LightGlue not installed | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ModuleNotFoundError: kornia` | Ultra Mode dependency missing | `pip install kornia` |
| CUDA OOM during matching | Too many candidates | Reduce top-K or use smaller search radius |
| Low inlier count (< 20) on all candidates | Query too degraded or area not indexed | Enable Ultra Mode; verify area is indexed |
| Gemini AI Coarse fails | Missing API key | `export GEMINI_API_KEY=...` |
| Indexing crashes mid-run | Network timeout or API limit | Restart — indexing resumes from last saved chunk |
| Wrong city matched | Radius too large, overlapping indexes | Reduce search radius to tightest reasonable value |
| MPS slow vs. expected | DISK used on MPS (slower than ALIKED on CUDA) | Expected behavior — ALIKED requires CUDA |

---

## Model Reference

| Model | Role | Activated On |
|-------|------|-------------|
| CosPlace (CVPR 2022) | 512-dim global visual fingerprint | Always |
| ALIKED (IEEE TIP 2023) | Local keypoints — 1024 per image | CUDA only |
| DISK (NeurIPS 2020) | Local keypoints — 768 per image | MPS / CPU |
| LightGlue (ICCV 2023) | Deep feature matching | Always |
| LoFTR (CVPR 2021) | Detector-free dense matching | Ultra Mode only |

---

## Key Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | 300 | Do not change — controls panorama density |
| Top-K candidates (Stage 1) | 500–1000 | Larger = slower Stage 2, more recall |
| Heading refinement range | ±45° at 15° steps | Applied to top 15 candidates |
| FOVs tested | 70°, 90°, 110° | Handles zoom mismatch |
| Spatial cluster cell size | 50m | Used for consensus scoring |
| Neighborhood expansion (Ultra) | 100m radius | From best match centroid |
| Ultra Mode LoFTR threshold | < 50 inliers | Triggers descriptor hopping |
