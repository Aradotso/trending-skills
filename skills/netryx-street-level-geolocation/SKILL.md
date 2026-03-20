```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - run netryx geolocation pipeline
  - identify location from street photo
  - open source geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls and indexes street-view panoramas, then matches query images using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and spatial refinement — all running on your local hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (installed from GitHub, not PyPI)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### macOS tkinter fix (if GUI renders blank)
```bash
brew install python-tk@3.11  # match your Python version
```

### Optional: Gemini API for AI Coarse geolocation
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4GB     | 8GB+        |
| RAM       | 8GB     | 16GB+       |
| Storage   | 10GB    | 50GB+       |
| Python    | 3.9+    | 3.10+       |

**GPU backends:**
- NVIDIA → CUDA (uses ALIKED, 1024 keypoints)
- Apple Silicon → MPS (uses DISK, 768 keypoints)
- CPU → DISK (significantly slower)

---

## Launch the GUI

```bash
python test_super.py
```

This is the primary entry point. It provides a full GUI for indexing, searching, and viewing results on a map.

---

## Core Workflow

### Step 1: Create an Index

Index a geographic area before any search can run. The system crawls street-view panoramas in a grid pattern, extracts CosPlace descriptors, and saves them incrementally.

**In GUI:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is resumable — interrupt and restart freely.

**Index output structure:**
```
cosplace_parts/        # Raw embedding chunks (.npz files)
index/
├── cosplace_descriptors.npy   # All 512-dim descriptors
└── metadata.npz               # Coordinates, headings, panoid IDs
```

### Step 2: Search

**In GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Enter known approximate coordinates + radius
   - **AI Coarse**: Let Gemini analyze visual clues to guess the region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. View result coordinates and confidence score on the map

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search (cosine similarity, haversine radius filter)
    │
    └── Top 500–1000 candidates
    │
    ▼
Download Panoramas → Crop at 3 FOVs (70°, 90°, 110°)
    │
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├── LightGlue feature matching
    ├── RANSAC geometric verification
    │
    ▼
Heading Refinement (±45°, 15° steps, top 15 candidates)
    │
    ├── Spatial consensus clustering (50m cells)
    ├── Confidence scoring (uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode

Enable for difficult images: night shots, motion blur, low-texture scenes, or heavily compressed images.

**Adds:**
- **LoFTR**: Detector-free dense matching (handles blur/low-contrast)
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

Enable via the **Ultra Mode** checkbox in the GUI before running the search.

---

## Multi-City Indexing Pattern

All embeddings live in one unified index. Radius filtering at search time handles city separation automatically:

```python
# Index Paris
# center_lat=48.8566, center_lon=2.3522, radius=5

# Index London
# center_lat=51.5074, center_lon=-0.1278, radius=5

# Search Paris only — set center to Paris coords with tight radius
# Search London only — set center to London coords with tight radius
# No city selection needed — coordinates + radius handle routing
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main application (GUI + indexing + search)
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # High-performance standalone index builder (large datasets)
├── requirements.txt       # Python dependencies
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
├── index/
│   ├── cosplace_descriptors.npy
│   └── metadata.npz
└── README.md
```

---

## Using `cosplace_utils.py` Directly

```python
import torch
from cosplace_utils import get_cosplace_model, get_descriptor

# Load model (downloads weights on first run)
model = get_cosplace_model()
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device).eval()

# Extract 512-dim descriptor from an image file
descriptor = get_descriptor(model, "path/to/street_photo.jpg", device)
# descriptor.shape == (512,)
```

---

## Using `build_index.py` for Large-Scale Indexing

For datasets larger than 5km radius, use the standalone index builder for better performance:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 10 \
  --resolution 300
```

This writes directly to `cosplace_parts/` and is optimized for long unattended runs.

---

## Index Search (Programmatic Pattern)

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * asin(sqrt(a))

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz")
lats = meta["lats"]
lons = meta["lons"]
headings = meta["headings"]
panoids = meta["panoids"]

# Radius filter (e.g., 2km around Paris center)
center_lat, center_lon, radius_m = 48.8566, 2.3522, 2000
mask = np.array([
    haversine(center_lat, center_lon, la, lo) <= radius_m
    for la, lo in zip(lats, lons)
])
filtered_descriptors = descriptors[mask]  # (M, 512)

# Cosine similarity search
query_desc = get_descriptor(model, "query.jpg", device)  # (512,)
sims = filtered_descriptors @ query_desc  # cosine sim (normalized vectors)
top_k_idx = np.argsort(sims)[::-1][:500]
```

---

## Common Patterns

### Pattern 1: Batch geolocation of multiple images
```python
import os
from pathlib import Path

image_dir = Path("images_to_geolocate/")
results = {}

for img_path in image_dir.glob("*.jpg"):
    desc = get_descriptor(model, str(img_path), device)
    sims = filtered_descriptors @ desc
    best_idx = np.argmax(sims)
    results[img_path.name] = {
        "lat": float(lats[mask][best_idx]),
        "lon": float(lons[mask][best_idx]),
        "similarity": float(sims[best_idx]),
        "panoid": str(panoids[mask][best_idx])
    }

import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Pattern 2: Check device and feature extractor that will be used
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    extractor_name = "ALIKED"
    max_keypoints = 1024
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    extractor_name = "DISK"
    max_keypoints = 768
else:
    device = torch.device("cpu")
    extractor_name = "DISK"
    max_keypoints = 768

print(f"Using {extractor_name} on {device} with {max_keypoints} keypoints")
```

### Pattern 3: Verify LightGlue is installed correctly
```python
try:
    from lightglue import LightGlue, ALIKED, DISK
    from lightglue.utils import load_image, rbd
    print("LightGlue installed correctly")
except ImportError:
    print("LightGlue missing — run:")
    print("pip install git+https://github.com/cvg/LightGlue.git")
```

### Pattern 4: Verify Ultra Mode dependencies (LoFTR)
```python
try:
    import kornia
    from kornia.feature import LoFTR
    print(f"kornia {kornia.__version__} — Ultra Mode available")
except ImportError:
    print("kornia not installed — Ultra Mode unavailable")
    print("Install: pip install kornia")
```

---

## Troubleshooting

### GUI renders blank on macOS
```bash
brew install python-tk@3.11  # Replace 3.11 with your Python version
```

### `ImportError: No module named 'lightglue'`
LightGlue must be installed from GitHub, not PyPI:
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory
- Reduce candidates: lower the top-K from 500 to 200 in the GUI
- Enable Ultra Mode only when necessary (it loads additional models)
- Ensure no other GPU-heavy processes are running

### Indexing stalls or crashes
- Indexing is incremental — restart and it resumes from the last saved chunk in `cosplace_parts/`
- Check internet connectivity (panorama downloads require broadband)
- Reduce grid resolution if rate-limited by the street-view provider

### Poor match accuracy
- Increase search radius to ensure the correct area is indexed
- Enable Ultra Mode for low-quality/blurry query images
- Ensure the target area is densely indexed (use radius ≤ 1km for high precision)
- Try AI Coarse mode if you have no prior location knowledge (requires `GEMINI_API_KEY`)

### `metadata.npz` not found
The compiled index hasn't been built yet. Either:
1. Run the GUI and complete a **Create Index** pass, or
2. Run `build_index.py` standalone

The index auto-builds from `cosplace_parts/` chunks.

### Confidence score is low despite correct result
This is expected when the query image is taken from an unusual angle, at night, or with heavy compression. Enable Ultra Mode and check the spatial consensus cluster — the correct location often appears as the highest-density cluster even if a single candidate has more inliers.

---

## Models Reference

| Model | Role | Source |
|-------|------|--------|
| CosPlace | Global visual place recognition | `pip install` via requirements |
| ALIKED | Local feature extraction (CUDA) | via LightGlue package |
| DISK | Local feature extraction (MPS/CPU) | via LightGlue package |
| LightGlue | Deep feature matching | `pip install git+https://github.com/cvg/LightGlue.git` |
| LoFTR | Dense matching, Ultra Mode only | `pip install kornia` |

All model weights are downloaded automatically on first use.
```
