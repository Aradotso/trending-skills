```markdown
---
name: netryx-street-level-geolocation
description: Expertise in using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - reverse geolocate image
  - osint geolocation tool
  - find location from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies the precise GPS coordinates of any street-level photograph. It works by matching your query image against an index of crawled street-view panoramas using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and spatial refinement. Sub-50m accuracy. No internet-indexed landmarks required. Runs on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR dense matching for Ultra Mode
pip install kornia
```

### Requirements
- Python 3.9+ (3.10+ recommended)
- GPU: NVIDIA (CUDA, 4GB+ VRAM) or Apple Silicon (MPS) — CPU works but is slow
- RAM: 8GB minimum, 16GB+ recommended
- Storage: 10GB minimum; 50GB+ for large indexed areas

### Optional: Gemini API for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

> macOS users: if the GUI appears blank, run `brew install python-tk@3.11` to upgrade tkinter.

---

## Core Workflow

### Step 1 — Create an Index

Before you can search, you must index a geographic area. This crawls Street View panoramas and extracts CosPlace fingerprints into a local database.

**In the GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon) of the area
3. Set search radius (start with 0.5–1 km for testing)
4. Set grid resolution (300 is the recommended default — don't change this)
5. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is resumable — if interrupted, restarting continues from the last checkpoint.

### Step 2 — Search

**In the GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Let Gemini infer the region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search**, then **Start Full Search**
5. A real-time visualization shows candidates being evaluated
6. Result: GPS coordinates + confidence score displayed on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI, indexing, and search
├── cosplace_utils.py      # CosPlace model loader and descriptor extraction
├── build_index.py         # High-performance standalone index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw per-chunk embeddings (.npz), written during indexing
├── index/
│   ├── cosplace_descriptors.npy   # Compiled 512-dim descriptor matrix
│   └── metadata.npz               # Coordinates, headings, panorama IDs
└── README.md
```

---

## Pipeline Deep Dive

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dimensional descriptor from the query image
- Also extracts from a horizontally flipped version (handles reversed perspectives)
- Cosine similarity search against the full index
- Haversine radius filter restricts results to your specified area
- Returns top 500–1000 candidates
- Runs in **<1 second** (single matrix multiply)

### Stage 2 — Local Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads panorama tiles from Street View (8 tiles, stitched)
- Crops at indexed heading angle
- Generates 3 FOV crops: 70°, 90°, 110° (handles zoom mismatch)
- Feature extractor selection is automatic:
  - **CUDA**: ALIKED (1024 keypoints)
  - **MPS / CPU**: DISK (768 keypoints)
- LightGlue matches keypoints; RANSAC filters geometric outliers
- Runs in **2–5 minutes** for 300–500 candidates

### Stage 3 — Refinement
- **Heading refinement**: Tests ±45° at 15° steps for top 15 candidates
- **Spatial consensus**: Clusters matches into 50m cells; prefers clusters over lone outliers
- **Confidence scoring**: Evaluates geographic clustering and uniqueness ratio vs. runner-up

### Ultra Mode
Enable for difficult images (night shots, motion blur, low texture):
- **LoFTR**: Detector-free dense matching — works without reliable keypoints
- **Descriptor hopping**: Re-searches index using the clean matched panorama's descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

---

## Index Architecture

All embeddings live in a **single unified index**. Cities do not need separate indexes.

```
# Index multiple cities into the same index:
# Paris center: 48.8566, 2.3522, radius 5km
# London center: 51.5074, -0.1278, radius 10km
# Tel Aviv center: 32.0853, 34.7818, radius 3km
# → All stored together in cosplace_descriptors.npy

# Search is scoped by coordinates + radius at query time:
# center=48.8566,2.3522  radius=5km  → only Paris results returned
# center=51.5074,-0.1278 radius=10km → only London results returned
```

**Data flow:**
```
Create Mode:
  Grid points → Street View API → Panoramas → Crops
    → CosPlace → cosplace_parts/*.npz (incremental chunks)

Auto-build:
  cosplace_parts/*.npz → index/cosplace_descriptors.npy
                       → index/metadata.npz

Search Mode:
  Query image → CosPlace descriptor
    → Index search (cosine sim + haversine filter)
    → Download top-N panoramas
    → ALIKED/DISK keypoints + LightGlue matching
    → RANSAC verification → Refinement → GPS output
```

---

## Using CosPlace Utilities Directly

```python
# cosplace_utils.py exposes model loading and descriptor extraction
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-selects cuda / mps / cpu)
model = load_cosplace_model()

# Extract a 512-dim descriptor from an image file
img = Image.open("query.jpg")
descriptor = get_descriptor(model, img)  # returns np.ndarray shape (512,)

# Extract descriptor from flipped version (catches reversed perspectives)
descriptor_flipped = get_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT))
```

---

## Building a Large Index Programmatically

For large-scale indexing outside the GUI, use `build_index.py`:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --resolution 300
```

This writes chunks to `cosplace_parts/` incrementally and is safe to interrupt and resume.

---

## Common Patterns

### Checking GPU/Device in Use

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
    print("Using NVIDIA CUDA")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple MPS")
else:
    device = "cpu"
    print("Using CPU (slow)")
```

### Verifying Index is Built

```python
import os
import numpy as np

index_dir = "index"
desc_path = os.path.join(index_dir, "cosplace_descriptors.npy")
meta_path = os.path.join(index_dir, "metadata.npz")

if os.path.exists(desc_path) and os.path.exists(meta_path):
    descs = np.load(desc_path)
    meta = np.load(meta_path)
    print(f"Index loaded: {descs.shape[0]} panoramas indexed")
    print(f"Descriptor shape: {descs.shape}")  # (N, 512)
else:
    print("Index not found — run Create mode first")
```

### Manual Cosine Similarity Search (Radius-Filtered)

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Load index
descs = np.load("index/cosplace_descriptors.npy")   # (N, 512) float32
meta  = np.load("index/metadata.npz", allow_pickle=True)
lats  = meta["lats"]    # (N,)
lons  = meta["lons"]    # (N,)

# Query parameters
query_desc = get_descriptor(model, query_image)  # (512,)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 2.0
top_k = 500

# Radius mask (haversine)
mask = np.array([
    haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
    for i in range(len(lats))
])

# Cosine similarity search over masked entries
masked_descs = descs[mask]          # (M, 512)
query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
db_norms   = masked_descs / (np.linalg.norm(masked_descs, axis=1, keepdims=True) + 1e-8)
sims = db_norms @ query_norm        # (M,)

top_indices_local = np.argsort(sims)[::-1][:top_k]
masked_indices    = np.where(mask)[0]
top_indices       = masked_indices[top_indices_local]

print(f"Top match: lat={lats[top_indices[0]]:.6f}, lon={lons[top_indices[0]]:.6f}")
print(f"Similarity score: {sims[top_indices_local[0]]:.4f}")
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # match your Python version
```

### LightGlue import error
```bash
# Must be installed from GitHub, not PyPI
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory
- Reduce `top_k` candidates (default 500 → try 200)
- Use DISK instead of ALIKED (fewer keypoints: 768 vs 1024)
- Close other GPU processes

### MPS errors on Apple Silicon
- Ensure PyTorch ≥ 2.0 for stable MPS support: `pip install torch --upgrade`
- DISK is used automatically on MPS (ALIKED is CUDA-only)

### LoFTR not available (Ultra Mode)
```bash
pip install kornia
```

### Index not found at search time
- Confirm `cosplace_parts/` has `.npz` files (indexing produced output)
- Run the **Auto-build** step in the GUI, or re-run `build_index.py`
- Check that `index/cosplace_descriptors.npy` and `index/metadata.npz` exist

### Poor match accuracy
1. Enable **Ultra Mode** (adds LoFTR + descriptor hopping + neighborhood expansion)
2. Expand your search radius — the correct location may be just outside your filter
3. Ensure the target area is indexed at sufficient density (300 grid resolution)
4. Try AI Coarse mode if you're unsure of the approximate region

### Indexing stalled / no new panoramas
- Street View coverage may be sparse in the target area
- Try a different center coordinate with known coverage
- Reduce grid resolution cautiously (higher value = denser, but slower)

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim place descriptor | All |
| [ALIKED](https://github.com/naver/alike) | Local keypoints (1024 kp) | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints (768 kp) | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense matching, Ultra Mode | All (via kornia) |

Model selection is **automatic** based on detected hardware — no manual configuration required.
```
