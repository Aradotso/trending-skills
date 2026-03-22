---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to precise GPS coordinates using CosPlace + LightGlue CV pipeline.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - index street view panoramas
  - run netryx search
  - identify location from photo
  - use netryx geolocation tool
  - osint image geolocation
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, builds a searchable visual index using CosPlace descriptors, then matches query images via ALIKED/DISK keypoints + LightGlue deep feature matching + RANSAC verification. Sub-50m accuracy, no internet presence required for the target location.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (install from source)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode on difficult images
pip install kornia
```

**GPU requirements:**
- NVIDIA CUDA: 4GB+ VRAM (ALIKED feature extractor)
- Apple Silicon (M1+): MPS (DISK feature extractor)
- CPU: Works, significantly slower

**macOS tkinter fix** (blank GUI):
```bash
brew install python-tk@3.11   # match your Python version
```

---

## Environment Variables

```bash
# Optional — only needed for AI Coarse blind geolocation mode
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles indexing, searching, and result visualization with a real-time map.

---

## Core Workflow

### 1. Create an Index

Index a geographic area by crawling street-view panoramas and extracting CosPlace fingerprints. All areas share one unified index; radius filtering handles separation at search time.

**In the GUI:**
1. Select **Create** mode
2. Enter center lat/lon (e.g. `48.8566, 2.3522` for Paris)
3. Set radius in km (start with `0.5` for testing)
4. Grid resolution: leave at `300` (default)
5. Click **Create Index**

**Index size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk |
|--------|-----------|---------------|------|
| 0.5 km | 500 | 30 min | 60 MB |
| 1 km | 2,000 | 1–2 hr | 250 MB |
| 5 km | 30,000 | 8–12 hr | 3 GB |
| 10 km | 100,000 | 24–48 hr | 7 GB |

Indexing is **resumable** — interrupting and restarting picks up where it left off from `cosplace_parts/`.

**For large areas, use the standalone high-performance builder:**
```bash
python build_index.py
```

### 2. Search / Geolocate an Image

**In the GUI:**
1. Select **Search** mode
2. Upload your street photo
3. Choose method:
   - **Manual**: Enter approximate center lat/lon + radius (faster, more accurate)
   - **AI Coarse**: Gemini analyzes the image for region clues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result shows GPS coordinates + confidence score on map

**Enable Ultra Mode** for degraded images (night, blur, low texture):
- Adds LoFTR dense matching
- Descriptor hopping (re-searches index using matched panorama's clean descriptor)
- Neighborhood expansion (±100m around best match)

---

## Project Structure

```
netryx/
├── test_super.py           # Main application — GUI, indexing, search pipeline
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone index builder for large datasets
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Pipeline Deep Dive

### Stage 1 — Global Retrieval (CosPlace)

Extracts a 512-dim descriptor from query + horizontally flipped query, then runs cosine similarity against the full index with haversine radius filter. Returns top 500–1000 candidates. Runs in <1 second via single matrix multiplication.

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

For each candidate:
- Downloads Google Street View panorama (8 tiles, stitched)
- Crops rectilinear view at indexed heading
- Generates multi-FOV crops: 70°, 90°, 110° (handles zoom mismatch)
- ALIKED (CUDA) or DISK (MPS/CPU) extracts local keypoints
- LightGlue matches keypoints between query and candidate
- RANSAC filters to geometrically consistent inliers
- Candidate with most inliers wins

Processes 300–500 candidates in 2–5 minutes.

### Stage 3 — Refinement

- **Heading refinement**: Tests ±45° at 15° steps × 3 FOVs for top 15 candidates
- **Spatial consensus**: Clusters matches into 50m cells; clusters beat single high-inlier outliers
- **Confidence scoring**: Evaluates geographic clustering + uniqueness ratio (best vs runner-up)

---

## Using CosPlace Utilities Directly

```python
# cosplace_utils.py exposes descriptor extraction
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor shape: (512,) — normalized float32 vector

# Search against index manually
import numpy as np
index = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta  = np.load("index/metadata.npz", allow_pickle=True)

scores = index @ descriptor                          # cosine similarity
top_k  = np.argsort(scores)[::-1][:500]

lats = meta["lats"][top_k]
lons = meta["lons"][top_k]
pano_ids = meta["pano_ids"][top_k]
headings = meta["headings"][top_k]
```

### Haversine Radius Filter

```python
import numpy as np

def haversine_filter(lats, lons, center_lat, center_lon, radius_km):
    """Return boolean mask of points within radius_km of center."""
    R = 6371.0
    dlat = np.radians(lats - center_lat)
    dlon = np.radians(lons - center_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(center_lat))
         * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a)) <= radius_km

# Usage
mask = haversine_filter(lats, lons, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
filtered_top_k = top_k[mask]
```

---

## Multi-Index Strategy (Multiple Cities)

All cities share one index file. Use coordinates + radius to isolate regions:

```python
# Index Paris
# GUI: center=48.8566,2.3522  radius=5km

# Index London  
# GUI: center=51.5074,-0.1278  radius=5km

# Search Paris only — London results never appear
# GUI: center=48.8566,2.3522  radius=5km

# Search London only
# GUI: center=51.5074,-0.1278  radius=5km
```

No per-city index files needed. The haversine filter handles isolation automatically.

---

## Common Patterns

### Check if Index Exists and Is Built

```python
import os
import numpy as np

def index_ready(index_dir="index"):
    desc_path = os.path.join(index_dir, "cosplace_descriptors.npy")
    meta_path = os.path.join(index_dir, "metadata.npz")
    if not (os.path.exists(desc_path) and os.path.exists(meta_path)):
        return False, "Index files missing — run Create Index or build_index.py"
    desc = np.load(desc_path)
    meta = np.load(meta_path, allow_pickle=True)
    n_desc = desc.shape[0]
    n_meta = len(meta["lats"])
    if n_desc != n_meta:
        return False, f"Mismatch: {n_desc} descriptors vs {n_meta} metadata rows"
    return True, f"Index ready: {n_desc} panoramas"

ok, msg = index_ready()
print(msg)
```

### Detect Available Hardware

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "ALIKED"      # Best: NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "DISK"         # Good: Apple Silicon
    else:
        return torch.device("cpu"), "DISK"         # Fallback: CPU

device, extractor = get_device()
print(f"Using {device} with {extractor} feature extractor")
```

### Batch Index Multiple Areas

```python
# Run indexing programmatically by calling the GUI's indexing logic
# (test_super.py's IndexThread can be imported and driven directly)
# For large-scale use, prefer build_index.py with custom config

areas = [
    {"name": "Paris",    "lat": 48.8566,  "lon": 2.3522,   "radius_km": 2},
    {"name": "Berlin",   "lat": 52.5200,  "lon": 13.4050,  "radius_km": 2},
    {"name": "NYC",      "lat": 40.7128,  "lon": -74.0060, "radius_km": 2},
]

# Run test_super.py GUI for each area in Create mode with above coords
# All results merge into the same cosplace_parts/ and index/
# Each search uses its own center+radius to isolate results
```

---

## Troubleshooting

### GUI is blank / doesn't render (macOS)
```bash
brew install python-tk@3.11   # use your exact Python version
# Then reactivate venv and relaunch
```

### `ModuleNotFoundError: No module named 'lightglue'`
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — only the GitHub source is supported
```

### `ModuleNotFoundError: No module named 'kornia'` (Ultra Mode)
```bash
pip install kornia
# Only required if using Ultra Mode (LoFTR)
```

### Out of VRAM during search
- Reduce candidate count in GUI (lower top-K)
- Disable Ultra Mode
- Switch to CPU: remove CUDA-related env vars and relaunch
- Close other GPU-heavy applications

### Index search returns no results
```python
# 1. Confirm index is populated
import numpy as np
d = np.load("index/cosplace_descriptors.npy")
print(f"Index has {d.shape[0]} entries")  # Should be > 0

# 2. Confirm your search radius covers the indexed area
# If you indexed lat=48.8566,lon=2.3522,radius=1km
# searching lat=51.5074,lon=-0.1278,radius=5km returns 0 results — correct behavior

# 3. Rebuild compiled index from parts if metadata.npz is stale
python build_index.py
```

### Indexing stalls / resumes from wrong point
```bash
# cosplace_parts/ holds incremental chunks — safe to inspect
ls cosplace_parts/
# Each .npz = one batch of panoramas already processed
# Delete a corrupt chunk and re-run to regenerate only that chunk
```

### Low confidence / wrong location result
- Enable **Ultra Mode** for difficult images
- Expand search radius in Manual mode
- Ensure the indexed area actually contains the query location
- Try AI Coarse mode to get a better region estimate first, then refine with Manual

### Windows path issues
```bash
# Activate venv on Windows:
venv\Scripts\activate

# Use backslashes or raw strings for paths in scripts:
index_dir = r"C:\Users\user\netryx\index"
```

---

## Key Files Reference

| File | Role |
|------|------|
| `test_super.py` | Full pipeline: GUI, indexing thread, search thread, map visualization |
| `cosplace_utils.py` | Load CosPlace model, extract 512-dim descriptors from PIL images |
| `build_index.py` | High-performance standalone index compiler for large datasets |
| `index/cosplace_descriptors.npy` | NumPy array `(N, 512)` — all CosPlace embeddings |
| `index/metadata.npz` | Arrays: `lats`, `lons`, `headings`, `pano_ids` — one entry per descriptor |
| `cosplace_parts/*.npz` | Raw incremental chunks written during indexing |

---

## Models Reference

| Model | Task | Hardware |
|-------|------|----------|
| CosPlace (CVPR 2022) | Global place recognition, 512-dim descriptor | Any |
| ALIKED (IEEE TIP 2023) | Local keypoint extraction, 1024 keypoints | CUDA only |
| DISK (NeurIPS 2020) | Local keypoint extraction, 768 keypoints | MPS / CPU |
| LightGlue (ICCV 2023) | Deep feature matching | Any |
| LoFTR (CVPR 2021) | Detector-free dense matching (Ultra Mode) | Any, needs `kornia` |
