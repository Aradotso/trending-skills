---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to precise GPS coordinates using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - index street view panoramas
  - run netryx geolocation
  - use netryx to locate an image
  - visual place recognition pipeline
  - find where a photo was taken
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the precise GPS coordinates of any street-level photograph. It crawls street-view panoramas into a searchable index, then matches a query image against that index using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature extraction (ALIKED/DISK), and deep feature matching (LightGlue). Sub-50m accuracy, no internet presence of the target image required, runs entirely on local hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

### Optional: Gemini API key for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"   # never hard-code; use env var
```

### macOS tkinter fix (blank GUI)

```bash
brew install python-tk@3.11   # match your Python version
```

---

## Launch the GUI

```bash
python test_super.py
```

All indexing and searching is driven from this single GUI entry point.

---

## Project Structure

```
netryx/
├── test_super.py           # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py       # CosPlace model loading & descriptor extraction
├── build_index.py          # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (compiled)
    └── metadata.npz               # Coordinates, headings, panoid IDs
```

---

## Core Workflow

### Step 1 — Create an Index

Index an area before searching. The GUI does this interactively; you can also drive it programmatically.

**GUI steps:**
1. Select **Create** mode
2. Enter center `latitude, longitude`
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Indexing time reference:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | 500 | 30 min | ~60 MB |
| 1 km | 2 000 | 1–2 h | ~250 MB |
| 5 km | 30 000 | 8–12 h | ~3 GB |
| 10 km | 100 000 | 24–48 h | ~7 GB |

Indexing is incremental — safe to interrupt and resume.

### Step 2 — Search

**GUI steps:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose **Manual** (provide center coords + radius) or **AI Coarse** (Gemini infers region)
4. Click **Run Search → Start Full Search**
5. Result: GPS pin on map + confidence score

---

## Programmatic Usage

### Extract a CosPlace descriptor

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)  # shape: (512,)
print("Descriptor shape:", descriptor.shape)
```

### Search the index against a query descriptor

```python
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta        = np.load("index/metadata.npz", allow_pickle=True)
latitudes   = meta["latitudes"]    # (N,)
longitudes  = meta["longitudes"]   # (N,)
headings    = meta["headings"]     # (N,)
panoids     = meta["panoids"]      # (N,)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_index(query_descriptor, center_lat, center_lon,
                 radius_km=2.0, top_k=500):
    """Return top-k candidate indices ranked by cosine similarity within radius."""
    # Radius mask
    dists = np.array([
        haversine_km(center_lat, center_lon, lat, lon)
        for lat, lon in zip(latitudes, longitudes)
    ])
    mask = dists <= radius_km

    # Cosine similarity (descriptors assumed L2-normalised)
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    sims = descriptors[mask] @ q                    # cosine scores

    local_indices = np.where(mask)[0]
    ranked = local_indices[np.argsort(sims)[::-1]]  # descending
    return ranked[:top_k]

candidates = search_index(descriptor, center_lat=48.8566,
                          center_lon=2.3522, radius_km=1.0)
print(f"Top candidate panoid: {panoids[candidates[0]]}")
print(f"  lat={latitudes[candidates[0]]:.6f}  lon={longitudes[candidates[0]]:.6f}")
```

### Build / rebuild the compiled index from parts

```python
# Run after adding new cosplace_parts/*.npz chunks
import subprocess
subprocess.run(["python", "build_index.py"], check=True)
```

Or directly from Python if `build_index.py` exposes a function:

```python
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("build_index",
                                               pathlib.Path("build_index.py"))
build_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_index)
build_index.build()   # adjust to actual function name in the file
```

---

## Pipeline Stages in Detail

### Stage 1 — Global Retrieval (CosPlace)

- Extracts 512-dim descriptor from query **and** its horizontal flip
- Both descriptors compared via cosine similarity against the index
- Haversine radius filter restricts candidates to the target area
- Returns top 500–1 000 candidates
- Runs in **< 1 second** (single matrix multiply)

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

```
For each candidate:
  1. Download panorama tiles from Street View (8 tiles, stitched)
  2. Crop rectilinear view at indexed heading
  3. Generate multi-FOV crops: 70°, 90°, 110°
  4. Extract keypoints:
       CUDA  → ALIKED (1024 keypoints)
       MPS/CPU → DISK (768 keypoints)
  5. LightGlue deep feature matching vs. query keypoints
  6. RANSAC geometric verification → inlier count
```

Best match = candidate with most RANSAC-verified inliers.  
Processing 300–500 candidates: **2–5 minutes** depending on hardware.

### Stage 3 — Refinement

```
Heading refinement : top-15 candidates × ±45° at 15° steps × 3 FOVs
Spatial consensus  : cluster matches into 50 m cells; prefer clusters
Confidence score   : geographic clustering tightness + uniqueness ratio
```

### Ultra Mode (optional, slower)

Enable the **Ultra Mode** checkbox in the GUI for difficult images (night, motion blur, low texture).

What it adds:
- **LoFTR** — detector-free dense matching (handles blur/low-contrast)
- **Descriptor hopping** — re-searches index using a descriptor from the matched panorama if initial match is weak (< 50 inliers)
- **Neighbourhood expansion** — searches all panoramas within 100 m of the best match

---

## Configuration Reference

All configuration is passed through the GUI or by modifying constants in `test_super.py`. Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| Grid resolution | 300 | Panorama density during indexing; don't change |
| Radius (search) | user-set | Haversine filter radius in km |
| Top-K candidates | 500–1000 | Candidates passed to Stage 2 |
| Heading steps | ±45° / 15° | Refinement sweep range |
| Spatial cell size | 50 m | Consensus clustering granularity |
| Neighbourhood expansion | 100 m | Ultra Mode only |
| Weak match threshold | 50 inliers | Triggers descriptor hopping in Ultra Mode |

---

## Hardware & Device Selection

```python
import torch

# Netryx auto-selects; mirror this logic in custom scripts
if torch.cuda.is_available():
    device = torch.device("cuda")        # ALIKED, full speed
elif torch.backends.mps.is_available():
    device = torch.device("mps")         # Mac M-series, DISK
else:
    device = torch.device("cpu")         # Works, significantly slower
```

**Minimum requirements:** 4 GB GPU VRAM, 8 GB RAM, Python 3.9+  
**Recommended:** NVIDIA GPU with 8 GB+ VRAM (CUDA) or Apple M1+ (MPS)

---

## Index Design Patterns

### Multi-city indexing

All cities share one unified index. The radius filter at search time isolates results:

```python
# Index Paris, London, Tokyo — all into the same index/
# Then search by specifying center + radius:

# Paris only
candidates = search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=5)

# London only
candidates = search_index(desc, center_lat=51.5074, center_lon=-0.1278, radius_km=5)
```

### Incremental indexing

New `cosplace_parts/*.npz` files are appended automatically during indexing.  
Rebuild the compiled index after adding new areas:

```bash
python build_index.py
```

---

## Troubleshooting

### GUI appears blank on macOS

```bash
brew install python-tk@3.11   # match your exact Python version
```

### `import lightglue` fails

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### `import kornia` fails (Ultra Mode unavailable)

```bash
pip install kornia
```

### CUDA out of memory

- Reduce `top_k` candidates (e.g. 300 instead of 500)
- Switch to DISK instead of ALIKED by forcing MPS/CPU device
- Reduce FOV count if modifying pipeline directly

### Indexing stops / resumes incorrectly

The index writes incrementally to `cosplace_parts/`. Delete corrupted `.npz` files in that folder and re-run; completed chunks are skipped.

### Low confidence score / wrong result

1. Enable **Ultra Mode** for degraded images
2. Increase search radius if location estimate is uncertain
3. Use a higher grid resolution index for the target area (re-index)
4. Try **AI Coarse** mode if manual center coordinates are uncertain

### Gemini AI Coarse mode not available

```bash
export GEMINI_API_KEY="your_key_here"
```
Verify it is set: `echo $GEMINI_API_KEY`

---

## Key Dependencies

| Package | Role |
|---------|------|
| `torch` | Model inference backbone |
| `lightglue` (GitHub) | Deep feature matching |
| `kornia` | LoFTR dense matching (Ultra Mode) |
| `numpy` | Index storage and cosine similarity |
| `Pillow` | Image loading and preprocessing |
| `tkinter` | GUI (stdlib, may need upgrade on macOS) |

---

## Quick-Start Checklist

```
[ ] Clone repo and create venv
[ ] pip install -r requirements.txt
[ ] pip install git+https://github.com/cvg/LightGlue.git
[ ] (optional) pip install kornia
[ ] (optional) export GEMINI_API_KEY=...
[ ] python test_super.py
[ ] Create mode → set coords + radius → Create Index  (wait for completion)
[ ] python build_index.py  (if not auto-built)
[ ] Search mode → upload photo → Manual/AI Coarse → Run Search
[ ] Read GPS result + confidence score on map
```
