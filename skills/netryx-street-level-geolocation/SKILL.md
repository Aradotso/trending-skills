---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to sub-50m GPS accuracy using CosPlace, ALIKED/DISK, and LightGlue — runs entirely on local hardware.
triggers:
  - geolocate a street photo
  - find GPS coordinates from street image
  - street level geolocation
  - identify location from photo
  - reverse image geolocation
  - index street view panoramas
  - run netryx geolocation
  - osint image location finder
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that finds precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls street-view panoramas into a searchable index, then uses a three-stage computer vision pipeline — global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and geometric verification (RANSAC) — to pinpoint the photo's location. No cloud service required; everything runs on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue matcher
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Optional — Gemini API key for AI Coarse region guessing

```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

GPU backends: **CUDA** (NVIDIA), **MPS** (Apple Silicon M1+), **CPU** (slow fallback).

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank window fix: `brew install python-tk@3.11`

---

## Core Workflow

### 1 — Create an Index

Index a geographic area before searching. The indexer crawls street-view panoramas, extracts CosPlace fingerprints, and saves them to `cosplace_parts/`.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon, radius (km), grid resolution (default 300)
3. Click **Create Index** — resumes automatically if interrupted

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 h         | ~250 MB    |
| 5 km   | ~30,000   | 8–12 h        | ~3 GB      |
| 10 km  | ~100,000  | 24–48 h       | ~7 GB      |

**Standalone high-performance builder (large areas):**

```bash
python build_index.py
```

### 2 — Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose **Manual** (enter lat/lon + radius) or **AI Coarse** (Gemini guesses region)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS pin on map + confidence score

**Ultra Mode** (checkbox): Enables LoFTR dense matching, descriptor hopping, and 100 m neighborhood expansion. Use for night shots, blurry, or low-texture images. ~2–3× slower.

---

## Project Structure

```
netryx/
├── test_super.py          # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loader + descriptor extraction
├── build_index.py         # Standalone index builder for large datasets
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (built during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Pipeline Deep-Dive

### Stage 1 — Global Retrieval (CosPlace)

```python
# cosplace_utils.py pattern — extract a 512-dim fingerprint
from cosplace_utils import get_cosplace_model, get_descriptor
import torch
from PIL import Image

model = get_cosplace_model()          # loads CosPlace ResNet-50 backbone
img   = Image.open("query.jpg")

descriptor        = get_descriptor(model, img)           # (512,) tensor
descriptor_flipped = get_descriptor(model, img.transpose(
                        Image.FLIP_LEFT_RIGHT))          # catches reversed views

# Both descriptors are stacked and compared against the index via cosine similarity
```

Index search (cosine similarity + haversine radius filter) returns top 500–1000 candidates in < 1 second regardless of index size.

### Stage 2 — Local Feature Matching (ALIKED/DISK + LightGlue)

```python
# Automatic device selection mirrors test_super.py behaviour
import torch

if torch.cuda.is_available():
    device   = "cuda"
    extractor_name = "aliked"    # 1024 keypoints
elif torch.backends.mps.is_available():
    device   = "mps"
    extractor_name = "disk"      # 768 keypoints — stable on MPS
else:
    device   = "cpu"
    extractor_name = "disk"

from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

extractor = (ALIKED(max_num_keypoints=1024) if extractor_name == "aliked"
             else DISK(max_num_keypoints=768)).eval().to(device)
matcher   = LightGlue(features=extractor_name).eval().to(device)

def match_pair(img_path_a, img_path_b):
    img_a = load_image(img_path_a).to(device)
    img_b = load_image(img_path_b).to(device)

    feats_a = extractor.extract(img_a)
    feats_b = extractor.extract(img_b)

    matches_raw = matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, matches_raw = rbd(feats_a), rbd(feats_b), rbd(matches_raw)

    kpts_a = feats_a["keypoints"][matches_raw["matches"][..., 0]]
    kpts_b = feats_b["keypoints"][matches_raw["matches"][..., 1]]
    return kpts_a, kpts_b          # matched keypoint coordinates
```

### Stage 2b — RANSAC Geometric Verification

```python
import cv2
import numpy as np

def count_inliers(kpts_a: torch.Tensor, kpts_b: torch.Tensor) -> int:
    """Returns number of geometrically consistent matches."""
    if len(kpts_a) < 4:
        return 0
    pts_a = kpts_a.cpu().numpy()
    pts_b = kpts_b.cpu().numpy()
    _, mask = cv2.findFundamentalMat(
        pts_a, pts_b,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )
    return int(mask.sum()) if mask is not None else 0
```

### Stage 3 — Multi-FOV Heading Refinement

The pipeline tests 3 fields of view (70°, 90°, 110°) × heading offsets (±45° in 15° steps) for the top 15 candidates to handle zoom/direction mismatches.

```python
FOV_LIST     = [70, 90, 110]
HEADING_OFFSETS = [-45, -30, -15, 0, 15, 30, 45]   # degrees

def refine_candidate(panoid, base_heading, query_kpts, extractor, matcher, device):
    best_inliers = 0
    best_heading = base_heading
    for fov in FOV_LIST:
        for offset in HEADING_OFFSETS:
            heading = (base_heading + offset) % 360
            crop    = download_sv_crop(panoid, heading, fov)   # fetch from Street View
            kpts_q, kpts_c = match_pair_tensors(query_kpts, crop,
                                                extractor, matcher, device)
            n = count_inliers(kpts_q, kpts_c)
            if n > best_inliers:
                best_inliers = n
                best_heading = heading
    return best_heading, best_inliers
```

### Ultra Mode — LoFTR Dense Matching

```python
import kornia.feature as KF
import torch

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_inliers(img_a_gray: torch.Tensor, img_b_gray: torch.Tensor) -> int:
    """Detector-free matching — handles blur and low contrast."""
    with torch.no_grad():
        out = loftr({"image0": img_a_gray.unsqueeze(0),
                     "image1": img_b_gray.unsqueeze(0)})
    kpts_a = out["keypoints0"].squeeze(0)
    kpts_b = out["keypoints1"].squeeze(0)
    return count_inliers(kpts_a, kpts_b)
```

---

## Index Management

All cities/areas share **one unified index**. Radius filtering at query time isolates the right region — no city-selection step needed.

```
cosplace_parts/          ← incremental .npz chunks written during indexing
index/
  cosplace_descriptors.npy   ← stacked (N, 512) float32 array
  metadata.npz               ← lat, lon, heading, panoid arrays of length N
```

**Auto-build the searchable index from parts:**

The GUI triggers this automatically, or you can call the standalone builder:

```bash
python build_index.py
```

**Load the index manually for custom scripts:**

```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta        = np.load("index/metadata.npz")
lats        = meta["lats"]      # (N,)
lons        = meta["lons"]      # (N,)
headings    = meta["headings"]  # (N,)
panoids     = meta["panoids"]   # (N,) str

print(f"Index contains {len(lats):,} panorama views")
```

**Radius-filtered cosine search:**

```python
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_desc, center_lat, center_lon, radius_km, top_k=500):
    """Returns indices of top_k candidates within radius, sorted by similarity."""
    import numpy as np
    
    descriptors = np.load("index/cosplace_descriptors.npy")
    meta        = np.load("index/metadata.npz")
    
    # Haversine mask
    dists = np.array([haversine_km(center_lat, center_lon, la, lo)
                      for la, lo in zip(meta["lats"], meta["lons"])])
    mask  = dists <= radius_km
    
    if mask.sum() == 0:
        return []
    
    sims      = cosine_similarity(query_desc.reshape(1, -1),
                                  descriptors[mask])[0]
    local_idx = np.argsort(sims)[::-1][:top_k]
    global_idx = np.where(mask)[0][local_idx]
    return global_idx, sims[local_idx]
```

---

## Common Patterns

### Batch geolocate a folder of images

```python
from pathlib import Path
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image

model = get_cosplace_model()

results = {}
for img_path in Path("photos/").glob("*.jpg"):
    img  = Image.open(img_path)
    desc = get_descriptor(model, img)
    idxs, sims = search_index(desc,
                              center_lat=48.8566,   # Paris
                              center_lon=2.3522,
                              radius_km=5.0)
    meta = np.load("index/metadata.npz")
    best = idxs[0]
    results[img_path.name] = {
        "lat":        float(meta["lats"][best]),
        "lon":        float(meta["lons"][best]),
        "confidence": float(sims[0]),
        "panoid":     str(meta["panoids"][best]),
    }
    print(f"{img_path.name}: {results[img_path.name]}")
```

### Check available device

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

print("Running on:", get_device())
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GUI window blank on macOS | Old system tkinter | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `No candidates found` | Index empty or wrong radius | Re-check center coords; increase radius |
| Low inliers (<20) on valid photo | FOV/zoom mismatch | Enable **Ultra Mode**; check heading refinement logs |
| MPS crash with ALIKED | ALIKED unstable on MPS | Pipeline auto-selects DISK on MPS — no action needed |
| LoFTR import error | kornia not installed | `pip install kornia` |
| Slow indexing | Single-threaded on CPU | Use `build_index.py` for parallel CPU build, or use CUDA GPU |
| `GEMINI_API_KEY` not found | Env var not set | `export GEMINI_API_KEY=...` — only needed for AI Coarse mode |
| Index search returns wrong city | Radius too large | Tighten radius; all cities share one index, radius is the selector |

### Confidence score interpretation

| Score | Meaning |
|-------|---------|
| > 80 inliers | High confidence — strong geometric verification |
| 40–80 inliers | Medium — result likely correct, verify visually |
| < 40 inliers | Low — try Ultra Mode or expand index coverage |

---

## Key Files Reference

| File | Role |
|------|------|
| `test_super.py` | Entry point — full GUI, indexing loop, search pipeline |
| `cosplace_utils.py` | `get_cosplace_model()`, `get_descriptor()` |
| `build_index.py` | High-throughput CLI index builder for large areas |
| `index/cosplace_descriptors.npy` | Searchable descriptor matrix `(N, 512)` |
| `index/metadata.npz` | Parallel arrays: `lats`, `lons`, `headings`, `panoids` |
| `cosplace_parts/*.npz` | Incremental chunks — safe to delete after index build |
