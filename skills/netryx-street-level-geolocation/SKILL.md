```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - netryx geolocation
  - index street view panoramas
  - identify location from photo
  - osint geolocation tool
  - run netryx search
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the exact GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a searchable visual index using CosPlace embeddings, and then matches query images via LightGlue feature matching — achieving sub-50m accuracy with no internet image presence required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching
pip install kornia
```

### Requirements
- Python 3.9+
- GPU: NVIDIA (CUDA, 4GB+ VRAM) or Apple Silicon (MPS) or CPU (slow)
- RAM: 8GB minimum, 16GB+ recommended
- Storage: 10GB+ (index grows with coverage area)

### Optional: Gemini API for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### 1. Create an Index (Index a Geographic Area)

In the GUI:
1. Select **Create** mode
2. Enter center lat/lon of target area
3. Set radius (0.5–10km)
4. Set grid resolution (default: 300, don't change)
5. Click **Create Index**

Index build times:

| Radius | Panoramas | Time (M2 Max) | Size |
|--------|-----------|---------------|------|
| 0.5km  | ~500      | 30 min        | ~60MB |
| 1km    | ~2,000    | 1–2 hrs       | ~250MB |
| 5km    | ~30,000   | 8–12 hrs      | ~3GB |
| 10km   | ~100,000  | 24–48 hrs     | ~7GB |

Index is saved incrementally — safe to interrupt and resume.

### 2. Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual**: Enter center lat/lon + radius if you know the approximate region
   - **AI Coarse**: Uses Gemini to estimate region from visual clues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result shown on map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI application
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created)
└── index/
    ├── cosplace_descriptors.npy   # 512-dim descriptor matrix
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Key Components & Code Patterns

### Load CosPlace and Extract a Descriptor

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image

# Load model (auto-selects CUDA > MPS > CPU)
model = load_cosplace_model()

# Extract 512-dim descriptor from any PIL image
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = get_descriptor(model, img)  # shape: (512,)
```

### Build or Load the Searchable Index

```python
import numpy as np

# Load prebuilt index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]       # (N,)
lons = meta["lons"]       # (N,)
panoids = meta["panoids"] # (N,)
headings = meta["headings"] # (N,)
```

### Radius-Filtered Cosine Search

```python
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_index(query_descriptor, descriptors, lats, lons,
                 center_lat, center_lon, radius_km=2.0, top_k=500):
    # Radius mask
    mask = np.array([
        haversine_km(center_lat, center_lon, la, lo) <= radius_km
        for la, lo in zip(lats, lons)
    ])
    
    filtered_desc = descriptors[mask]
    filtered_idx = np.where(mask)[0]
    
    # Cosine similarity (descriptors assumed L2-normalized)
    query_norm = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    scores = filtered_desc @ query_norm
    
    top_local = np.argsort(scores)[::-1][:top_k]
    top_global = filtered_idx[top_local]
    
    return top_global, scores[top_local]

# Usage
top_indices, scores = search_index(
    descriptor, descriptors, lats, lons,
    center_lat=48.8566, center_lon=2.3522, radius_km=1.5
)
```

### Flip-Augmented Search (Catches Reversed Perspectives)

```python
from PIL import ImageOps

img_flipped = ImageOps.mirror(img)
descriptor_flipped = get_descriptor(model, img_flipped)

# Search with both and merge
idx_orig, scores_orig = search_index(descriptor, descriptors, lats, lons, clat, clon)
idx_flip, scores_flip = search_index(descriptor_flipped, descriptors, lats, lons, clat, clon)

# Combine unique candidates
all_candidates = list(set(idx_orig.tolist() + idx_flip.tolist()))
```

### Feature Matching with LightGlue (ALIKED on CUDA)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

def match_images(img_path_query, img_path_candidate):
    img0 = load_image(img_path_query).to(device)
    img1 = load_image(img_path_candidate).to(device)

    with torch.no_grad():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
    matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]
    return matched_kps0, matched_kps1, matches01["matching_scores0"]
```

### RANSAC Geometric Verification

```python
import cv2
import numpy as np

def ransac_inliers(kps0, kps1, threshold=4.0):
    """Returns number of geometrically consistent inliers."""
    if len(kps0) < 4:
        return 0
    pts0 = kps0.cpu().numpy()
    pts1 = kps1.cpu().numpy()
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, threshold)
    if mask is None:
        return 0
    return int(mask.sum())

# Usage after matching
kps0, kps1, scores = match_images("query.jpg", "candidate.jpg")
inliers = ransac_inliers(kps0, kps1)
print(f"Verified inliers: {inliers}")  # >30 = strong match
```

### DISK on MPS/CPU (Mac fallback)

```python
from lightglue import DISK

extractor = DISK(max_num_keypoints=768).eval().to(device)
matcher = LightGlue(features="disk").eval().to(device)
# Same API as ALIKED
```

### Ultra Mode: LoFTR Dense Matching

```python
import kornia
import kornia.feature as KF
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_match(gray0: torch.Tensor, gray1: torch.Tensor):
    """gray0/gray1: (1, 1, H, W) float tensors [0,1]"""
    with torch.no_grad():
        input_dict = {"image0": gray0.to(device), "image1": gray1.to(device)}
        correspondences = loftr(input_dict)
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    return mkpts0, mkpts1
```

### Spatial Consensus Clustering

```python
from collections import defaultdict

def spatial_consensus(candidates, lats, lons, scores, cell_size_m=50):
    """Prefer clusters of matches over isolated high-score outliers."""
    clusters = defaultdict(list)
    for idx, score in zip(candidates, scores):
        # Quantize to ~50m grid cells
        cell_lat = round(lats[idx] / (cell_size_m / 111320), 4)
        cell_lon = round(lons[idx] / (cell_size_m / (111320 * 0.7)), 4)
        clusters[(cell_lat, cell_lon)].append((idx, score))

    best_cluster = max(clusters.values(), key=lambda c: sum(s for _, s in c))
    best_idx = max(best_cluster, key=lambda x: x[1])[0]
    return best_idx, lats[best_idx], lons[best_idx]
```

---

## Build Index Programmatically (Large Scale)

Use `build_index.py` for large datasets outside the GUI:

```bash
# High-performance standalone indexer
python build_index.py \
  --center 48.8566,2.3522 \
  --radius 5 \
  --resolution 300 \
  --output ./index
```

The index is built incrementally from `cosplace_parts/*.npz` chunks — safe to restart.

---

## Index Architecture

All areas share a single unified index. The radius filter at search time isolates regions:

```
Create Mode:
  Grid points → Street View API → Panoramas → CosPlace → cosplace_parts/*.npz

Auto-build:
  cosplace_parts/*.npz → index/cosplace_descriptors.npy + index/metadata.npz

Search Mode:
  Query → CosPlace descriptor → radius-filtered cosine search
       → top-500 candidates → panorama download → ALIKED/DISK + LightGlue
       → RANSAC → heading refinement → spatial consensus → GPS result
```

You can index Paris + Tokyo + London into the same index and search each by specifying center coordinates and radius.

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| CosPlace | Global 512-dim visual fingerprint | Any |
| ALIKED | Local keypoints (1024 kp) | CUDA preferred |
| DISK | Local keypoints (768 kp) | MPS/CPU fallback |
| LightGlue | Deep feature matching | Any |
| LoFTR | Dense matching, blur-robust (Ultra Mode) | CUDA/MPS |

---

## Confidence Interpretation

| Inliers (RANSAC) | Confidence |
|------------------|------------|
| < 15 | Low — treat as estimate |
| 15–50 | Medium — likely correct area |
| 50–150 | High — sub-50m accuracy |
| 150+ | Very High — near-exact match |

---

## Troubleshooting

**GUI appears blank on macOS**
```bash
brew install python-tk@3.11
# Re-run with the Homebrew Python
/opt/homebrew/bin/python3.11 test_super.py
```

**LightGlue import error**
```bash
pip install git+https://github.com/cvg/LightGlue.git --force-reinstall
```

**CUDA out of memory**
- Reduce `max_num_keypoints` in ALIKED (try 512 instead of 1024)
- Reduce top-K candidates from 500 to 200

**MPS not detected on Apple Silicon**
```python
import torch
print(torch.backends.mps.is_available())  # Must be True
# If False: pip install --upgrade torch torchvision
```

**Index search returns no results**
- Check that `index/cosplace_descriptors.npy` exists (run Create mode first)
- Verify your search radius actually overlaps the indexed area
- Confirm `metadata.npz` has matching entry count: `len(lats) == len(descriptors)`

**LoFTR not available**
```bash
pip install kornia
python -c "import kornia.feature; print('LoFTR OK')"
```

**Slow indexing**
- Use `build_index.py` instead of GUI for large areas
- Ensure GPU is active: check `torch.cuda.is_available()` or `torch.backends.mps.is_available()`
- Indexing is I/O bound on panorama downloads — broadband required

---

## Environment Variables

```bash
export GEMINI_API_KEY="..."    # Optional: AI Coarse geolocation mode only
```

No other secrets required. Street View data is fetched via public tile endpoints during indexing and search.
```
