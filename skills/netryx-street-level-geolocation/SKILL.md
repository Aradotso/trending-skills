```markdown
---
name: netryx-street-level-geolocation
description: Use Netryx to build street-level geolocation pipelines that identify GPS coordinates from photos using CosPlace, ALIKED/DISK, and LightGlue locally.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - build a geolocation index
  - reverse geolocate from street view
  - local image geolocation pipeline
  - netryx geolocation
  - find location from photo without landmarks
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies GPS coordinates from street-level photos. It uses a three-stage computer vision pipeline (global retrieval → local geometric verification → refinement) to match any street photo against a pre-indexed database of Street View panoramas. Sub-50m accuracy, no landmarks required, runs entirely on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode (LoFTR dense matching)
pip install kornia
```

### Environment Variables

```bash
# Optional: only needed for AI Coarse blind geolocation via Gemini
export GEMINI_API_KEY="your_key_here"
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4GB     | 8GB+        |
| RAM       | 8GB     | 16GB+       |
| Storage   | 10GB    | 50GB+       |
| Python    | 3.9+    | 3.10+       |

GPU backends: CUDA (NVIDIA), MPS (Apple M1+), or CPU fallback.

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### 1. Create an Index

Index a geographic area before searching. This crawls Street View panoramas and extracts CosPlace fingerprints.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon
3. Set radius (0.5–10 km)
4. Set grid resolution (default: 300)
5. Click **Create Index**

**Indexing time estimates:**

| Radius  | ~Panoramas | Time (M2 Max) | Index Size |
|---------|------------|---------------|------------|
| 0.5 km  | ~500       | 30 min        | ~60 MB     |
| 1 km    | ~2,000     | 1–2 hours     | ~250 MB    |
| 5 km    | ~30,000    | 8–12 hours    | ~3 GB      |
| 10 km   | ~100,000   | 24–48 hours   | ~7 GB      |

Index saves incrementally — safe to interrupt and resume.

### 2. Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose **Manual** (enter coords + radius) or **AI Coarse** (Gemini auto-detects region)
4. Click **Run Search** → **Start Full Search**
5. Result appears on map with confidence score

---

## Programmatic Usage

### Load CosPlace and Extract a Descriptor

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = get_cosplace_model(device=device)

image = Image.open("query_photo.jpg").convert("RGB")
descriptor = get_descriptor(model, image, device=device)
# descriptor shape: (512,) — float32 numpy array
print(f"Descriptor shape: {descriptor.shape}")
```

### Search the Index Programmatically

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Returns distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * asin(sqrt(a))

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
headings = meta["headings"]  # (N,)
panoids = meta["panoids"]    # (N,)

# Query
center_lat, center_lon = 48.8566, 2.3522  # Paris
radius_m = 2000  # 2 km

# Radius filter
mask = np.array([
    haversine(center_lat, center_lon, lats[i], lons[i]) <= radius_m
    for i in range(len(lats))
])
filtered_desc = descriptors[mask]
filtered_indices = np.where(mask)[0]

# Cosine similarity search
query_desc = descriptor / np.linalg.norm(descriptor)
filtered_desc_norm = filtered_desc / np.linalg.norm(filtered_desc, axis=1, keepdims=True)
similarities = filtered_desc_norm @ query_desc

# Top 500 candidates
top_k = min(500, len(similarities))
top_local = np.argsort(similarities)[::-1][:top_k]
top_global = filtered_indices[top_local]

candidates = [
    {
        "panoid": panoids[i],
        "lat": lats[i],
        "lon": lons[i],
        "heading": headings[i],
        "similarity": similarities[top_local[rank]]
    }
    for rank, i in enumerate(top_global)
]

print(f"Top candidate: {candidates[0]}")
```

### Build Index from Parts (Large Datasets)

```bash
# Use the standalone high-performance index builder
python build_index.py
```

```python
# Or trigger from code after crawling parts into cosplace_parts/
import subprocess
subprocess.run(["python", "build_index.py"], check=True)
```

### Feature Matching with LightGlue (ALIKED on CUDA)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda")

# Load extractor and matcher
extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

# Load images
image0 = load_image("query_photo.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

# Extract features
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# Match
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]
print(f"Matched keypoints: {kpts0.shape[0]}")
```

### Feature Matching with DISK on MPS/CPU

```python
import torch
from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd

device = torch.device("mps")  # or "cpu"

extractor = DISK(max_num_keypoints=768).eval().to(device)
matcher = LightGlue(features="disk").eval().to(device)

image0 = load_image("query_photo.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
print(f"Matches: {matches01['matches'].shape[0]}")
```

### RANSAC Geometric Verification

```python
import cv2
import numpy as np

def ransac_verify(kpts0_np, kpts1_np, threshold=3.0):
    """
    Returns number of geometric inliers after RANSAC.
    kpts0_np, kpts1_np: (N, 2) float32 arrays of matched keypoint coords.
    """
    if len(kpts0_np) < 4:
        return 0
    _, inlier_mask = cv2.findFundamentalMat(
        kpts0_np, kpts1_np,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.99
    )
    if inlier_mask is None:
        return 0
    return int(inlier_mask.sum())

# Example usage after LightGlue matching
kpts0_np = kpts0.cpu().numpy()
kpts1_np = kpts1.cpu().numpy()
inliers = ransac_verify(kpts0_np, kpts1_np)
print(f"Inliers after RANSAC: {inliers}")
```

---

## Pipeline Stages Reference

### Stage 1 — Global Retrieval (CosPlace)
- Extracts 512-dim descriptor from query + horizontally flipped version
- Cosine similarity against all indexed panoramas
- Haversine radius filter limits to specified area
- Returns top 500–1000 candidates
- **Speed:** <1 second (single matrix multiply)

### Stage 2 — Local Geometric Verification
- Downloads panorama tiles from Street View (8 tiles, stitched)
- Crops at 3 FOVs: 70°, 90°, 110° at indexed heading
- Extracts ALIKED (CUDA) or DISK (MPS/CPU) keypoints
- LightGlue deep matching + RANSAC verification
- **Speed:** 2–5 minutes for 300–500 candidates

### Stage 3 — Refinement
- Heading refinement: ±45° at 15° steps for top 15 candidates
- Spatial consensus: clusters matches into 50m cells
- Confidence scoring: clustering density + uniqueness ratio

### Ultra Mode
- **LoFTR**: dense detector-free matching for blur/night images
- **Descriptor hopping**: re-searches index using clean matched panorama descriptor
- **Neighborhood expansion**: searches all panoramas within 100m of best match

---

## Index Structure

```
index/
├── cosplace_descriptors.npy   # (N, 512) float32 — all CosPlace descriptors
└── metadata.npz               # lats, lons, headings, panoids arrays

cosplace_parts/                # Raw chunks saved during indexing (auto-merged)
```

All areas share one unified index. Radius filter at search time isolates regions — no per-city index needed.

---

## Confidence Score Interpretation

| Score | Meaning |
|-------|---------|
| >0.85 | High confidence — reliable match |
| 0.65–0.85 | Medium confidence — likely correct |
| <0.65 | Low confidence — use Ultra Mode or expand radius |

---

## Common Patterns

### Multi-Area Index Strategy

```python
# Index multiple cities into the same index — no separation needed
# Paris:   center=(48.8566, 2.3522),  radius=5km
# London:  center=(51.5074, -0.1278), radius=5km
# All stored in the same cosplace_descriptors.npy

# At search time, radius filter handles isolation:
# search(query, center=(48.8566, 2.3522), radius_m=5000)  → only Paris results
# search(query, center=(51.5074, -0.1278), radius_m=5000) → only London results
```

### Choosing the Right Extractor

```python
import torch

def get_extractor_and_matcher(device: torch.device):
    from lightglue import LightGlue, ALIKED, DISK
    if device.type == "cuda":
        extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
        matcher = LightGlue(features="aliked").eval().to(device)
    else:
        # MPS and CPU: use DISK (ALIKED has MPS stability issues)
        extractor = DISK(max_num_keypoints=768).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)
    return extractor, matcher
```

### Flipped Descriptor for Reversed Perspectives

```python
from PIL import Image, ImageOps
import numpy as np

def get_combined_descriptor(model, image_path, device):
    from cosplace_utils import get_descriptor
    img = Image.open(image_path).convert("RGB")
    img_flipped = ImageOps.mirror(img)
    
    desc = get_descriptor(model, img, device=device)
    desc_flipped = get_descriptor(model, img_flipped, device=device)
    
    # Average both — improves recall for reversed camera angles
    combined = (desc + desc_flipped) / 2
    combined /= np.linalg.norm(combined)
    return combined
```

### Spatial Consensus Clustering

```python
from collections import defaultdict

def cluster_candidates(candidates, cell_size_m=50):
    """
    Groups candidates into geographic cells and returns
    the most populated cluster center as the best estimate.
    """
    def cell_key(lat, lon):
        # Approx meters per degree
        lat_cell = int(lat * 111320 / cell_size_m)
        lon_cell = int(lon * 111320 * abs(cos(radians(lat))) / cell_size_m)
        return (lat_cell, lon_cell)
    
    from math import cos, radians
    clusters = defaultdict(list)
    for c in candidates:
        key = cell_key(c["lat"], c["lon"])
        clusters[key].append(c)
    
    best_cluster = max(clusters.values(), key=len)
    lats = [c["lat"] for c in best_cluster]
    lons = [c["lon"] for c in best_cluster]
    return {
        "lat": sum(lats) / len(lats),
        "lon": sum(lons) / len(lons),
        "support": len(best_cluster)
    }
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11  # match your Python version
```

### CUDA out of memory
- Reduce `max_num_keypoints` in ALIKED: use `512` instead of `1024`
- Reduce candidate count from 500 to 200
- Enable `torch.cuda.empty_cache()` between candidates

### LightGlue import error
```bash
# Reinstall from source — PyPI version may be stale
pip uninstall lightglue -y
pip install git+https://github.com/cvg/LightGlue.git
```

### MPS errors on Apple Silicon
```bash
# ALIKED can be unstable on MPS — use DISK instead (handled automatically)
# If MPS crashes: fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Index search returns no results
- Verify `index/cosplace_descriptors.npy` and `index/metadata.npz` both exist
- Confirm search radius covers the indexed area
- Check that `cosplace_parts/` chunks were merged: run `python build_index.py`

### Low inlier count (<20) on all candidates
- Try **Ultra Mode** (enables LoFTR + descriptor hopping)
- Increase search radius — the correct location may not be indexed yet
- Verify query image is street-level (aerial/indoor photos won't match)

### Slow indexing
- Use `build_index.py` for large datasets (more efficient than GUI indexer)
- Indexing is I/O bound on Street View downloads — stable broadband helps
- Index saves incrementally; safe to pause and resume

---

## Project Files

| File | Purpose |
|------|---------|
| `test_super.py` | Main GUI application + full pipeline |
| `cosplace_utils.py` | CosPlace model loading + descriptor extraction |
| `build_index.py` | Standalone index builder for large datasets |
| `requirements.txt` | Python dependencies |
| `cosplace_parts/` | Raw embedding chunks (auto-created during indexing) |
| `index/` | Compiled searchable index (descriptors + metadata) |
```
