```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - netryx geolocation
  - index street view panoramas
  - locate where a photo was taken
  - visual place recognition pipeline
  - osint geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes Street View panoramas, then matches query images against those indexes using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and geometric verification (RANSAC). Sub-50m accuracy with no landmarks required. Runs entirely on local hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching
pip install kornia
```

**macOS tkinter fix** (if GUI renders blank):
```bash
brew install python-tk@3.11   # match your Python version
```

**Optional Gemini API key** (AI Coarse geolocation mode):
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.10+ |
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 50 GB+ |

GPU backends:
- **NVIDIA**: CUDA (uses ALIKED, 1024 keypoints)
- **Apple Silicon**: MPS (uses DISK, 768 keypoints)
- **CPU**: Supported but slow

---

## Launch GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles both **Create** (indexing) and **Search** modes.

---

## Core Workflow

### Step 1 — Index an Area (Create Mode)

Before searching you must index street panoramas for your target area. The index is stored incrementally and resumes on interruption.

**GUI steps:**
1. Select **Create** mode
2. Enter center lat/lon of target area
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

**Data flow:**
```
Grid points → Street View API → Panoramas → CosPlace crops
→ cosplace_parts/*.npz → cosplace_descriptors.npy + metadata.npz
```

For large-scale indexing, use the standalone builder:
```bash
python build_index.py
```

### Step 2 — Search (Search Mode)

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide center lat/lon + radius if region is known
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py            # Main application (GUI + all pipeline logic)
├── cosplace_utils.py        # CosPlace model loading + descriptor extraction
├── build_index.py           # High-performance standalone index builder
├── requirements.txt
├── cosplace_parts/          # Raw per-chunk embeddings (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim global descriptors
    └── metadata.npz               # lat, lon, heading, panoid per entry
```

---

## Pipeline Deep Dive

### Stage 1: Global Retrieval (CosPlace)

```python
# Conceptual usage of cosplace_utils.py
from cosplace_utils import load_cosplace_model, extract_descriptor
import numpy as np
from PIL import Image

model = load_cosplace_model(device="cuda")  # or "mps" or "cpu"

img = Image.open("query.jpg")
descriptor = extract_descriptor(model, img, device="cuda")
# descriptor shape: (512,) — normalized float32 vector

# Also extract flipped version to catch reversed perspectives
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = extract_descriptor(model, img_flipped, device="cuda")
```

Index search (cosine similarity):
```python
# descriptors.npy: shape (N, 512)
# metadata.npz: keys lat, lon, heading, panoid
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz")

query_vec = descriptor / np.linalg.norm(descriptor)
index_vecs = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)

scores = index_vecs @ query_vec          # cosine similarity, shape (N,)
top_idx = np.argsort(scores)[::-1][:500]  # top 500 candidates
```

Radius filter (haversine):
```python
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

center_lat, center_lon = 48.8566, 2.3522
radius_km = 2.0

lats = meta["lat"]
lons = meta["lon"]

mask = np.array([
    haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
    for i in range(len(lats))
])
filtered_idx = top_idx[mask[top_idx]]
```

### Stage 2: Local Feature Matching (ALIKED/DISK + LightGlue)

```python
# LightGlue with ALIKED (CUDA) — used internally by test_super.py
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

# Load and extract from query image
query_img = load_image("query.jpg").to(device)
query_feats = extractor.extract(query_img)

# Load and extract from a candidate panorama crop
candidate_img = load_image("candidate_crop.jpg").to(device)
candidate_feats = extractor.extract(candidate_img)

# Match
matches_data = matcher({"image0": query_feats, "image1": candidate_feats})
query_feats, candidate_feats, matches_data = [
    rbd(x) for x in [query_feats, candidate_feats, matches_data]
]

matches = matches_data["matches"]           # (M, 2) matched keypoint indices
scores = matches_data["matching_scores0"]   # confidence per match
```

RANSAC geometric verification:
```python
import cv2
import numpy as np

kp0 = query_feats["keypoints"][matches[:, 0]].cpu().numpy()
kp1 = candidate_feats["keypoints"][matches[:, 1]].cpu().numpy()

if len(kp0) >= 4:
    _, inlier_mask = cv2.findHomography(kp0, kp1, cv2.RANSAC, 5.0)
    num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
else:
    num_inliers = 0
# Higher num_inliers = better geometric match
```

Multi-FOV crops (70°, 90°, 110°) are tested per candidate to handle zoom mismatches:
```python
FOV_LIST = [70, 90, 110]
best_inliers = 0
best_fov = None

for fov in FOV_LIST:
    crop = extract_rectilinear_crop(panorama, heading, fov)
    inliers = match_and_verify(query_feats, crop, extractor, matcher)
    if inliers > best_inliers:
        best_inliers = inliers
        best_fov = fov
```

### Stage 3: Heading Refinement + Spatial Consensus

```python
# Heading refinement: test ±45° at 15° steps for top 15 candidates
HEADING_OFFSETS = range(-45, 46, 15)  # -45, -30, -15, 0, 15, 30, 45

for candidate in top_15_candidates:
    base_heading = candidate["heading"]
    for offset in HEADING_OFFSETS:
        test_heading = (base_heading + offset) % 360
        crop = extract_rectilinear_crop(panorama, test_heading, fov=90)
        inliers = match_and_verify(query_feats, crop, extractor, matcher)
        # Track best heading per candidate

# Spatial consensus: cluster matches into 50m cells
# Prefer clusters over lone high-inlier outliers
```

---

## Ultra Mode

Enable via the **Ultra Mode** checkbox in GUI. Adds three enhancements for difficult images (night, blur, low texture):

### 1. LoFTR Dense Matching (kornia)
```python
import kornia.feature as KF
import torch

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

img0 = load_as_tensor("query.jpg", device)         # (1,1,H,W) grayscale
img1 = load_as_tensor("candidate.jpg", device)

with torch.no_grad():
    correspondences = loftr({"image0": img0, "image1": img1})

kp0 = correspondences["keypoints0"]  # matched points in query
kp1 = correspondences["keypoints1"]  # matched points in candidate
conf = correspondences["confidence"]
```

### 2. Descriptor Hopping
```python
# If best match has < 50 inliers, extract CosPlace from the matched
# panorama (clean/high-quality) and re-search the index
if best_match_inliers < 50:
    matched_pano_img = download_panorama(best_match_panoid)
    new_descriptor = extract_descriptor(model, matched_pano_img, device)
    # Re-run index search with new_descriptor
```

### 3. Neighborhood Expansion
```python
# Search all panoramas within 100m of the best match
best_lat, best_lon = best_match["lat"], best_match["lon"]
neighbors = [
    i for i in range(len(lats))
    if haversine_km(best_lat, best_lon, lats[i], lons[i]) <= 0.1  # 100m
]
# Re-run Stage 2 on all neighbors
```

---

## Index Architecture

The index is global and coordinate-scoped — no per-city separation:

```python
# All regions share one index; radius filtering handles scoping
# Index Paris, London, and Tokyo into the same files:
#   cosplace_parts/*.npz (raw chunks)
#   index/cosplace_descriptors.npy
#   index/metadata.npz

# Search Paris only:
search(center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Search London only:
search(center_lat=51.5074, center_lon=-0.1278, radius_km=5.0)
```

---

## Common Patterns

### Pattern 1: Quick Test on a Small Area
```bash
# Index a 500m radius (fastest for testing, ~30 min)
# In GUI: Create mode → lat=48.8566, lon=2.3522, radius=0.5, grid=300
# Then: Search mode → upload photo → Manual → same coords, radius=1.0
```

### Pattern 2: Large City Indexing (Overnight)
```bash
# Use standalone builder for better performance on large datasets
python build_index.py
# Edit build_index.py to set center coords, radius, output path
# Run overnight; resumes automatically if interrupted
```

### Pattern 3: OSINT / Unknown Location
```bash
# Set GEMINI_API_KEY, use AI Coarse mode
# Gemini analyzes signs, architecture, vegetation → estimates region
# Netryx then searches that region automatically
export GEMINI_API_KEY="$GEMINI_API_KEY"
python test_super.py
# GUI: Search → AI Coarse → upload image → Run Search
```

### Pattern 4: Difficult Image (Night / Blur)
```bash
# Enable Ultra Mode checkbox before running search
# Pipeline adds LoFTR + descriptor hopping + neighborhood expansion
# Expect 2–3x longer runtime but significantly better recall
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| GUI renders blank | macOS bundled tkinter bug | `brew install python-tk@3.11` |
| CUDA out of memory | Too many keypoints | Reduce `max_num_keypoints` in extractor init |
| Index search returns 0 results | Radius too small or wrong coords | Increase radius; verify center lat/lon |
| Low inlier count (<20) everywhere | Query image too degraded | Enable Ultra Mode; try with a cleaner crop |
| Indexing stops mid-way | API rate limit or network | Just re-run — indexing resumes from last checkpoint |
| `kornia` not found in Ultra Mode | Not installed | `pip install kornia` |
| MPS device errors on Mac | PyTorch version mismatch | `pip install --upgrade torch torchvision` |

### Device Selection Logic
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    # Uses ALIKED (1024 keypoints)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # Uses DISK (768 keypoints) — ALIKED not fully supported on MPS
else:
    device = torch.device("cpu")
    # Uses DISK — slowest
```

### Confidence Score Interpretation
```
> 80 inliers + geographic clustering → High confidence (reliable result)
50–80 inliers                        → Medium confidence (likely correct)
20–50 inliers                        → Low confidence (verify manually)
< 20 inliers                         → Very low (try Ultra Mode or wider index)
```

---

## Models Reference

| Model | Role | Paper |
|-------|------|-------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim descriptor (retrieval) | CVPR 2022 |
| [ALIKED](https://github.com/naver/alike) | Local keypoints — CUDA | IEEE TIP 2023 |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints — MPS/CPU | NeurIPS 2020 |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | ICCV 2023 |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense matching, Ultra Mode | CVPR 2021 |
```
