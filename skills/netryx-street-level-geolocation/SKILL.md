---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo locally
  - find GPS coordinates from street image
  - run Netryx geolocation
  - index street view panoramas
  - street level geolocation pipeline
  - use CosPlace LightGlue for location matching
  - open source GeoGuessr tool
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls and indexes street-view panoramas, extracts visual fingerprints with CosPlace, and verifies matches using ALIKED/DISK keypoints + LightGlue deep feature matching — all on your own hardware, no cloud required.

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

### Optional — Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### Platform GPU support

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU | CUDA | Uses ALIKED (1024 keypoints) |
| Apple Silicon | MPS | Uses DISK (768 keypoints) |
| CPU only | CPU | Works, significantly slower |

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix:** `brew install python-tk@3.11` (match your Python version)

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area before any search. The index stores 512-dim CosPlace descriptors for all crawled street-view panoramas.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon of target area
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Indexing time/size reference:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **incremental** — safe to interrupt and resume.

### Step 2 — Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose **Manual** (enter center coords + radius) or **AI Coarse** (Gemini guesses region)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone index builder (for large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (.npz), created during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (matrix)
    └── metadata.npz               # Coordinates, headings, panoid IDs
```

---

## Three-Stage Pipeline (How It Works)

### Stage 1 — Global Retrieval (CosPlace)

```python
# Conceptual: what happens internally in cosplace_utils.py
import torch
from cosplace_utils import get_cosplace_model, get_descriptor

model = get_cosplace_model()          # loads pretrained CosPlace (512-dim output)

descriptor = get_descriptor(model, image_tensor)           # query image fingerprint
descriptor_flipped = get_descriptor(model, flipped_tensor) # catches reversed perspectives

# Index search = cosine similarity matrix multiply (sub-second regardless of index size)
# + haversine radius filter to restrict to the search area
# → returns top 500–1000 candidate panoramas
```

### Stage 2 — Local Geometric Verification (ALIKED/DISK + LightGlue)

```python
# Conceptual: verification loop over candidates
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

# Feature extractor chosen by hardware
extractor = ALIKED(max_num_keypoints=1024).eval().cuda()   # NVIDIA
# extractor = DISK(max_num_keypoints=768).eval().to('mps') # Apple Silicon

matcher = LightGlue(features='aliked').eval().cuda()

# For each candidate panorama:
#   1. Download panorama from Street View (8 tiles, stitched)
#   2. Crop at indexed heading angle
#   3. Generate multi-FOV crops: 70°, 90°, 110°
#   4. Extract keypoints from each crop + query image
#   5. LightGlue match → RANSAC geometric verification
#   6. Count inliers — highest inlier count = best match

feats0 = extractor.extract(query_image)
feats1 = extractor.extract(candidate_crop)
matches01 = matcher({'image0': feats0, 'image1': feats1})
matches01 = rbd(matches01)  # remove batch dimension
inliers = matches01['matches'].shape[0]
```

### Stage 3 — Refinement

```python
# Internally performed after Stage 2:
# 1. Heading refinement: test ±45° at 15° steps, 3 FOVs, for top 15 candidates
# 2. Spatial consensus: cluster matches into 50m cells
#    → cluster of multiple candidates beats a single high-inlier outlier
# 3. Confidence scoring: geographic clustering + uniqueness ratio (best vs runner-up)
```

---

## Ultra Mode

Enable via GUI checkbox. Adds three enhancements for difficult images (night, blur, low texture):

| Enhancement | What It Does |
|-------------|-------------|
| **LoFTR** | Detector-free dense matching — no keypoints needed, handles blur |
| **Descriptor hopping** | Re-searches index using the matched panorama's clean descriptor |
| **Neighborhood expansion** | Searches all panoramas within 100m of best match |

```python
# Ultra Mode requires kornia (LoFTR)
import kornia
# Installed via: pip install kornia
```

---

## Index Architecture

The index is **unified and location-agnostic**. You can index multiple cities into the same index:

```
cosplace_parts/         ← incremental chunks written during indexing
    part_0000.npz
    part_0001.npz
    ...

index/
    cosplace_descriptors.npy   ← stacked matrix of all descriptors (N × 512)
    metadata.npz               ← lat, lon, heading, panoid for each row
```

**Search is radius-filtered by coordinates**, so one index handles multiple cities:

```
Index Paris 5km  ┐
Index London 5km ├─→ Single unified index
Index Tokyo 5km  ┘

Search center=(48.8566, 2.3522), radius=5km  → only Paris results
Search center=(51.5074, -0.1278), radius=5km → only London results
```

---

## Standalone Index Builder (Large Datasets)

For areas > 5km radius, use the high-performance builder directly:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --resolution 300
```

This writes incrementally to `cosplace_parts/` and can be stopped/resumed safely.

---

## Common Patterns

### Pattern: Full search from Python (scripting the pipeline)

The primary interface is the GUI (`test_super.py`), but you can reuse individual components:

```python
import numpy as np
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torchvision.transforms as T
import torch

# Load model
model = get_cosplace_model()
model.eval()

# Prepare image
transform = T.Compose([
    T.Resize((480, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
img = Image.open("query.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)

# Extract descriptor
with torch.no_grad():
    desc = get_descriptor(model, tensor)   # shape: (512,)

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]

# Cosine similarity search
desc_norm = desc / np.linalg.norm(desc)
db_norm = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
scores = db_norm @ desc_norm                       # (N,)

# Radius filter (haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

center_lat, center_lon = 48.8566, 2.3522
radius_m = 2000
distances = haversine(center_lat, center_lon, lats, lons)
mask = distances < radius_m

# Top candidates within radius
filtered_scores = np.where(mask, scores, -1)
top_indices = np.argsort(filtered_scores)[::-1][:500]

for idx in top_indices[:10]:
    print(f"lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, score={scores[idx]:.4f}")
```

### Pattern: LightGlue feature matching between two images

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features='aliked').eval().to(device)

image0 = load_image("query.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    result = matcher({'image0': feats0, 'image1': feats1})
    result = rbd(result)

matches = result['matches']        # (M, 2) matched keypoint indices
scores  = result['scores']         # (M,) match confidence scores
print(f"Matched keypoints: {matches.shape[0]}")

# RANSAC geometric verification (requires OpenCV)
import cv2
import numpy as np

kps0 = feats0['keypoints'][0].cpu().numpy()
kps1 = feats1['keypoints'][0].cpu().numpy()
m0   = matches[:, 0].cpu().numpy()
m1   = matches[:, 1].cpu().numpy()

pts0 = kps0[m0]
pts1 = kps1[m1]

if len(pts0) >= 4:
    _, inlier_mask = cv2.findFundamentalMat(
        pts0, pts1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"RANSAC inliers: {inliers}")
```

### Pattern: DISK on Apple Silicon (MPS)

```python
import torch
from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd

device = torch.device("mps")   # Apple Silicon

extractor = DISK(max_num_keypoints=768).eval().to(device)
matcher = LightGlue(features='disk').eval().to(device)

image0 = load_image("query.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    result  = rbd(matcher({'image0': feats0, 'image1': feats1}))

print(f"Matches: {result['matches'].shape[0]}")
```

### Pattern: Ultra Mode — LoFTR dense matching

```python
import torch
import kornia.feature as KF
from kornia.utils import image_to_tensor
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = KF.LoFTR(pretrained='outdoor').eval().to(device)

def load_gray_tensor(path, size=(480, 640)):
    img = Image.open(path).convert("L").resize((size[1], size[0]))
    t = image_to_tensor(np.array(img), keepdim=False).float() / 255.0
    return t.unsqueeze(0).to(device)    # (1, 1, H, W)

img0 = load_gray_tensor("query.jpg")
img1 = load_gray_tensor("candidate.jpg")

with torch.no_grad():
    out = loftr({'image0': img0, 'image1': img1})

kps0     = out['keypoints0'].cpu().numpy()   # (M, 2)
kps1     = out['keypoints1'].cpu().numpy()   # (M, 2)
conf     = out['confidence'].cpu().numpy()   # (M,)

# Filter by confidence
good = conf > 0.5
print(f"LoFTR high-confidence matches: {good.sum()}")
```

---

## Troubleshooting

### GUI appears blank on macOS

```bash
brew install python-tk@3.11   # match your Python version exactly
```

### LightGlue import error

```bash
# Must be installed from GitHub, not PyPI
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory during search

Reduce keypoints or process fewer candidates at once:
```python
# In extractor initialization, lower max_num_keypoints
extractor = ALIKED(max_num_keypoints=512).eval().cuda()
```

### Index build interrupted / incomplete

Safe to re-run — the builder reads existing `cosplace_parts/*.npz` files and skips already-indexed panoramas. Just re-run:
```bash
python build_index.py --lat LAT --lon LON --radius R --resolution 300
```

### No matches found / low inlier count

1. Verify the search area covers the photo's location (lat/lon + radius)
2. Try **Ultra Mode** for degraded/low-texture images
3. Increase search radius
4. Ensure the area was indexed (check `cosplace_parts/` is non-empty)
5. For night/blur images, LoFTR (Ultra Mode) significantly outperforms ALIKED/DISK

### MPS (Apple Silicon) errors with ALIKED

ALIKED runs on CUDA only. Netryx automatically falls back to DISK on MPS:
```python
# The app handles this internally — DISK is used on MPS/CPU automatically
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
extractor = DISK(max_num_keypoints=768).eval().to(device)
```

### Slow indexing

Use `build_index.py` directly (bypasses GUI overhead) and ensure GPU is active:
```python
import torch
print(torch.cuda.is_available())          # NVIDIA
print(torch.backends.mps.is_available())  # Apple Silicon
```

---

## Model Reference

| Model | Role | Source |
|-------|------|--------|
| CosPlace | Global 512-dim visual fingerprint | [github.com/gmberton/cosplace](https://github.com/gmberton/cosplace) |
| ALIKED | Local keypoints — CUDA | [github.com/naver/alike](https://github.com/naver/alike) |
| DISK | Local keypoints — MPS/CPU | [github.com/cvlab-epfl/disk](https://github.com/cvlab-epfl/disk) |
| LightGlue | Deep feature matcher | [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue) |
| LoFTR | Detector-free dense matcher (Ultra) | [kornia](https://kornia.readthedocs.io) |
