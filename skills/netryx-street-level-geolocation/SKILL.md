```markdown
---
name: netryx-street-level-geolocation
description: Expert knowledge for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - run netryx geolocation pipeline
  - osint image geolocation locally
  - find where a photo was taken
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls street-view panoramas, builds a searchable index of visual fingerprints, then matches query images against that index using a three-stage computer vision pipeline — all on your own hardware, no external SaaS required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (deep feature matcher)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### GPU Support

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU | CUDA | ALIKED extractor, fastest |
| Apple Silicon (M1+) | MPS | DISK extractor, fast |
| CPU only | — | Works, significantly slower |

Minimum 4GB VRAM, 8GB RAM. 16GB RAM recommended for large indexes.

---

## Optional: Gemini API Key (AI Coarse Mode)

AI Coarse mode uses Gemini to guess a region from visual clues (signs, architecture) when you have no prior knowledge of where a photo was taken.

```bash
export GEMINI_API_KEY="your_key_here"
```

Not required for manual searches where you already know the approximate region.

---

## Launching the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix:** `brew install python-tk@3.11` (match your Python version)

The GUI has two modes: **Create** (build an index) and **Search** (geolocate a photo).

---

## Core Workflow

### Step 1 — Build an Index

An index is a database of CosPlace visual fingerprints for every street-view panorama in a geographic area. Build it once, search it many times.

**In GUI:**
1. Select **Create** mode
2. Enter center lat/lon of the area
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Index build time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk Size |
|--------|-----------|---------------|-----------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Indexing is **resumable** — safe to interrupt and restart.

**For large areas, use the standalone high-performance builder:**

```bash
python build_index.py
```

### Step 2 — Search (Geolocate an Image)

**In GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Enter known center coordinates + radius
   - **AI Coarse**: Let Gemini estimate the region automatically
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Multi-Area Index Strategy

All indexed areas share one unified index. The radius filter at search time scopes results geographically — no city selection UI needed.

```
Index Paris (48.8566, 2.3522, r=5km)
Index London (51.5074, -0.1278, r=5km)
Index Tel Aviv (32.0853, 34.7818, r=5km)

Search: center=Paris coords, radius=5km → only Paris results
Search: center=London coords, radius=10km → only London results
```

---

## Pipeline Deep Dive

### Stage 1 — Global Retrieval (CosPlace)

```
Query image
  → CosPlace 512-dim descriptor
  → Also extract descriptor of horizontally-flipped image
  → Cosine similarity vs. entire index (single matrix multiply, <1s)
  → Haversine radius filter
  → Top 500–1000 candidates
```

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

```
For each candidate:
  → Download Google Street View panorama (8 tiles, stitched)
  → Crop at indexed heading angle
  → Generate 3 FOV crops: 70°, 90°, 110°
  → Extract keypoints with ALIKED (CUDA) or DISK (MPS/CPU)
  → LightGlue deep feature matching vs. query keypoints
  → RANSAC geometric verification (filter false matches)
  
Best candidate = highest verified inlier count
~2–5 min for 300–500 candidates
```

### Stage 3 — Refinement

```
Top 15 candidates:
  → Heading refinement: test ±45° at 15° steps × 3 FOVs
  → Spatial consensus: cluster matches into 50m cells
  → Confidence scoring: clustering density + uniqueness ratio
  
Output: GPS coordinates + confidence score
```

### Ultra Mode

Enable for difficult images (night, blur, low texture, rain):

```
Additional steps:
  → LoFTR: detector-free dense matching (handles blur/low-contrast)
  → Descriptor hopping: if best match <50 inliers, re-search index
    using CosPlace descriptor of the matched panorama (clean image)
  → Neighborhood expansion: search all panoramas within 100m of best match
```

Enable via the **Ultra Mode** checkbox in the GUI before running search.

---

## Project Structure

```
netryx/
├── test_super.py           # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py       # CosPlace model loader + descriptor extraction
├── build_index.py          # High-performance standalone index builder
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks (.npz), written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim visual fingerprints
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model, device = load_cosplace_model()

# Extract 512-dim fingerprint from any street image
img = Image.open("street_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device)  # shape: (512,)
print(f"Descriptor shape: {descriptor.shape}")
```

### Search the Index Programmatically

```python
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]
panoids = meta["panoids"]
headings = meta["headings"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """Radius-filtered cosine similarity search."""
    # Normalize descriptors
    q = query_descriptor / np.linalg.norm(query_descriptor)
    d = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)

    # Cosine similarity (matrix multiply)
    scores = d @ q  # shape: (N,)

    # Haversine radius filter
    mask = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
        for i in range(len(lats))
    ])

    filtered_scores = np.where(mask, scores, -1.0)
    top_indices = np.argsort(filtered_scores)[::-1][:top_k]

    return [
        {
            "idx": int(i),
            "score": float(filtered_scores[i]),
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "panoid": str(panoids[i]),
            "heading": float(headings[i]),
        }
        for i in top_indices if filtered_scores[i] > 0
    ]

# Usage
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

model, device = load_cosplace_model()
img = Image.open("mystery_photo.jpg").convert("RGB")
desc = extract_descriptor(model, img, device)

# Search Paris area
candidates = search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=3.0)
print(f"Top candidate: {candidates[0]}")
```

### Run LightGlue Matching Against a Candidate

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

# Choose extractor based on device
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def match_images(query_path, candidate_path):
    """Returns number of RANSAC-verified inliers."""
    img0 = load_image(query_path).to(device)
    img1 = load_image(candidate_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
    kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]

    if len(kpts0) < 8:
        return 0, None

    # RANSAC geometric verification
    import cv2
    pts0 = kpts0.cpu().numpy()
    pts1 = kpts1.cpu().numpy()
    _, inlier_mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.99)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return inliers, matches01

inliers, _ = match_images("query.jpg", "candidate_crop.jpg")
print(f"Verified inliers: {inliers}")
# >30 inliers = likely match, >80 inliers = strong match
```

### Using Ultra Mode with LoFTR

```python
import kornia
import torch
from kornia.feature import LoFTR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = LoFTR(pretrained="outdoor").eval().to(device)

def loftr_match(img0_tensor, img1_tensor):
    """Dense matching for difficult images (blur, night, low texture)."""
    # LoFTR expects grayscale, resized to multiples of 8
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    def prep(t):
        # Convert to grayscale, normalize to [0,1]
        gray = TF.rgb_to_grayscale(t).float() / 255.0
        # Resize to nearest multiple of 8
        h, w = gray.shape[-2:]
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        return F.interpolate(gray.unsqueeze(0), size=(h8, w8))

    input_dict = {
        "image0": prep(img0_tensor).to(device),
        "image1": prep(img1_tensor).to(device),
    }
    with torch.no_grad():
        correspondences = loftr(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Filter by confidence
    mask = confidence > 0.5
    return mkpts0[mask], mkpts1[mask]
```

---

## Configuration Reference

All configuration is done via the GUI or by editing `test_super.py` directly. Key tunable parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| Grid resolution | 300 | Panorama density during indexing — don't lower |
| Top-K candidates | 500 | Candidates passed to Stage 2 — increase for large areas |
| ALIKED keypoints | 1024 | CUDA only — more = slower but more matches |
| DISK keypoints | 768 | MPS/CPU — trade-off speed vs. recall |
| Heading offset range | ±45° at 15° steps | Stage 3 refinement sweep |
| Spatial cluster radius | 50m | Consensus clustering cell size |
| LoFTR confidence threshold | 0.5 | Ultra Mode: filter dense matches |
| Neighborhood expansion | 100m | Ultra Mode: re-search around best match |

---

## Common Patterns

### Pattern: Index a City District, Search Incrementally

```python
# Index multiple districts into the same unified index
# Run each separately (or sequentially) — index accumulates

# District 1: Marais, Paris
# GUI: center=(48.8566, 2.3522), radius=1km

# District 2: Montmartre
# GUI: center=(48.8867, 2.3431), radius=1km

# Search covers both automatically when radius is large enough
# GUI: center=(48.8700, 2.3480), radius=3km → finds results in both districts
```

### Pattern: Batch Geolocation (No GUI)

```python
import subprocess
import os

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
center = (48.8566, 2.3522)
radius_km = 2.0

# Netryx is primarily GUI-driven; for batch use, call the pipeline
# functions from test_super.py directly after importing
# (examine test_super.py for the exact function signatures)

# Minimal integration example:
# from test_super import run_search_pipeline
# for img in images:
#     result = run_search_pipeline(img, center[0], center[1], radius_km)
#     print(f"{img}: {result['lat']}, {result['lon']} (confidence: {result['confidence']})")
```

### Pattern: Confidence Score Interpretation

```
< 0.3  → Low confidence — result may be wrong, enable Ultra Mode
0.3–0.6 → Moderate — likely correct neighborhood, verify manually  
0.6–0.8 → Good — typically correct street
> 0.8  → High confidence — sub-50m accuracy expected
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # Match your Python version exactly
```

### `ImportError: No module named 'lightglue'`
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Note: install from GitHub, not PyPI
```

### `ImportError: No module named 'kornia'` (Ultra Mode)
```bash
pip install kornia
# Ultra Mode only — not required for standard pipeline
```

### CUDA out of memory
- Reduce ALIKED `max_num_keypoints` from 1024 to 512
- Reduce top-K candidates from 500 to 200
- Ensure no other GPU processes are running

### Index search returns zero results
- Confirm index was built (check `index/cosplace_descriptors.npy` exists)
- Verify search radius overlaps with indexed area coordinates
- Run `python build_index.py` to rebuild/merge `cosplace_parts/` into the index

### Indexing stalled / interrupted
Safe to restart — indexing resumes from last saved chunk in `cosplace_parts/`. Do not delete `cosplace_parts/` unless starting fresh.

### Low inlier counts (<20) on all candidates
- Enable **Ultra Mode** (LoFTR + descriptor hopping)
- Verify image is truly street-level (not aerial, not heavily cropped)
- Expand search radius — indexed area may not include the photo's location
- Check image isn't mirrored (the pipeline tries flipped descriptors but double-check)

### MPS (Apple Silicon) errors
```bash
# Ensure PyTorch MPS build is installed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or use the standard pip install — recent PyTorch includes MPS support
```

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim visual fingerprint | All (inference only) |
| [ALIKED](https://github.com/naver/alike) | Local keypoint extraction | CUDA preferred |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoint extraction | MPS/CPU fallback |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense matching (Ultra Mode) | CUDA/CPU via kornia |

Models are downloaded automatically on first use via PyTorch Hub / HuggingFace.
```
