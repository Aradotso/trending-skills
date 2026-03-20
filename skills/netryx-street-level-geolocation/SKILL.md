```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - use netryx to locate a photo
  - index street view panoramas
  - identify location from street photo
  - open source geolocation pipeline
  - run netryx geolocation search
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the exact GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a local index of visual fingerprints, and then matches query images against that index using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and geometric refinement (RANSAC + spatial consensus).

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (deep feature matcher)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

GPU backends: CUDA (NVIDIA), MPS (Apple Silicon M1+), or CPU (slow).

### Optional: Gemini API Key (AI Coarse Mode)

```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix**: `brew install python-tk@3.11` (match your Python version)

---

## Core Workflow

### Step 1 — Create an Index

The index must be built before any search. It crawls street-view panoramas for a geographic area and stores CosPlace embeddings.

In the GUI:
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set search radius (start with `0.5`–`1` km for testing)
4. Set grid resolution (`300` recommended — do not change without reason)
5. Click **Create Index**

Indexing is resumable — if interrupted, re-run and it continues from where it left off.

**Indexing time estimates:**

| Radius  | ~Panoramas | Time (M2 Max) | Index Size |
|---------|------------|---------------|------------|
| 0.5 km  | ~500       | 30 min        | ~60 MB     |
| 1 km    | ~2,000     | 1–2 hours     | ~250 MB    |
| 5 km    | ~30,000    | 8–12 hours    | ~3 GB      |
| 10 km   | ~100,000   | 24–48 hours   | ~7 GB      |

### Step 2 — Search

In the GUI:
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes the image for region clues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result appears on the map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py          # Main application — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading & descriptor extraction
├── build_index.py         # Standalone index builder for large-scale datasets
├── requirements.txt
├── cosplace_parts/        # Raw per-chunk embeddings (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## The Three-Stage Pipeline

### Stage 1 — Global Retrieval (CosPlace)

Extracts a 512-dimensional visual fingerprint from the query image (and its horizontal flip), then performs cosine similarity search against the index, filtered by haversine radius.

```python
# cosplace_utils.py usage pattern
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import numpy as np

model = load_cosplace_model(device="cuda")  # or "mps" / "cpu"

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device="cuda")
# descriptor.shape == (512,)

# Also extract flipped version for robustness
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flip = extract_descriptor(model, flipped, device="cuda")
```

### Stage 2 — Local Feature Matching (ALIKED/DISK + LightGlue)

For each top candidate, downloads the panorama, generates multi-FOV crops (70°, 90°, 110°), extracts keypoints, and runs LightGlue + RANSAC geometric verification.

```python
# Feature extractor selection by device (handled internally by test_super.py)
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

if device == "cuda":
    # ALIKED — 1024 keypoints, best accuracy
    from lightglue import ALIKED
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    # DISK — 768 keypoints, MPS/CPU compatible
    from lightglue import DISK
    extractor = DISK(max_num_keypoints=768).eval().to(device)

from lightglue import LightGlue
matcher = LightGlue(features="aliked" if device == "cuda" else "disk").eval().to(device)
```

### Stage 3 — Refinement

- **Heading refinement**: Tests ±45° offsets at 15° steps across 3 FOVs for the top 15 candidates
- **Spatial consensus**: Clusters matches into 50m cells; prefers clusters over isolated high-inlier outliers
- **Confidence scoring**: Evaluates geographic clustering and uniqueness ratio (best vs. runner-up at different location)

---

## Ultra Mode

Enable via the **Ultra Mode** checkbox in the GUI for difficult images (night, motion blur, low texture).

Adds three enhancements:
1. **LoFTR** — detector-free dense matcher (handles blur/low-contrast where keypoint detectors fail)
2. **Descriptor hopping** — if best match has <50 inliers, extracts CosPlace from the matched panorama and re-searches
3. **Neighborhood expansion** — searches all panoramas within 100m of the best match

```python
# Ultra Mode LoFTR usage (requires kornia)
import kornia.feature as KF
import torch

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

# Input: grayscale tensors, shape (1, 1, H, W), normalized to [0, 1]
with torch.no_grad():
    output = loftr({
        "image0": query_gray_tensor,
        "image1": candidate_gray_tensor
    })

keypoints0 = output["keypoints0"]   # (N, 2)
keypoints1 = output["keypoints1"]   # (N, 2)
confidence  = output["confidence"]  # (N,)
```

---

## Index Architecture

All embeddings live in a **single unified index** — multiple cities can coexist. The radius filter at search time handles isolation.

```
# Data flow

# Indexing
Grid points → Street View API → Panoramas → CosPlace → cosplace_parts/*.npz

# Auto-build (triggered after indexing)
cosplace_parts/*.npz → index/cosplace_descriptors.npy + index/metadata.npz

# Search
Query image → CosPlace → cosine similarity (radius-filtered) →
Top 500 candidates → Download panoramas → ALIKED/DISK + LightGlue →
RANSAC → Heading refinement → Spatial consensus → GPS result
```

**Searching a specific city from a multi-city index:**
- Paris search: `center=(48.8566, 2.3522), radius=5km`
- London search: `center=(51.5074, -0.1278), radius=10km`

No city selection needed — coordinates + radius handle isolation automatically.

---

## Build Index Standalone (Large Datasets)

For large areas, use `build_index.py` directly instead of the GUI:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --resolution 300
```

This is optimized for long-running indexing jobs and writes to `cosplace_parts/` incrementally.

---

## Working Code Examples

### Extract a CosPlace Descriptor from Any Image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = load_cosplace_model(device=device)

def get_descriptor(image_path: str) -> "np.ndarray":
    img = Image.open(image_path).convert("RGB")
    desc = extract_descriptor(model, img, device=device)
    return desc  # shape: (512,)

desc = get_descriptor("street_photo.jpg")
print(f"Descriptor shape: {desc.shape}")  # (512,)
```

### Cosine Similarity Search Against the Index

```python
import numpy as np

# Load pre-built index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]    # (N,)
lons = meta["lons"]    # (N,)
headings = meta["headings"]  # (N,)
panoids  = meta["panoids"]   # (N,)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def search_index(query_desc, center_lat, center_lon, radius_km=2.0, top_k=500):
    # Radius filter
    dists = haversine_km(center_lat, center_lon, lats, lons)
    mask = dists <= radius_km
    if mask.sum() == 0:
        return []

    local_descs = descriptors[mask]       # (M, 512)
    local_indices = np.where(mask)[0]

    # Cosine similarity (descriptors assumed L2-normalized)
    sims = local_descs @ query_desc       # (M,)
    order = np.argsort(-sims)[:top_k]

    results = []
    for rank, idx in enumerate(order):
        global_idx = local_indices[idx]
        results.append({
            "rank": rank,
            "similarity": float(sims[idx]),
            "lat": float(lats[global_idx]),
            "lon": float(lons[global_idx]),
            "heading": float(headings[global_idx]),
            "panoid": str(panoids[global_idx]),
        })
    return results

# Usage
query_desc = get_descriptor("query.jpg")
candidates = search_index(query_desc, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
print(f"Top match: {candidates[0]}")
```

### Run LightGlue Matching Between Two Images

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="aliked").eval().to(device)

# load_image returns (3, H, W) tensor normalized to [0, 1]
image0 = load_image("query.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches_data = matcher({"image0": feats0, "image1": feats1})

# Remove batch dimension
feats0, feats1, matches_data = rbd(feats0), rbd(feats1), rbd(matches_data)

matched_kps0 = feats0["keypoints"][matches_data["matches"][..., 0]]  # (M, 2)
matched_kps1 = feats1["keypoints"][matches_data["matches"][..., 1]]  # (M, 2)
print(f"Matched keypoints: {matched_kps0.shape[0]}")
```

### RANSAC Geometric Verification

```python
import cv2
import numpy as np

def ransac_inliers(pts0: np.ndarray, pts1: np.ndarray, threshold: float = 4.0) -> int:
    """Returns number of RANSAC inliers. Higher = stronger geometric match."""
    if len(pts0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        pts0.astype(np.float32),
        pts1.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999,
    )
    if mask is None:
        return 0
    return int(mask.ravel().sum())

# After LightGlue matching
pts0 = matched_kps0.cpu().numpy()
pts1 = matched_kps1.cpu().numpy()
inliers = ransac_inliers(pts0, pts1)
print(f"Verified inliers: {inliers}")
# >100 inliers = strong match; <30 = weak/false match
```

---

## Models Reference

| Model    | Role                          | Backend       | Paper         |
|----------|-------------------------------|---------------|---------------|
| CosPlace | Global visual place recognition | All devices | CVPR 2022     |
| ALIKED   | Local keypoint extraction     | CUDA only     | IEEE TIP 2023 |
| DISK     | Local keypoint extraction     | MPS / CPU     | NeurIPS 2020  |
| LightGlue| Deep feature matching         | All devices   | ICCV 2023     |
| LoFTR    | Dense detector-free matching  | Ultra Mode    | CVPR 2021     |

---

## Common Patterns

### Multi-City Index Strategy

```python
# Index multiple cities — all share one index file
# Paris
python test_super.py  # Create, lat=48.8566, lon=2.3522, radius=5

# Tokyo (adds to existing index)
python test_super.py  # Create, lat=35.6762, lon=139.6503, radius=5

# Search Paris only
search_index(query_desc, center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Search Tokyo only
search_index(query_desc, center_lat=35.6762, center_lon=139.6503, radius_km=5.0)
```

### Choosing Search Radius

```python
# Known city, unknown district → 5–10 km
# Known district              → 1–2 km
# Unknown location (AI Coarse mode first) → Gemini estimates region, then 10–20 km
# Completely blind            → Use AI Coarse mode
```

### Confidence Interpretation

- **>100 RANSAC inliers** — High confidence, likely correct
- **50–100 inliers** — Medium confidence, check visually
- **<50 inliers** — Low confidence — enable Ultra Mode or expand radius
- **Spatial consensus** — Multiple candidates clustering at same location increases reliability

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # Match your Python version
```

### CUDA out of memory
Reduce `max_num_keypoints` in the extractor:
```python
extractor = ALIKED(max_num_keypoints=512).eval().to(device)  # default 1024
```

### LightGlue import error
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — must be installed from GitHub
```

### LoFTR not available (Ultra Mode disabled)
```bash
pip install kornia
# Verify:
python -c "import kornia.feature as KF; KF.LoFTR(pretrained='outdoor')"
```

### Index search returns 0 candidates
- Verify the indexed area overlaps with your search coordinates + radius
- Check `cosplace_parts/` is non-empty and `index/` was built
- Increase `radius_km` in the search

### Poor match quality (few inliers)
1. Enable **Ultra Mode** in the GUI
2. Check that your query image is street-level (not aerial or indoor)
3. Verify the area is indexed — add more panoramas with a tighter grid resolution
4. Try the flipped descriptor variant (the pipeline does this automatically)

### Indexing stalls or crashes
The indexer is resumable — re-run with the same parameters. It skips already-processed panoramas stored in `cosplace_parts/`.

### MPS (Apple Silicon) errors with ALIKED
ALIKED is CUDA-only. On MPS, the pipeline automatically falls back to DISK — this is expected behavior.

```python
# Verify device selection
import torch
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
# MPS → DISK extractor will be used automatically
```
```
