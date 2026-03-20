```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for Netryx, a locally-hosted open-source street-level geolocation engine using CosPlace + ALIKED/DISK + LightGlue to identify GPS coordinates from street photos.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - build a geolocation index
  - run netryx search
  - osint image geolocation
  - where was this photo taken
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas into a searchable index, then uses a three-stage computer vision pipeline (global retrieval → geometric verification → refinement) to match a query image against the physical world — not the internet.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (deep feature matcher)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Optional: Gemini API key for AI Coarse location guessing

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

### 1. Create an Index (crawl + embed an area)

In the GUI:
- Select **Create** mode
- Enter center lat/lon of the area
- Set radius (km) and grid resolution (default: 300)
- Click **Create Index**

Index is saved incrementally to `cosplace_parts/` — safe to interrupt and resume.

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

### 2. Search for a photo's location

In the GUI:
- Select **Search** mode
- Upload a street-level image
- Choose **Manual** (provide center coords + radius) or **AI Coarse** (Gemini guesses region)
- Click **Run Search** → **Start Full Search**
- Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (auto-created)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace 512-dim descriptor (+ flipped variant)
    ▼
Index Search — cosine similarity, haversine radius filter
    │  Top 500–1000 candidates
    ▼
Download panoramas → rectilinear crops at 3 FOVs (70°, 90°, 110°)
    │
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├── LightGlue feature matching
    └── RANSAC geometric verification → inlier count
    ▼
Heading refinement (±45°, 15° steps, top 15 candidates)
    │
    ├── Spatial consensus clustering (50m cells)
    └── Confidence scoring (uniqueness ratio)
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Code Examples

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape == (512,)
print(f"Descriptor shape: {descriptor.shape}")
```

### Search the index manually (no GUI)

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
panoids = meta["panoids"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km=2.0, top_k=500):
    """
    Returns indices of top_k candidates within radius, sorted by cosine similarity.
    """
    # Haversine radius filter
    dists = np.array([haversine_km(center_lat, center_lon, la, lo)
                      for la, lo in zip(lats, lons)])
    mask = dists <= radius_km
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        return []

    # Cosine similarity
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    cands = descriptors[candidate_idx]
    norms = np.linalg.norm(cands, axis=1, keepdims=True) + 1e-8
    cands_norm = cands / norms
    sims = cands_norm @ q

    # Top-k
    top_local = np.argsort(-sims)[:top_k]
    return candidate_idx[top_local].tolist()

# Usage
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)
img = Image.open("query_photo.jpg").convert("RGB")
desc = extract_descriptor(model, img, device=device)

candidates = search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
print(f"Found {len(candidates)} candidates")
for idx in candidates[:5]:
    print(f"  panoid={panoids[idx]}  lat={lats[idx]:.6f}  lon={lons[idx]:.6f}")
```

### Build index from an existing list of panorama IDs

```python
# build_index.py is a standalone high-performance builder for large datasets
# Run it directly:
#   python build_index.py

# Or import for programmatic use:
import subprocess
result = subprocess.run(
    ["python", "build_index.py",
     "--parts-dir", "cosplace_parts",
     "--output-dir", "index"],
    check=True
)
```

### Ultra Mode — programmatic trigger

Ultra Mode is enabled via the GUI checkbox, but the key behaviors it enables:

```python
# Ultra Mode adds three mechanisms internally in test_super.py:
# 1. LoFTR dense matching (handles blur/low-texture)
# 2. Descriptor hopping (re-search using matched panorama's descriptor)
# 3. Neighborhood expansion (search panoramas within 100m of best match)

# To use LoFTR directly with kornia:
import torch
import kornia.feature as KF
from kornia.color import rgb_to_grayscale

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def match_loftr(img0_tensor, img1_tensor, device):
    """
    img0_tensor, img1_tensor: float32 tensors shape (1,1,H,W) grayscale, values 0-1
    Returns matched keypoints (mkpts0, mkpts1) and confidence scores.
    """
    with torch.no_grad():
        batch = {
            "image0": img0_tensor.to(device),
            "image1": img1_tensor.to(device),
        }
        correspondences = loftr(batch)
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    conf   = correspondences["confidence"].cpu().numpy()
    return mkpts0, mkpts1, conf
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_inliers(mkpts0, mkpts1, threshold=4.0):
    """
    Returns number of RANSAC inliers for a set of matched keypoints.
    threshold: reprojection error in pixels.
    """
    if len(mkpts0) < 8:
        return 0, None
    F, mask = cv2.findFundamentalMat(
        mkpts0.astype(np.float32),
        mkpts1.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999,
        maxIters=10000
    )
    if mask is None:
        return 0, None
    inliers = int(mask.ravel().sum())
    return inliers, mask
```

---

## Common Patterns

### Pattern: Multi-city unified index

```python
# You can index multiple cities into the same index files.
# The radius filter at search time handles scoping automatically.
# No city selection needed — just provide center + radius.

# Index Paris (run once)
# GUI: Create, center=(48.8566, 2.3522), radius=5km

# Index London (run once, appends to same index)
# GUI: Create, center=(51.5074, -0.1278), radius=5km

# Search only Paris:
candidates = search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Search only London:
candidates = search_index(desc, center_lat=51.5074, center_lon=-0.1278, radius_km=5.0)
```

### Pattern: Flip augmentation for retrieval

```python
# Netryx extracts descriptors from both the original and
# horizontally-flipped image to catch reversed perspectives.

from PIL import Image, ImageOps

def extract_with_flip(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_flip = ImageOps.mirror(img)

    desc_orig = extract_descriptor(model, img, device=device)
    desc_flip = extract_descriptor(model, img_flip, device=device)

    # Use the better-matching descriptor, or average both
    return desc_orig, desc_flip
```

### Pattern: Multi-FOV crop for zoom mismatch

```python
import numpy as np
from PIL import Image

def perspective_crop(panorama_img, heading_deg, fov_deg, output_size=(640, 480)):
    """
    Extract a rectilinear crop from an equirectangular panorama.
    heading_deg: 0=North, 90=East, 180=South, 270=West
    fov_deg: horizontal field of view (try 70, 90, 110)
    """
    # Netryx tests FOVs [70, 90, 110] to handle zoom mismatches
    # between query photo and indexed panorama view.
    # Implementation uses standard equirectangular→perspective projection.
    pass  # Full implementation is in test_super.py
```

---

## Hardware & Device Selection

```python
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        extractor_type = "ALIKED"   # 1024 keypoints
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        extractor_type = "DISK"     # 768 keypoints
    else:
        device = torch.device("cpu")
        extractor_type = "DISK"
    return device, extractor_type

device, extractor = get_device()
print(f"Using device: {device}, feature extractor: {extractor}")
```

---

## Configuration Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | 300 | Higher = denser panorama coverage. Don't change unless needed. |
| Top-K candidates | 500–1000 | Candidates passed to Stage 2 |
| RANSAC threshold | 4px | Reprojection error threshold |
| Heading refinement range | ±45° @ 15° steps | Applied to top 15 candidates |
| Spatial cluster cell size | 50m | For consensus clustering |
| Ultra neighborhood radius | 100m | Expansion around best match |
| FOV variants | 70°, 90°, 110° | Tested per candidate to handle zoom mismatch |

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # match your Python version
```

### CUDA out of memory
- Reduce top-K candidates (edit search params in GUI or source)
- Use DISK instead of ALIKED (automatically used on MPS/CPU)
- Enable Ultra Mode only when necessary

### Index search returns 0 candidates
```python
# Verify index is built and radius covers your area
import numpy as np
meta = np.load("index/metadata.npz", allow_pickle=True)
print(f"Index contains {len(meta['lats'])} panoramas")
print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
# If empty, re-run Create Index or check build_index.py
```

### LightGlue import error
```bash
# Ensure it was installed from GitHub, not PyPI
pip uninstall lightglue -y
pip install git+https://github.com/cvg/LightGlue.git
```

### Interrupted indexing / resume
The index builder saves incrementally to `cosplace_parts/*.npz`. Simply re-run Create Index with the same parameters — it will skip already-processed panoramas and continue.

### Low confidence scores / wrong results
1. Verify the search radius actually covers the area where the photo was taken
2. Enable **Ultra Mode** for blurry, night, or low-texture images
3. Increase grid resolution when creating the index for denser coverage
4. Check that the query image is genuinely street-level (not aerial, interior, etc.)

### LoFTR not available (Ultra Mode)
```bash
pip install kornia
# Verify:
python -c "import kornia.feature as KF; print(KF.LoFTR)"
```

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| CosPlace (512-dim) | Global descriptor / retrieval | All |
| ALIKED (1024 kp) | Local keypoints + descriptors | CUDA only |
| DISK (768 kp) | Local keypoints + descriptors | MPS / CPU |
| LightGlue | Deep feature matching | All |
| LoFTR | Dense detector-free matching (Ultra) | All (via kornia) |

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `test_super.py` | Full application: GUI, indexing, full search pipeline |
| `cosplace_utils.py` | `load_cosplace_model()`, `extract_descriptor()` |
| `build_index.py` | Standalone batch index builder for large areas |
| `index/cosplace_descriptors.npy` | All descriptor vectors (N×512 float32) |
| `index/metadata.npz` | `lats`, `lons`, `panoids`, `headings` arrays |
| `cosplace_parts/*.npz` | Raw per-chunk embeddings before index compilation |
```
