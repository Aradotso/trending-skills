---
name: netryx-street-level-geolocation
description: Use Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from any street photo using CosPlace, ALIKED/DISK, and LightGlue computer vision pipelines.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - run netryx geolocation
  - build a street view index
  - OSINT geolocation from photo
  - netryx search and index
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that takes any street-level photograph and returns precise GPS coordinates (sub-50m accuracy). It crawls street-view panoramas, builds a searchable visual index using CosPlace embeddings, then verifies matches with ALIKED/DISK keypoint extraction and LightGlue deep feature matching — all on your own hardware, no cloud API needed.

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

# Optional: LoFTR dense matcher for Ultra Mode
pip install kornia
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11      # match your Python version
```

### Gemini API key (optional — AI Coarse mode only)
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles both **Create** (indexing) and **Search** modes.

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large areas)
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Workflow Overview

### Step 1 — Build an Index (Create Mode)

Index an area before you can search it. The indexer crawls Street View panoramas on a grid, extracts CosPlace fingerprints, and saves them to `cosplace_parts/`.

**In the GUI:**
1. Select **Create** mode
2. Enter center `lat, lon`
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Indexing is incremental — safe to interrupt and resume.

**Radius → time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 h | ~250 MB |
| 5 km | ~30,000 | 8–12 h | ~3 GB |
| 10 km | ~100,000 | 24–48 h | ~7 GB |

For large areas, use the standalone builder:
```bash
python build_index.py
```

---

### Step 2 — Search

**In the GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual** — enter approximate `lat, lon` + radius if you know the region
   - **AI Coarse** — let Gemini estimate region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score displayed on map

Enable **Ultra Mode** checkbox for degraded images (night, blur, low texture). Slower but activates LoFTR, descriptor hopping, and neighborhood expansion.

---

## Three-Stage Pipeline (What's Happening Internally)

### Stage 1 — Global Retrieval via CosPlace
```python
# cosplace_utils.py usage pattern
from cosplace_utils import get_cosplace_model, get_descriptor

model = get_cosplace_model()                  # loads CosPlace (512-dim output)
descriptor = get_descriptor(model, image)     # np.ndarray shape (512,)
flipped_desc = get_descriptor(model, image_flipped)

# Index search: cosine similarity against all stored descriptors
# cosplace_descriptors.npy shape: (N, 512)
import numpy as np
index_descs = np.load("index/cosplace_descriptors.npy")   # (N, 512)
scores = index_descs @ descriptor                          # dot product = cosine sim (if normalized)
top_k = np.argsort(scores)[::-1][:500]                    # top-500 candidates
```

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads panorama tiles from Street View, stitches them
- Crops rectilinear views at 3 FOVs: 70°, 90°, 110°
- Extracts keypoints with ALIKED (CUDA) or DISK (MPS/CPU)
- Matches keypoints with LightGlue, filters with RANSAC
- Candidate with most RANSAC inliers = best match

### Stage 3 — Refinement
- Heading refinement: tests ±45° at 15° steps for top-15 candidates
- Spatial consensus: clusters matches into 50m cells, prefers clusters over outliers
- Confidence score: evaluates geographic clustering + uniqueness ratio vs runner-up

---

## Code Examples

### Extract a CosPlace descriptor from any image

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import numpy as np

model = get_cosplace_model()

img = Image.open("street_photo.jpg").convert("RGB")
descriptor = get_descriptor(model, img)     # np.ndarray (512,)
print("Descriptor shape:", descriptor.shape)
print("Norm:", np.linalg.norm(descriptor))  # should be ~1.0 (normalized)
```

### Load the index and run a similarity search

```python
import numpy as np

# Load compiled index
descs = np.load("index/cosplace_descriptors.npy")       # (N, 512)
meta  = np.load("index/metadata.npz", allow_pickle=True)

lats     = meta["lats"]       # (N,) float64
lons     = meta["lons"]       # (N,) float64
headings = meta["headings"]   # (N,) float64
panoids  = meta["panoids"]    # (N,) str

# Cosine similarity search
query_desc = get_descriptor(model, img)                  # (512,)
scores = descs @ query_desc                              # (N,)

# Haversine radius filter (search within 5 km of Paris)
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

center_lat, center_lon = 48.8566, 2.3522
radius_m = 5000

mask = np.array([
    haversine(center_lat, center_lon, lats[i], lons[i]) <= radius_m
    for i in range(len(lats))
])

filtered_scores = np.where(mask, scores, -1)
top_k_idx = np.argsort(filtered_scores)[::-1][:500]

for idx in top_k_idx[:5]:
    print(f"  panoid={panoids[idx]}  lat={lats[idx]:.6f}  lon={lons[idx]:.6f}  score={scores[idx]:.4f}")
```

### Feature matching with LightGlue (ALIKED, CUDA)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="aliked").eval().to(device)

query_img     = load_image("query.jpg").to(device)
candidate_img = load_image("candidate_crop.jpg").to(device)

feats0 = extractor.extract(query_img)
feats1 = extractor.extract(candidate_img)

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

kpts0 = feats0["keypoints"][matches01["matches"][:, 0]]
kpts1 = feats1["keypoints"][matches01["matches"][:, 1]]
print(f"Matched keypoints: {len(kpts0)}")
```

### Feature matching with DISK (MPS / CPU fallback)

```python
import torch
from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

extractor = DISK(max_num_keypoints=768).eval().to(device)
matcher   = LightGlue(features="disk").eval().to(device)

feats0 = extractor.extract(load_image("query.jpg").to(device))
feats1 = extractor.extract(load_image("candidate.jpg").to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
matches01 = rbd(matches01)
print(f"Matches before RANSAC: {matches01['matches'].shape[0]}")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_inliers(kpts0_np, kpts1_np, threshold=4.0):
    """Returns number of RANSAC inliers for a candidate match."""
    if len(kpts0_np) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        kpts0_np.astype(np.float32),
        kpts1_np.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.99
    )
    if mask is None:
        return 0
    return int(mask.sum())

inliers = ransac_inliers(kpts0.cpu().numpy(), kpts1.cpu().numpy())
print(f"RANSAC inliers: {inliers}")
```

### Multi-FOV crop generation

```python
from PIL import Image
import numpy as np

def rectilinear_crop(panorama: Image.Image, heading_deg: float, fov_deg: float,
                     out_w: int = 640, out_h: int = 480) -> Image.Image:
    """
    Extract a rectilinear crop from an equirectangular panorama.
    heading_deg: 0=North, 90=East, 180=South, 270=West
    fov_deg: horizontal field of view in degrees
    """
    pano_w, pano_h = panorama.size
    pano_np = np.array(panorama)

    f = (out_w / 2) / np.tan(np.radians(fov_deg / 2))
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))
    xc = xs - out_w / 2
    yc = ys - out_h / 2

    # Ray directions in camera space
    ray_x = xc / f
    ray_y = yc / f
    ray_z = np.ones_like(ray_x)

    # Rotate by heading
    h_rad = np.radians(heading_deg)
    rot_x = ray_x * np.cos(h_rad) - ray_z * np.sin(h_rad)
    rot_z = ray_x * np.sin(h_rad) + ray_z * np.cos(h_rad)

    lon = np.arctan2(rot_x, rot_z)
    lat = np.arctan2(ray_y, np.sqrt(rot_x**2 + rot_z**2))

    px = ((lon / (2 * np.pi)) % 1.0) * pano_w
    py = (0.5 - lat / np.pi) * pano_h

    from scipy.ndimage import map_coordinates
    crop = np.stack([
        map_coordinates(pano_np[:, :, c], [py.ravel(), px.ravel()],
                        order=1, mode='wrap').reshape(out_h, out_w)
        for c in range(3)
    ], axis=-1)
    return Image.fromarray(crop.astype(np.uint8))

# Usage: generate 3 FOVs for one candidate
for fov in [70, 90, 110]:
    crop = rectilinear_crop(panorama, heading_deg=135.0, fov_deg=fov)
    crop.save(f"crop_fov{fov}.jpg")
```

---

## Common Patterns

### Pattern: Full manual search pipeline in code

```python
# 1. Load index
import numpy as np
descs = np.load("index/cosplace_descriptors.npy")
meta  = np.load("index/metadata.npz", allow_pickle=True)

# 2. Extract query descriptor
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image

model = get_cosplace_model()
query = Image.open("mystery_street.jpg").convert("RGB")
q_desc = get_descriptor(model, query)

# 3. Top-500 retrieval within 3km of a known city center
scores = descs @ q_desc
# ... apply haversine mask (see above) ...
top_k = np.argsort(filtered_scores)[::-1][:500]

# 4. For each candidate: download pano, crop at 3 FOVs, match, RANSAC
best_inliers = 0
best_idx = None
for idx in top_k:
    panoid  = meta["panoids"][idx]
    heading = meta["headings"][idx]
    # download pano tiles from Street View, stitch → `panorama`
    for fov in [70, 90, 110]:
        crop = rectilinear_crop(panorama, heading, fov)
        # ... extract keypoints, match with LightGlue, RANSAC ...
        if inliers > best_inliers:
            best_inliers = inliers
            best_idx = idx

print(f"Best match: lat={meta['lats'][best_idx]:.6f}, lon={meta['lons'][best_idx]:.6f}")
print(f"Inliers: {best_inliers}")
```

### Pattern: Checking GPU/device availability

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    feature_extractor = "aliked"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    feature_extractor = "disk"       # ALIKED has MPS issues
else:
    device = torch.device("cpu")
    feature_extractor = "disk"

print(f"Using device: {device}, extractor: {feature_extractor}")
```

### Pattern: Ultra Mode — LoFTR dense matching

```python
import kornia.feature as KF
import torch

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_match(img0: torch.Tensor, img1: torch.Tensor):
    """img0, img1: (1, 1, H, W) grayscale tensors, normalized 0–1"""
    with torch.no_grad():
        batch = {"image0": img0, "image1": img1}
        correspondences = loftr(batch)
    kpts0 = correspondences["keypoints0"].cpu().numpy()
    kpts1 = correspondences["keypoints1"].cpu().numpy()
    conf  = correspondences["confidence"].cpu().numpy()
    # filter by confidence
    mask = conf > 0.5
    return kpts0[mask], kpts1[mask]
```

### Pattern: Spatial consensus clustering

```python
from collections import defaultdict

def cluster_candidates(top_matches, cell_size_m=50):
    """
    Group matches into geographic cells and return the largest cluster.
    top_matches: list of (lat, lon, inliers)
    """
    def cell_key(lat, lon):
        # ~50m cell at mid-latitudes
        return (round(lat * 1000), round(lon * 1500))

    cells = defaultdict(list)
    for lat, lon, inliers in top_matches:
        cells[cell_key(lat, lon)].append((lat, lon, inliers))

    best_cell = max(cells.values(), key=lambda c: sum(x[2] for x in c))
    best_match = max(best_cell, key=lambda x: x[2])
    return best_match[0], best_match[1], len(best_cell)

lat, lon, cluster_size = cluster_candidates(top_matches)
print(f"Consensus location: {lat:.6f}, {lon:.6f} ({cluster_size} supporting matches)")
```

---

## Configuration Reference

| Parameter | Where | Default | Notes |
|-----------|-------|---------|-------|
| Grid resolution | GUI / indexer | 300 | Higher = denser coverage; don't tamper |
| Search radius | GUI Manual mode | user-set | km, used for haversine filter |
| Top-K candidates | pipeline | 500–1000 | Trades speed vs. recall |
| Heading refinement steps | pipeline | ±45° @ 15° | Tests 7 offsets × 3 FOVs = 21 crops |
| Spatial consensus cell | pipeline | 50 m | Cluster radius for consensus |
| Neighborhood expansion (Ultra) | Ultra Mode | 100 m | Searches nearby panoramas |
| RANSAC threshold | pipeline | 4.0 px | cv2.FM_RANSAC reprojection threshold |
| Max keypoints ALIKED | CUDA | 1024 | Reduce if OOM |
| Max keypoints DISK | MPS/CPU | 768 | Reduce if OOM |
| FOV range | pipeline | 70, 90, 110° | Handles zoom mismatch |

---

## Troubleshooting

### GUI is blank on macOS
```bash
brew install python-tk@3.11   # use your actual Python version
```

### CUDA out of memory
```python
# Reduce keypoints
extractor = ALIKED(max_num_keypoints=512).eval().to(device)
# Or process candidates in smaller batches
```

### LightGlue import error
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — it's a different package
```

### LoFTR not available
```bash
pip install kornia
# kornia>=0.7 includes LoFTR as kornia.feature.LoFTR
```

### Index search returns no results
- Check the radius — it must overlap the area you indexed
- Verify `index/cosplace_descriptors.npy` and `index/metadata.npz` exist
- If `cosplace_parts/` has `.npz` files but no compiled index, run `build_index.py`

### Indexing stopped mid-way
Just re-run — the indexer checks existing `cosplace_parts/` chunks and skips already-processed panoramas.

### Slow matching on CPU
- Reduce `top_k` candidates from 500 to 100–200
- Reduce max keypoints to 256
- Disable Ultra Mode
- Consider using a GPU — CPU throughput is ~10× slower than MPS, ~30× slower than CUDA

### Confidence score is low but location looks correct
Enable Ultra Mode (descriptor hopping re-searches the index using the clean matched panorama as query, often finding the exact correct entry that a degraded query missed).

---

## Key Facts for AI Agents

- **Entry point**: `test_super.py` — everything runs from here (GUI, indexing, search)
- **Index is unified**: one global index for all cities; lat/lon + radius filter handles routing
- **Device selection is automatic**: ALIKED on CUDA, DISK on MPS/CPU — don't override unless debugging
- **LightGlue must be installed from GitHub**, not PyPI
- **Gemini key is optional** — Manual mode works without it and is recommended
- **Indexing is the bottleneck** — 5km area takes 8–12h; plan accordingly
- **Ultra Mode** is opt-in — only enable for genuinely hard images (night, motion blur, low contrast)
- **RANSAC inlier count** is the primary match quality signal — >50 inliers = strong match
