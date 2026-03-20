```markdown
---
name: netryx-street-level-geolocation
description: Local-first street-level geolocation engine using CosPlace, ALIKED/DISK, and LightGlue to identify GPS coordinates from any street photo with sub-50m accuracy.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - run netryx geolocation
  - build a street view index
  - match street photo to coordinates
  - osint geolocation from image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, indexes them with CosPlace visual fingerprints, then uses ALIKED/DISK keypoint extraction and LightGlue feature matching to verify candidates — achieving sub-50m accuracy with no internet image match required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required — LightGlue is not on PyPI
pip install git+https://github.com/cvg/LightGlue.git

# Optional — enables Ultra Mode (LoFTR dense matching)
pip install kornia
```

### Gemini API key (optional — AI Coarse mode only)

```bash
export GEMINI_API_KEY="your_key_here"
```

### macOS tkinter fix (blank GUI)

```bash
brew install python-tk@3.11   # match your Python version
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

GPU backends: CUDA (NVIDIA) → ALIKED, MPS (Apple Silicon) → DISK, CPU → DISK (slow).

---

## Launch the GUI

```bash
python test_super.py
```

The GUI has two modes: **Create** (build an index) and **Search** (geolocate a photo).

---

## Core Workflow

### Step 1 — Build an Index

Index a geographic area before searching. The indexer crawls street-view panoramas, extracts CosPlace 512-dim descriptors, and saves them incrementally (resumable if interrupted).

**GUI steps:**
1. Select **Create** mode
2. Enter center lat/lon
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Indexing time reference:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hr        | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hr       | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hr      | ~7 GB      |

Index files written to:
```
cosplace_parts/      # raw .npz chunks per grid cell
index/
  cosplace_descriptors.npy   # all 512-dim descriptors
  metadata.npz               # lat, lon, heading, panoid
```

Multiple cities can share one index — radius filtering at search time isolates the right area automatically.

### Step 2 — Search

**GUI steps:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: provide approximate center lat/lon + radius
   - **AI Coarse**: Gemini infers region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Pipeline Internals

### Stage 1 — Global Retrieval (CosPlace)

```python
# cosplace_utils.py — descriptor extraction pattern
from cosplace_utils import load_cosplace_model, get_descriptor

model = load_cosplace_model(device="cuda")  # or "mps" / "cpu"

# Extract 512-dim fingerprint from a query image path
descriptor = get_descriptor(model, "query.jpg", device="cuda")

# The pipeline also extracts a flipped descriptor to catch reversed perspectives
import torchvision.transforms.functional as TF
from PIL import Image
import torch

img = Image.open("query.jpg").convert("RGB")
img_flipped = TF.hflip(img)
descriptor_flipped = get_descriptor(model, img_flipped, device="cuda")
```

Index search is a single cosine-similarity matrix multiply — runs in under 1 second regardless of index size.

```python
import numpy as np

# Load prebuilt index
descriptors = np.load("index/cosplace_descriptors.npy")  # shape (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats, lons = meta["lats"], meta["lons"]

# Cosine similarity (descriptors are L2-normalised)
scores = descriptors @ descriptor.T          # (N,)

# Haversine radius filter — keep only candidates within search_radius_km
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

center_lat, center_lon = 48.8566, 2.3522   # Paris
radius_km = 2.0

mask = np.array([
    haversine_km(center_lat, center_lon, la, lo) <= radius_km
    for la, lo in zip(lats, lons)
])
scores[~mask] = -1

top_indices = np.argsort(scores)[::-1][:500]   # top 500 candidates
```

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

```python
# Pseudo-code mirroring the internal pipeline
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = "cuda"   # or "mps" / "cpu"

# Feature extractor — ALIKED on CUDA, DISK elsewhere
if device == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device == "cuda" else "disk").eval().to(device)

query_img = load_image("query.jpg").to(device)
feats_query = extractor.extract(query_img)

best_inliers = 0
best_candidate = None

for candidate in top_candidates:
    # candidate["pano_crop"] is a rectilinear crop at indexed heading
    # Three FOVs (70°, 90°, 110°) are tested to handle zoom mismatches
    for fov_crop in candidate["fov_crops"]:
        cand_img = load_image(fov_crop).to(device)
        feats_cand = extractor.extract(cand_img)

        matches = matcher({"image0": feats_query, "image1": feats_cand})
        feats_query_r, feats_cand_r, matches_r = [
            rbd(x) for x in [feats_query, feats_cand, matches]
        ]

        pts0 = feats_query_r["keypoints"][matches_r["matches"][..., 0]]
        pts1 = feats_cand_r["keypoints"][matches_r["matches"][..., 1]]

        # RANSAC geometric verification
        if len(pts0) >= 4:
            import cv2
            _, inlier_mask = cv2.findHomography(
                pts0.cpu().numpy(), pts1.cpu().numpy(),
                cv2.RANSAC, ransacReprojThreshold=4.0
            )
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            if inliers > best_inliers:
                best_inliers = inliers
                best_candidate = candidate
```

### Stage 3 — Refinement

```python
# Heading refinement: ±45° at 15° steps for top 15 candidates
heading_offsets = range(-45, 46, 15)   # [-45, -30, -15, 0, 15, 30, 45]

# Spatial consensus clustering in 50m grid cells
def latlon_to_cell(lat, lon, cell_size_m=50):
    """Snap coordinates to nearest 50m grid cell."""
    lat_step = cell_size_m / 111320
    lon_step = cell_size_m / (111320 * np.cos(np.radians(lat)))
    return round(lat / lat_step), round(lon / lon_step)

from collections import Counter
cell_votes = Counter()
candidate_by_cell = {}

for c in top_candidates:
    cell = latlon_to_cell(c["lat"], c["lon"])
    cell_votes[cell] += c["inliers"]
    if cell not in candidate_by_cell or c["inliers"] > candidate_by_cell[cell]["inliers"]:
        candidate_by_cell[cell] = c

best_cell = cell_votes.most_common(1)[0][0]
final_result = candidate_by_cell[best_cell]
print(f"Result: {final_result['lat']:.6f}, {final_result['lon']:.6f}")
```

---

## Ultra Mode

Enable in the GUI checkbox for difficult images (night, blur, low texture).

What it adds:
1. **LoFTR** — detector-free dense matching (requires `kornia`)
2. **Descriptor hopping** — re-searches the index using the matched panorama's clean descriptor
3. **Neighborhood expansion** — checks all panoramas within 100m of the best match

```python
# LoFTR usage pattern (Ultra Mode)
import kornia.feature as KF
import torch, cv2
import numpy as np

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def to_gray_tensor(img_path, size=(480, 640)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size[1], size[0]))
    return torch.from_numpy(img).float()[None, None] / 255.0

query_t = to_gray_tensor("query.jpg").to(device)
cand_t  = to_gray_tensor("candidate.jpg").to(device)

with torch.no_grad():
    out = loftr({"image0": query_t, "image1": cand_t})

mkpts0 = out["keypoints0"].cpu().numpy()
mkpts1 = out["keypoints1"].cpu().numpy()
conf   = out["confidence"].cpu().numpy()

# Filter by confidence
good = conf > 0.5
mkpts0, mkpts1 = mkpts0[good], mkpts1[good]

# RANSAC
if len(mkpts0) >= 4:
    _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loader + descriptor extraction
├── build_index.py         # High-throughput standalone index builder
├── requirements.txt
├── cosplace_parts/        # Incremental .npz chunks (created at index time)
└── index/
    ├── cosplace_descriptors.npy   # Compiled descriptor matrix (N × 512)
    └── metadata.npz               # lats, lons, headings, panoid strings
```

---

## Common Patterns

### Check detected device

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
# ALIKED used on cuda, DISK used on mps/cpu
```

### Load the compiled index manually

```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")   # float32, L2-normalised
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Index size: {len(descriptors)} panoramas")
print(f"Keys: {list(meta.keys())}")   # lats, lons, headings, panoids
```

### Resume interrupted indexing

Just re-run **Create Index** with the same parameters. The indexer checks `cosplace_parts/` for existing chunks and skips completed grid cells automatically.

### Search multiple cities in one index

```python
# Paris search
center = (48.8566, 2.3522)
radius_km = 3.0

# London search — same index file, different center/radius
center = (51.5074, -0.1278)
radius_km = 5.0

# The haversine mask ensures only the relevant city's panoramas are scored
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank | macOS bundled tkinter bug | `brew install python-tk@3.11` |
| `ImportError: lightglue` | LightGlue not installed from source | `pip install git+https://github.com/cvg/LightGlue.git` |
| Ultra Mode disabled / LoFTR missing | kornia not installed | `pip install kornia` |
| CUDA OOM during matching | Too many keypoints | Reduce `max_num_keypoints` in extractor init |
| MPS fallback to CPU | PyTorch MPS not available | Upgrade to PyTorch ≥ 2.0 and macOS ≥ 13 |
| Index search returns 0 candidates | Radius too small or area not indexed | Increase radius or re-index area |
| Low inlier count on real match | FOV mismatch between query and panorama | Enable Ultra Mode; heading refinement covers ±45° |
| Indexing stalls mid-way | Street View API rate limit or network issue | Re-run — incremental save resumes from last chunk |

### Verify LightGlue install

```python
python -c "from lightglue import LightGlue, ALIKED, DISK; print('LightGlue OK')"
```

### Verify CosPlace descriptor extraction

```python
from cosplace_utils import load_cosplace_model, get_descriptor
import torch

device = "cpu"
model = load_cosplace_model(device=device)
desc = get_descriptor(model, "any_street_photo.jpg", device=device)
print(f"Descriptor shape: {desc.shape}")   # should be (512,)
```

---

## Models Reference

| Model | Role | Backend |
|-------|------|---------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global visual place recognition | All devices |
| [ALIKED](https://github.com/naver/alike) | Local keypoint extraction | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoint extraction | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All devices |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense detector-free matching (Ultra) | All devices |
```
