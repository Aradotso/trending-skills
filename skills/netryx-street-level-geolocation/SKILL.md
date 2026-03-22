---
name: netryx-street-level-geolocation
description: Local-first street-level geolocation engine using CosPlace + LightGlue to identify GPS coordinates from any street photo
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - identify location from photo
  - netryx geolocation
  - build street view index
  - match street photo to coordinates
  - osint geolocation from image

---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, builds a searchable index of visual fingerprints, then uses a three-stage computer vision pipeline (CosPlace → ALIKED/DISK → LightGlue) to match a query image to a real-world location. Sub-50m accuracy, no landmarks required, runs entirely on local hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 50 GB+ |
| Python | 3.9+ | 3.10+ |

GPU backends: CUDA (NVIDIA), MPS (Apple Silicon M1+), CPU (slow fallback).

### Optional: Gemini API Key (AI Coarse mode)

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

### 1. Create an Index

Before any search, you must index the target area. The indexer crawls Street View panoramas, extracts 512-dim CosPlace fingerprints, and saves them to disk.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lng of the area
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk Size |
|--------|-----------|---------------|-----------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — if interrupted, re-running continues from where it left off.

**High-performance standalone indexer (large areas):**

```bash
python build_index.py
```

### 2. Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center lat/lng + radius
   - **AI Coarse**: Let Gemini estimate the region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

### 3. Ultra Mode (difficult images)

Enable the **Ultra Mode** checkbox for blurry, night, or low-texture images. Adds:
- **LoFTR** detector-free dense matching
- **Descriptor hopping** (re-search using matched panorama's descriptor)
- **Neighborhood expansion** (searches all panoramas within 100m of best match)

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search — cosine similarity, radius-filtered (haversine)
    │
    └── Top 500–1000 candidates (<1 second, single matrix multiply)
    │
    ▼
For each candidate:
    ├── Download panorama (8 GSV tiles, stitched)
    ├── Crop at indexed heading
    ├── Multi-FOV crops: 70°, 90°, 110°
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    └── LightGlue deep feature matching + RANSAC verification
    │
    ▼
Heading Refinement (±45°, 15° steps, top 15 candidates, 3 FOVs)
    │
    ├── Spatial consensus clustering (50m cells)
    └── Confidence scoring (clustering density + uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI + indexing + search
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (created during indexing)
│   └── *.npz
└── index/                 # Compiled searchable index
    ├── cosplace_descriptors.npy   # All 512-dim descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## Key Models

| Model | Role | Hardware |
|-------|------|----------|
| CosPlace | Global visual place recognition | All |
| ALIKED | Local keypoint extraction | CUDA only |
| DISK | Local keypoint extraction | MPS / CPU |
| LightGlue | Deep feature matching | All |
| LoFTR | Detector-free dense matching (Ultra) | All |

---

## Code Examples

### Extract a CosPlace Descriptor Manually

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)

image = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, image, device=device)
# descriptor.shape == (512,)
print("Descriptor extracted:", descriptor.shape)
```

### Search the Index Programmatically

```python
import numpy as np

# Load pre-built index
descriptors = np.load("index/cosplace_descriptors.npy")       # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)

latitudes  = meta["latitudes"]    # (N,)
longitudes = meta["longitudes"]   # (N,)
headings   = meta["headings"]     # (N,)
panoids    = meta["panoids"]      # (N,)

# Cosine similarity search
query_desc = descriptor / np.linalg.norm(descriptor)
db_descs   = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
scores     = db_descs @ query_desc                             # (N,)

top_k = np.argsort(scores)[::-1][:500]

print("Top match:")
print(f"  lat={latitudes[top_k[0]]:.6f}, lng={longitudes[top_k[0]]:.6f}")
print(f"  heading={headings[top_k[0]]}, panoid={panoids[top_k[0]]}")
```

### Radius Filter with Haversine

```python
import numpy as np

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

# Filter index to 2km around a known center
center_lat, center_lon = 48.8566, 2.3522   # Paris
radius_km = 2.0

distances = haversine_km(center_lat, center_lon, latitudes, longitudes)
mask = distances <= radius_km

filtered_scores = scores[mask]
filtered_indices = np.where(mask)[0]
top_local = filtered_indices[np.argsort(filtered_scores)[::-1][:500]]
```

### LightGlue Feature Matching (Verification Stage)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="aliked").eval().to(device)

query_img     = load_image("query.jpg").to(device)
candidate_img = load_image("candidate_crop.jpg").to(device)

with torch.no_grad():
    feats0 = extractor.extract(query_img)
    feats1 = extractor.extract(candidate_img)
    matches_dict = matcher({"image0": feats0, "image1": feats1})

matches_dict = rbd(matches_dict)
matches      = matches_dict["matches"]       # (M, 2) matched keypoint indices
scores       = matches_dict["scores"]        # (M,)   match confidence

print(f"Matched keypoints: {len(matches)}")
```

### DISK Extractor (MPS / CPU fallback)

```python
from lightglue import DISK
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
extractor = DISK(max_num_keypoints=768).eval().to(device)

img = load_image("query.jpg").to(device)
with torch.no_grad():
    feats = extractor.extract(img)
```

### RANSAC Geometric Verification

```python
import cv2
import numpy as np

def ransac_inliers(kpts0, kpts1, matches):
    """Return number of geometrically consistent inlier matches."""
    if len(matches) < 8:
        return 0
    pts0 = kpts0[matches[:, 0]].cpu().numpy()
    pts1 = kpts1[matches[:, 1]].cpu().numpy()
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=8.0)
    if mask is None:
        return 0
    return int(mask.sum())

inliers = ransac_inliers(feats0["keypoints"][0], feats1["keypoints"][0], matches)
print(f"RANSAC inliers: {inliers}")
```

### Ultra Mode — LoFTR Dense Matching

```python
import kornia
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = kornia.feature.LoFTR(pretrained="outdoor").eval().to(device)

def load_gray_tensor(path, size=(480, 640)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return torch.tensor(img / 255.0, dtype=torch.float32)[None, None].to(device)

img0 = load_gray_tensor("query.jpg")
img1 = load_gray_tensor("candidate_crop.jpg")

with torch.no_grad():
    result = matcher({"image0": img0, "image1": img1})

mkpts0 = result["keypoints0"].cpu().numpy()
mkpts1 = result["keypoints1"].cpu().numpy()
conf   = result["confidence"].cpu().numpy()

# Filter by confidence
good = conf > 0.5
print(f"LoFTR matches (conf>0.5): {good.sum()}")
```

### Spatial Consensus Clustering

```python
import numpy as np
from collections import defaultdict

def cluster_candidates(lats, lons, inlier_counts, cell_m=50):
    """
    Group candidates into 50m grid cells.
    Return best cluster center weighted by inlier count.
    """
    cell_deg = cell_m / 111_320  # meters → degrees (approx)
    clusters = defaultdict(list)

    for lat, lon, score in zip(lats, lons, inlier_counts):
        key = (round(lat / cell_deg), round(lon / cell_deg))
        clusters[key].append((lat, lon, score))

    best_key  = max(clusters, key=lambda k: sum(s for _, _, s in clusters[k]))
    best_pts  = clusters[best_key]
    total_w   = sum(s for _, _, s in best_pts)
    mean_lat  = sum(lat * s for lat, _, s in best_pts) / total_w
    mean_lon  = sum(lon * s for _, lon, s in best_pts) / total_w
    return mean_lat, mean_lon, total_w
```

---

## Configuration Reference

All configuration is done through the GUI or by passing arguments directly to the functions in `test_super.py` and `build_index.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Grid resolution | 300 | Panorama density — do not change |
| Top-K candidates | 500–1000 | Candidates passed to Stage 2 |
| Heading refinement steps | ±45° / 15° | Offsets tested per top-15 candidate |
| FOV crops | 70°, 90°, 110° | Zoom-mismatch handling |
| Spatial cell size | 50 m | Clustering resolution |
| Ultra neighborhood | 100 m | Expansion radius in Ultra Mode |
| LoFTR re-search threshold | 50 inliers | Below this, descriptor hopping triggers |

---

## Common Patterns

### Pattern: Multi-city unified index

All cities go into one index. The radius filter at search time handles isolation automatically:

```python
# Index Paris, then London — same index files
# Search Paris: center=(48.8566, 2.3522), radius=5
# Search London: center=(51.5074, -0.1278), radius=5
# No collision — haversine filter restricts results to requested area
```

### Pattern: Confidence interpretation

```python
def interpret_confidence(inliers, cluster_size, uniqueness_ratio):
    """
    inliers        — RANSAC-verified keypoint matches for best candidate
    cluster_size   — number of candidates in the winning spatial cluster
    uniqueness_ratio — best_inliers / second_best_inliers (different location)
    """
    if inliers > 150 and uniqueness_ratio > 2.0:
        return "HIGH — reliable pinpoint"
    elif inliers > 80 and cluster_size >= 3:
        return "MEDIUM — likely correct area"
    elif inliers > 40:
        return "LOW — plausible but verify"
    else:
        return "VERY LOW — insufficient evidence"
```

### Pattern: Headless / batch search

The GUI wraps the pipeline in `test_super.py`. For batch usage, extract the search function and call it directly, passing image paths and center coordinates in a loop.

---

## Troubleshooting

**GUI appears blank on macOS**
```bash
brew install python-tk@3.11  # match your Python version
```

**`ImportError: No module named 'lightglue'`**
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — the GitHub version is required
```

**CUDA out of memory**
- Reduce `max_num_keypoints` in ALIKED (try 512 instead of 1024)
- Process fewer candidates per batch
- Enable `torch.cuda.empty_cache()` between candidates

**MPS errors on Apple Silicon**
- ALIKED is not supported on MPS; Netryx automatically falls back to DISK on MPS
- Ensure PyTorch ≥ 2.0 for stable MPS support: `pip install torch --upgrade`

**Index search returns 0 results**
- Verify the index was compiled: `index/cosplace_descriptors.npy` and `index/metadata.npz` must both exist
- If only `cosplace_parts/*.npz` exist, trigger index rebuild via `build_index.py`
- Check that your search radius actually overlaps indexed area coordinates

**Low inlier counts on all candidates**
- Enable Ultra Mode (LoFTR + descriptor hopping + neighborhood expansion)
- Try flipping the image horizontally before querying (reversed perspective)
- Ensure the query image is street-level, not aerial or indoor

**Indexing stalls / API errors**
- Street View API has rate limits; the indexer handles retries but very large areas may require multiple sessions
- Indexing is resumable — just re-run and it continues from the last saved checkpoint in `cosplace_parts/`
