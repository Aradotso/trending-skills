```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from any street photo using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - identify location from photo
  - osint geolocation tool
  - reverse geolocate image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that determines the exact GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a searchable visual index, then matches a query photo against that index using a three-stage computer vision pipeline — achieving sub-50m accuracy with no landmark recognition or internet image matching required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult images)
pip install kornia
```

### Platform Notes

| Platform | Feature Extractor | Notes |
|---|---|---|
| NVIDIA CUDA | ALIKED (1024 keypoints) | Best performance |
| Apple MPS (M1+) | DISK (768 keypoints) | Good performance |
| CPU | DISK | Works, significantly slower |

**macOS tkinter fix** (if GUI renders blank):
```bash
brew install python-tk@3.11   # match your Python version
```

### Optional: Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles both indexing and searching.

---

## Core Workflow

### Step 1 — Create an Index

Before any search, you must build a local index for the geographic area of interest.

1. Select **Create** mode in the GUI
2. Enter center coordinates (lat, lon)
3. Set search radius (km)
4. Set grid resolution (default: 300 — do not change)
5. Click **Create Index**

Indexing is incremental — safe to interrupt and resume.

**Radius/time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|---|---|---|---|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius (recommended)
   - **AI Coarse**: Gemini analyzes visual cues to guess region automatically
4. Click **Run Search** → **Start Full Search**
5. Result appears on map with GPS coordinates and confidence score

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim fingerprint)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search — cosine similarity, radius-filtered (haversine)
    │
    ├── Top 500–1000 candidates
    │
    ▼
Download panoramas → Crop at 3 FOVs (70°, 90°, 110°)
    │
    ├── ALIKED/DISK keypoint extraction
    ├── LightGlue deep feature matching
    ├── RANSAC geometric verification
    │
    ▼
Heading Refinement — ±45°, 15° steps, top 15 candidates
    │
    ├── Spatial consensus clustering (50m cells)
    ├── Confidence scoring (uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

**Timing:** Stage 1 (retrieval) < 1 second. Stages 2–3 (verification + refinement) = 2–5 minutes for 300–500 candidates.

---

## Project Structure

```
netryx/
├── test_super.py           # Main app — GUI, indexing pipeline, search pipeline
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks written during indexing
│   └── *.npz
└── index/                  # Compiled searchable index
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (matrix)
    └── metadata.npz               # Coordinates, headings, panoid IDs
```

---

## Key Modules and Usage Patterns

### Loading CosPlace Descriptors

```python
import numpy as np

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # shape: (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)

lats = meta["lats"]        # float array, shape (N,)
lons = meta["lons"]        # float array, shape (N,)
headings = meta["headings"]  # float array, shape (N,)
panoids = meta["panoids"]    # string array, shape (N,)
```

### Extracting a CosPlace Descriptor from an Image

```python
# cosplace_utils.py provides the model loader
from cosplace_utils import get_cosplace_model, extract_descriptor
from PIL import Image
import torch

model = get_cosplace_model()   # loads pretrained CosPlace weights
img = Image.open("query.jpg").convert("RGB")

descriptor = extract_descriptor(model, img)   # returns np.ndarray shape (512,)
```

### Cosine Similarity Search with Radius Filter

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
    """
    Returns indices of top_k most similar descriptors within radius_km of center.
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    in_radius = np.where(distances <= radius_km)[0]

    if len(in_radius) == 0:
        return []

    # Cosine similarity
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db = descriptors[in_radius]
    norms = np.linalg.norm(db, axis=1, keepdims=True) + 1e-8
    db_normed = db / norms

    scores = db_normed @ q                       # shape: (len(in_radius),)
    ranked = np.argsort(scores)[::-1][:top_k]    # descending

    return in_radius[ranked]   # original indices into full arrays
```

### Using Both Query and Flipped Descriptor (as Netryx does)

```python
import numpy as np
from PIL import Image, ImageOps

def get_combined_descriptor(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_flipped = ImageOps.mirror(img)

    desc = extract_descriptor(model, img)
    desc_flipped = extract_descriptor(model, img_flipped)

    # Average both — catches reversed perspectives
    combined = (desc + desc_flipped) / 2.0
    combined /= np.linalg.norm(combined) + 1e-8
    return combined
```

### Feature Matching with LightGlue (ALIKED on CUDA)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

def match_images(img0_path, img1_path):
    img0 = load_image(img0_path).to(device)
    img1 = load_image(img1_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches_data = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches_data = [rbd(x) for x in (feats0, feats1, matches_data)]

    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches_data["matches"]   # shape: (M, 2) — indices into kpts0, kpts1

    matched_kpts0 = kpts0[matches[:, 0]].cpu().numpy()
    matched_kpts1 = kpts1[matches[:, 1]].cpu().numpy()

    return matched_kpts0, matched_kpts1, len(matches)
```

### RANSAC Geometric Verification (filtering false matches)

```python
import cv2
import numpy as np

def ransac_verify(kpts0, kpts1, threshold=3.0):
    """
    Returns number of geometric inliers.
    Higher inliers = stronger geometric match.
    """
    if len(kpts0) < 8:
        return 0

    F, mask = cv2.findFundamentalMat(
        kpts0, kpts1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999
    )

    if mask is None:
        return 0

    inliers = int(mask.sum())
    return inliers
```

### DISK on MPS (Mac) or CPU

```python
from lightglue import DISK

# Netryx automatically selects extractor based on device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    extractor = DISK(max_num_keypoints=768).eval().to(device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    device = torch.device("cpu")
    extractor = DISK(max_num_keypoints=768).eval().to(device)
```

### Spatial Consensus Clustering (50m cells)

```python
import numpy as np
from collections import defaultdict

def cluster_by_location(candidate_lats, candidate_lons, inlier_counts,
                         cell_size_m=50):
    """
    Groups candidates into 50m grid cells.
    Returns the cell with the most total inliers.
    """
    # ~50m in degrees
    cell_deg = cell_size_m / 111_000

    clusters = defaultdict(list)
    for i, (lat, lon, inliers) in enumerate(zip(candidate_lats, candidate_lons, inlier_counts)):
        cell_lat = round(lat / cell_deg) * cell_deg
        cell_lon = round(lon / cell_deg) * cell_deg
        clusters[(cell_lat, cell_lon)].append((i, inliers))

    best_cell = max(clusters, key=lambda c: sum(x[1] for x in clusters[c]))
    best_indices = [idx for idx, _ in clusters[best_cell]]

    # Return the single best match within the winning cluster
    best_in_cluster = max(clusters[best_cell], key=lambda x: x[1])
    return best_indices, best_in_cluster[0]
```

### Ultra Mode — LoFTR Dense Matching

```python
import kornia
import torch
import cv2
import numpy as np

def loftr_match(img0_path, img1_path, device):
    matcher = kornia.feature.LoFTR(pretrained="outdoor").eval().to(device)

    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    img0_t = torch.from_numpy(img0).float()[None, None] / 255.0
    img1_t = torch.from_numpy(img1).float()[None, None] / 255.0

    with torch.no_grad():
        input_dict = {
            "image0": img0_t.to(device),
            "image1": img1_t.to(device)
        }
        correspondences = matcher(input_dict)

    kpts0 = correspondences["keypoints0"].cpu().numpy()
    kpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Filter by confidence
    mask = confidence > 0.5
    return kpts0[mask], kpts1[mask]
```

---

## Index Management

### Building the Index Standalone (Large Areas)

For large areas (5km+), use the dedicated builder instead of the GUI:

```bash
python build_index.py
```

### Multi-Region Index

The index is unified — all regions share a single `cosplace_descriptors.npy`. The radius filter at search time restricts results to the target area automatically.

```
# Index Paris, then London — same index file
# Search Paris: center=(48.8566, 2.3522), radius=5
# Search London: center=(51.5074, -0.1278), radius=5
# No overlap — haversine filter handles separation
```

### Index File Locations

```
cosplace_parts/          ← written incrementally during Create mode
    part_0000.npz
    part_0001.npz
    ...

index/
    cosplace_descriptors.npy   ← compiled after all parts done
    metadata.npz               ← lat/lon/heading/panoid per row
```

---

## Configuration Reference

| Parameter | Default | Notes |
|---|---|---|
| Grid resolution | 300 | Do not change — controls panorama sampling density |
| Top-K candidates | 500–1000 | Larger = slower but more thorough |
| ALIKED keypoints | 1024 | CUDA only |
| DISK keypoints | 768 | MPS/CPU |
| RANSAC threshold | 3.0 px | Inlier reprojection threshold |
| Heading refinement | ±45°, 15° steps | Applied to top 15 candidates |
| Cluster cell size | 50m | Spatial consensus grid |
| FOV crops | 70°, 90°, 110° | Handles zoom mismatch |
| LoFTR confidence | 0.5 | Ultra Mode filter threshold |
| Neighborhood expansion | 100m | Ultra Mode only |

---

## Common Patterns

### Pattern: Headless/Scripted Search

The GUI wraps the pipeline in `test_super.py`. To call the pipeline programmatically, study and import from `test_super.py` and `cosplace_utils.py` directly:

```python
# Pseudocode — adapt based on test_super.py internals
from cosplace_utils import get_cosplace_model, extract_descriptor
import numpy as np

model = get_cosplace_model()

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

# Extract query descriptor
desc = extract_descriptor(model, query_image)

# Search (see search_index example above)
candidates = search_index(desc, descriptors, meta["lats"], meta["lons"],
                          center_lat=48.8566, center_lon=2.3522, radius_km=2.0)

# candidates = indices into meta arrays for top matches
print(meta["lats"][candidates[0]], meta["lons"][candidates[0]])
```

### Pattern: Evaluate Match Confidence

```python
def confidence_score(best_inliers, second_best_inliers, cluster_size):
    """
    Uniqueness ratio + cluster bonus.
    """
    if second_best_inliers == 0:
        uniqueness = 1.0
    else:
        uniqueness = best_inliers / second_best_inliers

    cluster_bonus = min(cluster_size / 5.0, 1.0)   # saturates at 5 clustered matches
    score = 0.6 * uniqueness + 0.4 * cluster_bonus
    return min(score, 1.0)

# Example
score = confidence_score(best_inliers=120, second_best_inliers=45, cluster_size=3)
print(f"Confidence: {score:.2f}")   # → 0.76
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'lightglue'`

LightGlue must be installed from GitHub, not PyPI:
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### GUI renders blank on macOS

```bash
brew install python-tk@3.11   # match your exact Python version
```

### CUDA out of memory

Reduce keypoints per image:
```python
extractor = ALIKED(max_num_keypoints=512).eval().to(device)  # reduce from 1024
```

Or process candidates in smaller batches.

### MPS errors on Mac

Some ops fall back to CPU on MPS. Set environment variable to suppress warnings:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python test_super.py
```

### Index search returns no candidates

- Verify the center coordinates are correct (lat/lon not swapped)
- Increase the search radius
- Confirm the index was compiled (check `index/cosplace_descriptors.npy` exists)
- If only `cosplace_parts/` exists, the auto-build step did not complete — re-run indexing or call the build step manually via `build_index.py`

### Low inlier counts on all candidates (< 20 inliers)

- The query photo's location may not be in the indexed area
- Try **Ultra Mode** (enable checkbox in GUI) — adds LoFTR + descriptor hopping + neighborhood expansion
- Increase search radius
- Re-index with higher grid resolution coverage

### Indexing interrupted mid-run

Safe to re-run — the indexer writes `cosplace_parts/*.npz` incrementally and skips already-processed panoramas on resume.

---

## Model References

| Model | Role | Source |
|---|---|---|
| CosPlace | Global visual place recognition | [github.com/gmberton/cosplace](https://github.com/gmberton/cosplace) |
| ALIKED | Local keypoint extraction (CUDA) | [github.com/naver/alike](https://github.com/naver/alike) |
| DISK | Local keypoint extraction (MPS/CPU) | [github.com/cvlab-epfl/disk](https://github.com/cvlab-epfl/disk) |
| LightGlue | Deep feature matching | [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue) |
| LoFTR | Dense matching, Ultra Mode | [github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) |
```
