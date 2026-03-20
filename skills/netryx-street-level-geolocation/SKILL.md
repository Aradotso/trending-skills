```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a locally-hosted open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo locally
  - find GPS coordinates from street image
  - run Netryx geolocation
  - index street view panoramas
  - use netryx to locate an image
  - street level geolocation with computer vision
  - build a street view index for geolocation
  - run visual place recognition on a photo
---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that takes any street-level photograph and returns precise GPS coordinates (sub-50m accuracy). It crawls street-view panoramas, indexes them as 512-dimensional CosPlace fingerprints, then at query time runs CosPlace retrieval → ALIKED/DISK keypoint extraction → LightGlue feature matching → RANSAC geometric verification to identify the exact location. No cloud API needed for searching — everything runs on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR dense matcher for Ultra Mode
pip install kornia
```

### Platform requirements

| Hardware | Supported backend | Notes |
|---|---|---|
| NVIDIA GPU (4GB+ VRAM) | CUDA | ALIKED extractor (1024 keypoints) |
| Apple Silicon (M1+) | MPS | DISK extractor (768 keypoints) |
| CPU only | CPU | Works, significantly slower |

### Optional: Gemini API key (AI Coarse mode)

```bash
export GEMINI_API_KEY="your_key_here"   # from aistudio.google.com
```

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank-GUI fix: `brew install python-tk@3.11` (match your Python version).

---

## Core Workflow

### 1. Create an Index (crawl + embed an area)

In the GUI:
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Indexing is incremental — safe to interrupt and resume.

**Time/size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|---|---|---|---|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 h | ~250 MB |
| 5 km | ~30,000 | 8–12 h | ~3 GB |
| 10 km | ~100,000 | 24–48 h | ~7 GB |

Output files:
```
cosplace_parts/        # raw .npz embedding chunks
index/
  cosplace_descriptors.npy   # stacked 512-dim descriptors
  metadata.npz               # lat, lon, heading, panoid per entry
```

### 2. Search (locate a query image)

In the GUI:
1. Select **Search** mode
2. Upload a street photo
3. Choose search method:
   - **Manual**: provide approximate center lat/lon + radius
   - **AI Coarse**: Gemini infers the region from visual clues
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on an interactive map

### 3. Ultra Mode (hard images)

Enable **Ultra Mode** checkbox before searching. Adds:
- **LoFTR** dense matching (handles blur/low-texture/night)
- **Descriptor hopping**: re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: checks all panoramas within 100m of best match

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loader + descriptor extraction helpers
├── build_index.py         # Standalone high-throughput index builder (large areas)
├── requirements.txt
├── cosplace_parts/        # Incremental embedding chunks (auto-created)
└── index/
    ├── cosplace_descriptors.npy
    └── metadata.npz
```

---

## Using the Pipeline Programmatically

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape → (512,)
```

### Load the index and run a radius-filtered cosine search

```python
import numpy as np

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]          # (N,)
lons = meta["lons"]          # (N,)
headings = meta["headings"]  # (N,)
panoids = meta["panoids"]    # (N,)

def haversine_filter(lats, lons, center_lat, center_lon, radius_km):
    """Return boolean mask of entries within radius_km of center."""
    R = 6371.0
    dlat = np.radians(lats - center_lat)
    dlon = np.radians(lons - center_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(center_lat))
         * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a)) <= radius_km

def cosine_search(query_desc, descriptors, mask, top_k=500):
    """Cosine similarity search over masked subset."""
    q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
    d = descriptors[mask]
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    scores = d_norm @ q                          # dot product = cosine sim
    top_local = np.argsort(scores)[::-1][:top_k]
    # Map local indices back to global
    global_indices = np.where(mask)[0][top_local]
    return global_indices, scores[top_local]

# Example: search within 2km of Paris centre
center_lat, center_lon = 48.8566, 2.3522
mask = haversine_filter(lats, lons, center_lat, center_lon, radius_km=2.0)

query_desc = extract_descriptor(model, img, device=device)
candidates, scores = cosine_search(query_desc, descriptors, mask, top_k=500)

print(f"Top candidate: lat={lats[candidates[0]]:.6f}, lon={lons[candidates[0]]:.6f}")
```

### LightGlue feature matching between query and a candidate crop

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def match_images(img_path_a, img_path_b, device):
    image0 = load_image(img_path_a).to(device)
    image1 = load_image(img_path_b).to(device)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
    matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]
    return matched_kps0, matched_kps1, matches01["matching_scores0"]

kps0, kps1, scores = match_images("query.jpg", "candidate_crop.jpg", device)
print(f"Matched keypoints: {len(kps0)}")
```

### RANSAC geometric verification (count inliers)

```python
import cv2
import numpy as np

def ransac_inliers(kps0, kps1):
    """Return number of geometrically consistent matches."""
    pts0 = kps0.cpu().numpy()
    pts1 = kps1.cpu().numpy()
    if len(pts0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        pts0, pts1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999
    )
    return int(mask.sum()) if mask is not None else 0

inliers = ransac_inliers(kps0, kps1)
print(f"RANSAC inliers: {inliers}")  # >50 = strong match
```

### Build a standalone index for a large area

```bash
# Use build_index.py for large-scale crawling (more efficient than the GUI)
python build_index.py \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 5.0 \
    --resolution 300
```

---

## Multi-FOV crop strategy

Netryx tests three fields of view per candidate to handle zoom mismatches:

```python
FOV_VARIANTS = [70, 90, 110]   # degrees

# The pipeline crops the same panorama at each FOV and runs matching
# on all three — the FOV that yields most inliers wins.
# This is handled automatically inside the search pipeline in test_super.py.
```

---

## Heading refinement (post-match)

After finding a top candidate, the pipeline re-tests ±45° heading offsets at 15° steps:

```python
HEADING_OFFSETS = [-45, -30, -15, 0, 15, 30, 45]  # degrees
TOP_CANDIDATES_FOR_REFINEMENT = 15

# For each of the top 15 candidates, 7 headings × 3 FOVs = 21 crops tested.
# Best (heading, FOV) pair by inlier count is selected.
```

---

## Confidence scoring logic

```python
def compute_confidence(top_inliers, runner_up_inliers, cluster_size):
    """
    Heuristic confidence score returned with each result.
    - uniqueness_ratio: how much better best match is vs next distinct location
    - cluster_bonus: reward if multiple candidates agree on same 50m cell
    """
    uniqueness_ratio = top_inliers / (runner_up_inliers + 1e-8)
    cluster_bonus = min(cluster_size / 5.0, 1.0)   # caps at 5 agreeing candidates
    confidence = min((uniqueness_ratio * 0.6 + cluster_bonus * 0.4), 1.0)
    return round(confidence, 3)
```

---

## Index architecture — multi-city support

All cities live in one unified index. Search radius handles isolation:

```python
# Index Paris
python test_super.py  # Create mode, center=48.8566,2.3522, radius=5km

# Index London (same index files, just appends)
python test_super.py  # Create mode, center=51.5074,-0.1278, radius=5km

# Search only Paris (radius filter excludes London entries automatically)
mask = haversine_filter(lats, lons, 48.8566, 2.3522, radius_km=5.0)
```

---

## Common Patterns

### Pattern: Fully headless search (no GUI)

```python
# Pseudocode mirroring test_super.py internals — adapt to your use case
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import numpy as np, torch, cv2

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load_cosplace_model(device=device)

# 1. Load query
img = Image.open("unknown_street.jpg").convert("RGB")
q_desc = extract_descriptor(model, img, device=device)

# Also try flipped (catches reversed perspective)
q_desc_flip = extract_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT), device=device)
q_desc_combined = (q_desc + q_desc_flip) / 2   # average both

# 2. Load index
descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

# 3. Radius filter + cosine search
mask = haversine_filter(meta["lats"], meta["lons"], CENTER_LAT, CENTER_LON, RADIUS_KM)
candidates, scores = cosine_search(q_desc_combined, descriptors, mask, top_k=500)

# 4. For each candidate: download pano, crop, match, RANSAC
# (see test_super.py for full panorama download + tile stitching logic)
best_candidate = candidates[0]
print(f"Best guess: {meta['lats'][best_candidate]:.6f}, {meta['lons'][best_candidate]:.6f}")
```

### Pattern: Check if Ultra Mode dependencies are available

```python
def check_ultra_mode():
    try:
        import kornia
        print("Ultra Mode available (LoFTR via kornia)")
        return True
    except ImportError:
        print("Ultra Mode unavailable — install kornia: pip install kornia")
        return False
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: lightglue` | LightGlue not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| GUI appears blank on macOS | System tkinter bug | `brew install python-tk@3.11` (match your Python version) |
| ALIKED not loading on Mac | ALIKED requires CUDA | System auto-falls back to DISK on MPS/CPU — expected |
| Low inlier count (<20) on all candidates | Index doesn't cover query area | Re-index with a larger radius or different center |
| Ultra Mode import error | kornia not installed | `pip install kornia` |
| Indexing stops mid-run | Network timeout / API rate limit | Re-run — indexing resumes from last saved chunk |
| `GEMINI_API_KEY` not found | Env var not set | `export GEMINI_API_KEY="..."` before launching |
| OOM on GPU during matching | VRAM too small | Reduce `max_num_keypoints` in extractor init (e.g. 512) |
| Result confidence very low (<0.3) | Ambiguous scene, try Ultra Mode | Enable Ultra Mode checkbox before searching |

---

## Key constants to tune

```python
# In test_super.py — adjust these for performance vs accuracy tradeoffs

TOP_K_RETRIEVAL = 500          # candidates passed to Stage 2 (lower = faster)
GRID_RESOLUTION = 300          # panorama density during indexing (higher = denser)
RANSAC_THRESHOLD = 3.0         # pixels — stricter = fewer but more reliable inliers
REFINEMENT_TOP_N = 15          # candidates re-tested in heading refinement
CLUSTER_CELL_SIZE_M = 50       # spatial consensus cell size in meters
NEIGHBORHOOD_RADIUS_M = 100    # Ultra Mode expansion radius
FOV_VARIANTS = [70, 90, 110]   # fields of view tested per candidate
HEADING_OFFSETS = list(range(-45, 46, 15))  # 7 heading steps for refinement
```
```
