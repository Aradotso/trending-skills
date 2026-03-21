---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photograph to GPS coordinates using computer vision pipelines running entirely on local hardware.
triggers:
  - geolocate a street photo
  - find GPS coordinates from a street image
  - index street view panoramas for geolocation
  - run netryx geolocation
  - street level image geolocation locally
  - use netryx to find location of photo
  - build a netryx index for a city
  - geolocation with cosplace and lightglue
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies exact GPS coordinates from any street-level photograph. It crawls street-view panoramas, indexes them using CosPlace visual embeddings, and matches query images via ALIKED/DISK keypoint extraction and LightGlue deep feature matching — all on your own hardware. Sub-50m accuracy, no landmarks required.

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

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

**macOS tkinter fix** (if GUI renders blank):
```bash
brew install python-tk@3.11   # match your Python version
```

**Optional Gemini API key** for AI Coarse geolocation mode:
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 50 GB+ |
| Python | 3.9+ | 3.10+ |

GPU backends used automatically:
- **NVIDIA**: CUDA → ALIKED (1024 keypoints)
- **Mac M-series**: MPS → DISK (768 keypoints)
- **CPU**: DISK (slowest, functional)

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface for both indexing and searching.

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area by crawling street-view panoramas and extracting CosPlace fingerprints.

In the GUI:
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Indexing is incremental — safe to interrupt and resume.

**Time and size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Output files:
```
cosplace_parts/        # raw embedding chunks (.npz per batch)
index/
  cosplace_descriptors.npy   # all 512-dim descriptors
  metadata.npz               # lat/lon, headings, panorama IDs
```

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: provide approximate center lat/lon + radius (most reliable)
   - **AI Coarse**: Gemini estimates the region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score displayed on map

Enable **Ultra Mode** for degraded images (night, blur, low texture) — slower but more robust.

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI, indexing pipeline, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created)
└── index/
    ├── cosplace_descriptors.npy
    └── metadata.npz
```

---

## Pipeline Internals

### Stage 1 — Global Retrieval (CosPlace)

```python
# cosplace_utils.py — extract a 512-dim descriptor from an image
from cosplace_utils import get_cosplace_model, extract_descriptor
from PIL import Image

model, transform = get_cosplace_model()   # loads CosPlace ResNet-50 backbone

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, transform, img)  # shape: (512,)

# Also extract flipped version to catch reversed perspectives
import torchvision.transforms.functional as TF
flipped = TF.hflip(img)
descriptor_flipped = extract_descriptor(model, transform, flipped)
```

### Stage 2 — Index Search (cosine similarity + radius filter)

```python
import numpy as np

# Load prebuilt index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]     # (N,)
lons = meta["lons"]     # (N,)
headings = meta["headings"]
panoids = meta["panoids"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    # Radius filter
    distances = haversine_km(center_lat, center_lon, lats, lons)
    mask = distances <= radius_km

    filtered_desc = descriptors[mask]
    filtered_idx  = np.where(mask)[0]

    # Cosine similarity (single matrix multiply)
    query_norm = query_descriptor / np.linalg.norm(query_descriptor)
    desc_norm  = filtered_desc / np.linalg.norm(filtered_desc, axis=1, keepdims=True)
    sims = desc_norm @ query_norm

    top_local = np.argsort(sims)[::-1][:top_k]
    top_global = filtered_idx[top_local]

    return [
        {
            "panoid": panoids[i],
            "lat": lats[i],
            "lon": lons[i],
            "heading": headings[i],
            "similarity": sims[top_local[rank]]
        }
        for rank, i in enumerate(top_global)
    ]
```

### Stage 3 — Local Feature Matching (ALIKED/DISK + LightGlue)

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# Choose extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
    matcher   = LightGlue(features="aliked").eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)
    matcher   = LightGlue(features="disk").eval().to(device)

def match_images(query_path, candidate_path):
    img0 = load_image(query_path).to(device)
    img1 = load_image(candidate_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
    matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]

    return matched_kps0, matched_kps1, matches01["matching_scores0"]
```

### RANSAC Geometric Verification

```python
import cv2

def ransac_inliers(kps0, kps1):
    if len(kps0) < 4:
        return 0
    pts0 = kps0.cpu().numpy()
    pts1 = kps1.cpu().numpy()
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=4.0)
    if mask is None:
        return 0
    return int(mask.sum())
```

---

## Multi-FOV Cropping Pattern

Netryx tests 3 fields of view per candidate to handle zoom mismatches:

```python
FOV_CANDIDATES = [70, 90, 110]   # degrees

def get_best_fov_match(query_path, panorama_image, heading):
    best_inliers = 0
    best_fov = None

    for fov in FOV_CANDIDATES:
        crop = rectilinear_crop(panorama_image, heading=heading, fov=fov, pitch=0)
        kps0, kps1, _ = match_images(query_path, crop)
        inliers = ransac_inliers(kps0, kps1)
        if inliers > best_inliers:
            best_inliers = inliers
            best_fov = fov

    return best_fov, best_inliers
```

---

## Heading Refinement Pattern

After finding the initial best match, refine by testing heading offsets:

```python
HEADING_OFFSETS = range(-45, 46, 15)   # -45° to +45° in 15° steps

def refine_heading(query_path, panorama_image, base_heading):
    best_inliers = 0
    best_heading = base_heading

    for offset in HEADING_OFFSETS:
        heading = (base_heading + offset) % 360
        for fov in [70, 90, 110]:
            crop = rectilinear_crop(panorama_image, heading=heading, fov=fov, pitch=0)
            kps0, kps1, _ = match_images(query_path, crop)
            inliers = ransac_inliers(kps0, kps1)
            if inliers > best_inliers:
                best_inliers = inliers
                best_heading = heading

    return best_heading, best_inliers
```

---

## Spatial Consensus Clustering

Prefer geographically clustered matches over lone high-inlier outliers:

```python
from collections import defaultdict

CELL_SIZE_DEG = 50 / 111_000   # ~50m in degrees

def cluster_candidates(candidates):
    """Group candidates into 50m grid cells."""
    cells = defaultdict(list)
    for c in candidates:
        cell_lat = round(c["lat"] / CELL_SIZE_DEG)
        cell_lon = round(c["lon"] / CELL_SIZE_DEG)
        cells[(cell_lat, cell_lon)].append(c)

    # Score each cell: total inliers × count bonus
    best_cell = max(cells.values(), key=lambda g: sum(x["inliers"] for x in g) * len(g))
    return max(best_cell, key=lambda x: x["inliers"])
```

---

## Ultra Mode — LoFTR Dense Matching

For blurry or low-texture images, use LoFTR instead of keypoint-based matching:

```python
import kornia.feature as KF
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_inliers(query_path, candidate_path):
    def load_gray_tensor(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        t = torch.from_numpy(img).float()[None, None] / 255.0
        return t.to(device)

    img0 = load_gray_tensor(query_path)
    img1 = load_gray_tensor(candidate_path)

    with torch.no_grad():
        out = loftr({"image0": img0, "image1": img1})

    kps0 = out["keypoints0"].cpu().numpy()
    kps1 = out["keypoints1"].cpu().numpy()

    if len(kps0) < 4:
        return 0

    _, mask = cv2.findHomography(kps0, kps1, cv2.RANSAC, 4.0)
    return int(mask.sum()) if mask is not None else 0
```

---

## Descriptor Hopping (Ultra Mode)

Re-search the index using the matched panorama's clean descriptor instead of the degraded query:

```python
def descriptor_hop(matched_panoid, center_lat, center_lon, radius_km, top_k=500):
    """
    If initial match has <50 inliers, extract a CosPlace descriptor from the
    matched panorama (clean, high-quality) and re-run the index search.
    """
    pano_img = download_panorama(matched_panoid)          # fetch fresh panorama
    crop = rectilinear_crop(pano_img, heading=0, fov=90)  # front-facing crop
    hop_descriptor = extract_descriptor(model, transform, crop)
    return search_index(hop_descriptor, center_lat, center_lon, radius_km, top_k)
```

---

## Building a Large Index (CLI)

For indexing large areas, use the standalone high-performance builder instead of the GUI:

```bash
python build_index.py \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 5.0 \
    --grid-resolution 300 \
    --output-dir ./cosplace_parts
```

Then compile the parts into a searchable index:

```python
# Auto-build runs inside test_super.py on startup,
# or trigger manually:
import numpy as np
import glob

part_files = sorted(glob.glob("cosplace_parts/*.npz"))

all_descs, all_lats, all_lons, all_headings, all_panoids = [], [], [], [], []

for f in part_files:
    data = np.load(f, allow_pickle=True)
    all_descs.append(data["descriptors"])
    all_lats.append(data["lats"])
    all_lons.append(data["lons"])
    all_headings.append(data["headings"])
    all_panoids.append(data["panoids"])

np.save("index/cosplace_descriptors.npy", np.vstack(all_descs))
np.savez("index/metadata.npz",
    lats=np.concatenate(all_lats),
    lons=np.concatenate(all_lons),
    headings=np.concatenate(all_headings),
    panoids=np.concatenate(all_panoids)
)
print(f"Index built: {len(np.vstack(all_descs))} panoramas")
```

---

## Multi-City Index Strategy

All cities share one unified index — radius filtering handles isolation:

```python
# Index Paris (already done), now add London:
# Run Create mode with center=51.5074,-0.1278, radius=5km
# Both cities land in the same cosplace_parts/ and index/ files.

# Search Paris only:
results = search_index(query_desc, center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Search London only:
results = search_index(query_desc, center_lat=51.5074, center_lon=-0.1278, radius_km=5.0)

# No city selection needed — coordinates + radius are the only filter.
```

---

## Confidence Scoring

The final result includes a confidence score based on:

```python
def compute_confidence(top_candidates):
    if not top_candidates:
        return 0.0

    best = top_candidates[0]["inliers"]
    runner_up_diff_location = next(
        (c["inliers"] for c in top_candidates[1:]
         if haversine_km(top_candidates[0]["lat"], top_candidates[0]["lon"],
                         c["lat"], c["lon"]) > 0.1),
        0
    )

    # Uniqueness ratio: how much better is the best vs next different location
    uniqueness = best / (runner_up_diff_location + 1e-6)

    # Geographic clustering: reward if multiple top-10 are near the best
    cluster_count = sum(
        1 for c in top_candidates[:10]
        if haversine_km(top_candidates[0]["lat"], top_candidates[0]["lon"],
                        c["lat"], c["lon"]) < 0.05
    )

    confidence = min(1.0, (uniqueness / 10) * (cluster_count / 3))
    return round(confidence, 3)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI renders blank | macOS bundled tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| CUDA OOM during matching | Too many keypoints | Reduce `max_num_keypoints` (ALIKED: 512, DISK: 512) |
| 0 candidates returned | Radius too small or index empty | Increase radius; verify index was built |
| Low inliers on all candidates | Query image is degraded | Enable Ultra Mode (LoFTR + descriptor hopping) |
| Indexing stalls | Street View API rate limit | Lower grid resolution; indexing auto-resumes on restart |
| Wrong city matched | Radius too large with multi-city index | Tighten radius to city bounds |
| `kornia` import error | Ultra Mode optional dep missing | `pip install kornia` |

---

## Key Constants Reference

```python
# Pipeline defaults (from test_super.py)
TOP_K_CANDIDATES    = 500          # candidates from Stage 1
TOP_HEADING_REFINE  = 15           # candidates re-tested in heading refinement
HEADING_STEP_DEG    = 15           # degrees per heading refinement step
HEADING_RANGE_DEG   = 45           # ±45° heading refinement window
FOV_LIST            = [70, 90, 110]  # fields of view tested per candidate
CLUSTER_CELL_M      = 50           # spatial consensus cell size (meters)
NEIGHBORHOOD_M      = 100          # Ultra Mode neighborhood expansion radius
HOP_INLIER_THRESHOLD = 50          # trigger descriptor hop if below this
GRID_RESOLUTION     = 300          # panorama grid density (do not change)
COSPLACE_DIM        = 512          # descriptor dimensionality
```
