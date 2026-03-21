```markdown
---
name: netryx-street-level-geolocation
description: Expert in using Netryx, the open-source locally-hosted street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - identify location from photo
  - run netryx geolocation pipeline
  - osint geolocation from street image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a local-first open-source geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes street-view panoramas, then uses a three-stage computer vision pipeline (global retrieval → local geometric verification → refinement) to match a query photo to a physical location with sub-50m accuracy — no landmarks or internet presence required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (deep feature matching)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching (LoFTR)
pip install kornia
```

**macOS tkinter fix** (blank GUI issue):
```bash
brew install python-tk@3.11  # match your Python version
```

**Optional — AI Coarse mode (Gemini):**
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

This opens the main application with two modes: **Create** (index an area) and **Search** (geolocate a photo).

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area before searching. The indexer crawls street-view panoramas and extracts 512-dim CosPlace fingerprints.

**Via GUI:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time reference:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is resumable — interrupted runs continue from the last saved checkpoint.

**For large-scale indexing, use the standalone high-performance builder:**
```bash
python build_index.py
```

### Step 2 — Search (Geolocate a Photo)

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide known approximate coordinates + radius
   - **AI Coarse**: Gemini analyzes the image for visual region clues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

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
    ├── Top 500–1000 candidates ranked by visual similarity
    │
    ▼
Download Panoramas → Crop at 3 FOVs (70°, 90°, 110°)
    │
    ├── ALIKED (CUDA) / DISK (MPS/CPU) keypoint extraction
    ├── LightGlue deep feature matching
    ├── RANSAC geometric verification
    │
    ▼
Heading Refinement (±45°, 15° steps, top 15 candidates)
    │
    ├── Spatial consensus clustering (50m cells)
    ├── Confidence scoring (uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Platform-Specific Feature Extractors

| Platform | Feature Extractor | Keypoints |
|----------|------------------|-----------|
| NVIDIA CUDA | ALIKED | 1024 |
| Apple MPS (M1+) | DISK | 768 |
| CPU | DISK | 768 |

Netryx auto-detects and selects the appropriate extractor.

---

## Ultra Mode

Enable for difficult images: night shots, motion blur, low texture, heavy compression.

**Adds three enhancements:**
1. **LoFTR** — detector-free dense matcher; works without reliable keypoints
2. **Descriptor hopping** — if best match has <50 inliers, extracts CosPlace from the matched panorama (clean image) and re-searches
3. **Neighborhood expansion** — searches all panoramas within 100m of best match

Enable via the **Ultra Mode** checkbox in the GUI before searching.

---

## Index File Structure

```
netryx/
├── test_super.py              # Main GUI application
├── cosplace_utils.py          # CosPlace model + descriptor extraction
├── build_index.py             # Standalone large-scale index builder
├── requirements.txt
├── cosplace_parts/            # Raw embedding chunks (per-run .npz files)
├── index/
│   ├── cosplace_descriptors.npy   # All 512-dim descriptors (matrix)
│   └── metadata.npz               # lat/lon, headings, panorama IDs
└── README.md
```

**Index is city-agnostic** — index Paris, London, and Tel Aviv into the same index. Radius-filtered search by coordinates automatically scopes results. No city selection needed.

---

## Working Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-selects CUDA / MPS / CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load_cosplace_model(device=device)

# Extract 512-dim fingerprint from an image
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
print(descriptor.shape)  # (512,)

# Also extract flipped version for reverse-perspective robustness
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = extract_descriptor(model, img_flipped, device=device)
```

### Search the Index Manually

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]
panoids = meta["panoids"]
headings = meta["headings"]

# Query descriptor (from cosplace_utils)
query_desc = descriptor  # shape (512,)

# Radius filter (e.g., center=Paris, radius=2km)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 2.0

mask = np.array([
    haversine_km(center_lat, center_lon, lat, lon) <= radius_km
    for lat, lon in zip(lats, lons)
])

filtered_descs = descriptors[mask]
filtered_indices = np.where(mask)[0]

# Cosine similarity search
norms = np.linalg.norm(filtered_descs, axis=1, keepdims=True)
filtered_descs_normed = filtered_descs / (norms + 1e-8)
query_normed = query_desc / (np.linalg.norm(query_desc) + 1e-8)

similarities = filtered_descs_normed @ query_normed
top_k = np.argsort(similarities)[::-1][:500]

# Top candidates
for rank, idx in enumerate(top_k[:10]):
    global_idx = filtered_indices[idx]
    print(f"Rank {rank+1}: lat={lats[global_idx]:.6f}, lon={lons[global_idx]:.6f}, "
          f"panoid={panoids[global_idx]}, sim={similarities[idx]:.4f}")
```

### Run LightGlue Feature Matching

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Select extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def extract_features(image_path):
    image = load_image(image_path).to(device)
    return extractor.extract(image)

def match_images(path_query, path_candidate):
    feats0 = extract_features(path_query)
    feats1 = extract_features(path_candidate)
    
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01["matches"]
    
    matched_kpts0 = kpts0[matches[..., 0]]
    matched_kpts1 = kpts1[matches[..., 1]]
    
    return matched_kpts0, matched_kpts1, len(matches)

kpts0, kpts1, n_matches = match_images("query.jpg", "candidate_panorama_crop.jpg")
print(f"Matched keypoints: {n_matches}")
```

### RANSAC Geometric Verification

```python
import cv2
import numpy as np

def ransac_verify(kpts0, kpts1, threshold=3.0):
    """
    Filter matches to geometrically consistent inliers using RANSAC.
    Returns inlier count — higher = more confident match.
    """
    if len(kpts0) < 4:
        return 0
    
    pts0 = kpts0.cpu().numpy() if hasattr(kpts0, 'cpu') else np.array(kpts0)
    pts1 = kpts1.cpu().numpy() if hasattr(kpts1, 'cpu') else np.array(kpts1)
    
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, threshold)
    
    if mask is None:
        return 0
    
    inliers = int(mask.sum())
    return inliers

inlier_count = ransac_verify(kpts0, kpts1)
print(f"RANSAC inliers: {inlier_count}")
# Generally: >100 inliers = strong match, 50-100 = moderate, <50 = weak
```

### Spatial Consensus Clustering

```python
import numpy as np
from collections import defaultdict

def cluster_candidates(candidates, cell_size_m=50):
    """
    Group candidates into spatial cells (~50m).
    Prefer clusters over lone high-inlier outliers.
    
    candidates: list of dicts with keys: lat, lon, inliers
    """
    DEGREES_PER_METER = 1 / 111_000

    clusters = defaultdict(list)
    for c in candidates:
        cell_lat = round(c["lat"] / (cell_size_m * DEGREES_PER_METER))
        cell_lon = round(c["lon"] / (cell_size_m * DEGREES_PER_METER))
        clusters[(cell_lat, cell_lon)].append(c)
    
    # Score each cluster: sum of inliers * cluster size bonus
    best_cluster = None
    best_score = -1
    for cell, members in clusters.items():
        total_inliers = sum(m["inliers"] for m in members)
        score = total_inliers * (1 + 0.1 * len(members))  # size bonus
        if score > best_score:
            best_score = score
            best_cluster = members
    
    # Return best candidate from winning cluster
    return max(best_cluster, key=lambda x: x["inliers"])

# Example
candidates = [
    {"lat": 48.8566, "lon": 2.3522, "inliers": 120},
    {"lat": 48.8567, "lon": 2.3523, "inliers": 95},   # same cluster
    {"lat": 48.9100, "lon": 2.4000, "inliers": 200},  # lone outlier
]
result = cluster_candidates(candidates)
print(f"Best match: {result}")
# Returns the Paris cluster despite lower individual inliers
```

### Ultra Mode — LoFTR Dense Matching

```python
import kornia
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LoFTR
matcher_loftr = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()

def loftr_match(img_path_query, img_path_candidate, max_size=640):
    def load_gray_tensor(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        t = torch.from_numpy(img).float()[None, None] / 255.0
        return t.to(device)
    
    img0 = load_gray_tensor(img_path_query)
    img1 = load_gray_tensor(img_path_candidate)
    
    with torch.inference_mode():
        input_dict = {"image0": img0, "image1": img1}
        correspondences = matcher_loftr(input_dict)
    
    kpts0 = correspondences["keypoints0"].cpu().numpy()
    kpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()
    
    # Filter by confidence
    mask = confidence > 0.5
    return kpts0[mask], kpts1[mask], mask.sum()

kpts0, kpts1, n = loftr_match("blurry_night_query.jpg", "candidate_crop.jpg")
print(f"LoFTR matches (conf>0.5): {n}")
```

---

## Configuration Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | 300 | Panorama density during indexing. Don't change. |
| Top candidates | 500–1000 | Retrieved after CosPlace search |
| Heading refinement range | ±45° at 15° steps | Applied to top 15 candidates |
| FOV crops | 70°, 90°, 110° | Handles zoom mismatch |
| Spatial cluster cell | 50m | For consensus clustering |
| Ultra Mode re-search radius | 100m | Neighborhood expansion |
| LoFTR descriptor hop threshold | 50 inliers | Below this → descriptor hop triggered |
| ALIKED keypoints (CUDA) | 1024 | |
| DISK keypoints (MPS/CPU) | 768 | |

---

## Common Patterns & Tips

### Choosing Search Radius
- **Known country**: 50–100km radius
- **Known city**: 5–10km radius
- **Known neighborhood**: 0.5–2km radius
- Smaller radius = faster search + fewer false positives

### When to Use Ultra Mode
- Night or low-light images
- Motion blur or heavy JPEG compression
- Low-texture environments (fog, snow, featureless walls)
- Weak initial results (<50 inliers on best candidate)

### Index Strategy
- All cities go into **one shared index** — no need to maintain separate indexes
- Search is scoped by coordinates + radius, not by city name
- Run `build_index.py` for large areas (>5km radius) — it's faster than the GUI indexer
- Index is saved incrementally; safe to interrupt and resume

### AI Coarse Mode
- Uses Gemini Vision to analyze architecture, signage, vegetation, vehicles for region estimation
- Outputs approximate coordinates fed to the manual search
- Requires `GEMINI_API_KEY` environment variable
- Recommended only when you have zero prior knowledge of location; manual entry is faster and more reliable when you have any clue

---

## Troubleshooting

### GUI appears blank (macOS)
```bash
brew install python-tk@3.11  # replace 3.11 with your Python version
```

### LightGlue import error
```bash
# Reinstall from source — pip package may be outdated
pip uninstall lightglue -y
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory
- Reduce `max_num_keypoints` in ALIKED (e.g., 512 instead of 1024)
- Disable Ultra Mode (LoFTR is memory-intensive)
- Process fewer candidates (reduce top-K from 500 to 200)

### MPS errors on Apple Silicon
```python
# Force CPU if MPS is unstable
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### Indexing stalls / no panoramas found
- Verify the coordinates are in an area with street-view coverage
- Check internet connectivity (panorama download requires broadband)
- Reduce grid resolution temporarily to test connectivity

### Low confidence results / wrong location
1. Enable Ultra Mode
2. Increase indexed area radius
3. Try a tighter manual search radius if you have location prior
4. Ensure query image is street-level (not aerial, not indoor)

### `cosplace_descriptors.npy` not found
The index must be built before searching. The `cosplace_parts/` chunks are auto-compiled into `index/` when you first run a search. If missing, re-run Create mode or run:
```bash
python build_index.py  # rebuilds index from cosplace_parts/
```

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global place recognition descriptor | All |
| [ALIKED](https://github.com/naver/alike) | Local keypoint extraction | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoint extraction | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense detector-free matching (Ultra) | CUDA / CPU |
| Gemini Vision | AI coarse region estimation (optional) | Cloud API |
```
