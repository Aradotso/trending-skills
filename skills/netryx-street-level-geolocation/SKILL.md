```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, an open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision pipelines.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - reverse image geolocation
  - netryx geolocation
  - index street view area
  - match street photo to location
  - open source geoguessr
---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, indexes them as vector embeddings, and matches query photos against that index using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and spatial refinement.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching
pip install kornia
```

### Platform Requirements

| Platform | GPU Backend | Feature Extractor |
|----------|-------------|-------------------|
| NVIDIA GPU (4GB+ VRAM) | CUDA | ALIKED (1024 keypoints) |
| Apple Silicon (M1–M4) | MPS | DISK (768 keypoints) |
| CPU only | None | DISK (slow) |

### Optional: Gemini API for AI Coarse mode

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

### Step 1 — Create an Index

Index an area before searching. This crawls Street View panoramas and stores CosPlace embeddings.

**In the GUI:**
1. Select **Create** mode
2. Enter center lat/lon of the target area
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Index size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk |
|--------|-----------|---------------|------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — safe to interrupt and restart.

### Step 2 — Search

**In the GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Let Gemini estimate the region from visual clues
4. Click **Run Search** → **Start Full Search**
5. Result appears on map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py           # Main GUI application (entry point)
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (generated during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## How the Pipeline Works

### Stage 1 — Global Retrieval (CosPlace)

```python
# Conceptual usage of cosplace_utils.py
from cosplace_utils import extract_cosplace_descriptor
import numpy as np

# Extract 512-dim fingerprint from query image
descriptor = extract_cosplace_descriptor("query_photo.jpg")           # shape: (512,)
flipped_descriptor = extract_cosplace_descriptor("query_photo.jpg", flip=True)

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")               # shape: (N, 512)
metadata = np.load("index/metadata.npz", allow_pickle=True)

# Cosine similarity search
similarities = descriptors @ descriptor / (
    np.linalg.norm(descriptors, axis=1) * np.linalg.norm(descriptor)
)
top_indices = np.argsort(similarities)[::-1][:500]   # top 500 candidates
```

### Stage 2 — Local Geometric Verification

For each candidate panorama:
1. Download 8 Street View tiles → stitch panorama
2. Crop at indexed heading, 3 FOVs: 70°, 90°, 110°
3. Extract keypoints with ALIKED (CUDA) or DISK (MPS/CPU)
4. Match with LightGlue → RANSAC filter for geometric consistency
5. Count verified inliers → best match = most inliers

### Stage 3 — Refinement

- **Heading refinement**: Test ±45° at 15° steps across top 15 candidates
- **Spatial consensus**: Cluster matches into 50m cells; prefer clusters over isolated outliers
- **Confidence scoring**: Evaluate geographic clustering + uniqueness ratio (best vs. runner-up)

### Ultra Mode

Enable for difficult images (night, blur, low texture):
- **LoFTR**: Detector-free dense matching (handles blur/low-contrast)
- **Descriptor hopping**: Re-search index using descriptor from matched panorama
- **Neighborhood expansion**: Search all panoramas within 100m of best match

---

## Index Architecture

The index is **source-agnostic** and **geographically unified**:

```
# Multiple cities, one index:
# Index Paris → index Tel Aviv → index London
# All stored in cosplace_parts/ + index/

# Search is radius-filtered automatically:
# center=(48.8566, 2.3522), radius=5km → only Paris results
# center=(51.5074, -0.1278), radius=10km → only London results
```

**Data flow:**

```
CREATE:
  Grid points
    → Street View API
    → Panoramas + crops
    → CosPlace embeddings
    → cosplace_parts/*.npz

AUTO-BUILD:
  cosplace_parts/*.npz
    → index/cosplace_descriptors.npy
    → index/metadata.npz

SEARCH:
  Query image
    → CosPlace descriptor
    → Cosine similarity (radius-filtered)
    → Top 500 candidates
    → Download + crop panoramas
    → ALIKED/DISK keypoints
    → LightGlue matching
    → RANSAC verification
    → Heading refinement + spatial consensus
    → GPS result + confidence score
```

---

## Building a Large Index (CLI)

For large areas, use the standalone high-performance builder instead of the GUI:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --resolution 300
```

This writes incremental `.npz` chunks to `cosplace_parts/` and is safe to interrupt.

---

## Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
import torch
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torchvision.transforms as T

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = load_cosplace_model(device=device)

transform = T.Compose([
    T.Resize((480, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

img = Image.open("street_photo.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    descriptor = model(tensor).squeeze().cpu().numpy()   # shape: (512,)

print(f"Descriptor shape: {descriptor.shape}")
print(f"Norm: {(descriptor**2).sum()**0.5:.4f}")
```

### Radius-Filtered Index Search

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
    meta = np.load("index/metadata.npz", allow_pickle=True)
    lats = meta["lats"]
    lons = meta["lons"]
    panoids = meta["panoids"]
    headings = meta["headings"]

    # Radius filter (vectorized haversine approximation)
    dlat = np.radians(lats - center_lat)
    dlon = np.radians(lons - center_lon)
    a = np.sin(dlat/2)**2 + (
        np.cos(np.radians(center_lat)) *
        np.cos(np.radians(lats)) *
        np.sin(dlon/2)**2
    )
    distances_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    mask = distances_km <= radius_km

    filtered_desc = descriptors[mask]
    filtered_lats = lats[mask]
    filtered_lons = lons[mask]
    filtered_panoids = panoids[mask]
    filtered_headings = headings[mask]

    # Cosine similarity
    q = query_descriptor / np.linalg.norm(query_descriptor)
    norms = np.linalg.norm(filtered_desc, axis=1, keepdims=True)
    normed = filtered_desc / np.maximum(norms, 1e-8)
    sims = normed @ q

    top_idx = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "lat": float(filtered_lats[i]),
            "lon": float(filtered_lons[i]),
            "panoid": str(filtered_panoids[i]),
            "heading": float(filtered_headings[i]),
            "similarity": float(sims[i]),
        }
        for i in top_idx
    ]

# Usage
candidates = search_index(
    query_descriptor=descriptor,
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    top_k=500,
)
print(f"Found {len(candidates)} candidates")
print(f"Top match: {candidates[0]}")
```

### LightGlue Feature Matching (Conceptual)

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

def count_inliers(query_path, candidate_path):
    img0 = load_image(query_path).to(device)
    img1 = load_image(candidate_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    matched_kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
    matched_kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]

    # RANSAC geometric verification
    if len(matched_kpts0) >= 8:
        import cv2
        pts0 = matched_kpts0.cpu().numpy()
        pts1 = matched_kpts1.cpu().numpy()
        _, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.99)
        inliers = int(mask.sum()) if mask is not None else 0
    else:
        inliers = 0

    return inliers

inliers = count_inliers("query.jpg", "candidate_crop.jpg")
print(f"Geometric inliers: {inliers}")
```

### Ultra Mode — LoFTR Dense Matching

```python
import kornia
import kornia.feature as KF
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

def loftr_inliers(img0_path, img1_path):
    def preprocess(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        tensor = torch.from_numpy(img).float()[None, None] / 255.0
        return tensor.to(device)

    img0 = preprocess(img0_path)
    img1 = preprocess(img1_path)

    with torch.no_grad():
        result = matcher({"image0": img0, "image1": img1})

    kpts0 = result["keypoints0"].cpu().numpy()
    kpts1 = result["keypoints1"].cpu().numpy()
    conf  = result["confidence"].cpu().numpy()

    # Filter by confidence
    mask = conf > 0.5
    kpts0, kpts1 = kpts0[mask], kpts1[mask]

    if len(kpts0) >= 8:
        _, inlier_mask = cv2.findFundamentalMat(kpts0, kpts1, cv2.FM_RANSAC, 3.0)
        return int(inlier_mask.sum()) if inlier_mask is not None else 0
    return 0

score = loftr_inliers("blurry_query.jpg", "panorama_crop.jpg")
print(f"LoFTR inliers: {score}")
```

---

## Spatial Consensus Clustering

```python
from collections import defaultdict
import numpy as np

def cluster_candidates(candidates, cell_size_m=50):
    """Group candidates into ~50m grid cells; return cluster with most hits."""
    # Convert cell_size_m to approximate degrees
    cell_deg = cell_size_m / 111_000

    clusters = defaultdict(list)
    for c in candidates:
        cell = (
            round(c["lat"] / cell_deg),
            round(c["lon"] / cell_deg),
        )
        clusters[cell].append(c)

    # Find densest cluster
    best_cell = max(clusters, key=lambda k: len(clusters[k]))
    best_cluster = clusters[best_cell]

    # Return best match within cluster (highest inlier count)
    best_match = max(best_cluster, key=lambda c: c.get("inliers", 0))
    return best_match, len(best_cluster)

best, cluster_size = cluster_candidates(verified_candidates)
print(f"Best match: {best['lat']:.6f}, {best['lon']:.6f} "
      f"(cluster size: {cluster_size}, inliers: {best['inliers']})")
```

---

## Confidence Scoring

```python
def compute_confidence(candidates_with_inliers, best_match):
    """
    High confidence = strong best match + no competing cluster nearby.
    """
    best_inliers = best_match["inliers"]

    # Find best match at a different location (>100m away)
    def far_away(c):
        dlat = c["lat"] - best_match["lat"]
        dlon = c["lon"] - best_match["lon"]
        dist_m = ((dlat**2 + dlon**2)**0.5) * 111_000
        return dist_m > 100

    alternatives = [c for c in candidates_with_inliers if far_away(c)]
    runner_up_inliers = max((c["inliers"] for c in alternatives), default=0)

    uniqueness_ratio = best_inliers / max(runner_up_inliers, 1)

    if best_inliers >= 80 and uniqueness_ratio >= 3.0:
        return "HIGH"
    elif best_inliers >= 40 and uniqueness_ratio >= 1.5:
        return "MEDIUM"
    else:
        return "LOW"

confidence = compute_confidence(all_candidates, best)
print(f"Confidence: {confidence}")
```

---

## Common Patterns

### Pattern 1: Index a New City

```bash
# Via GUI: Select Create → enter city center → set radius → Create Index
# Via CLI (large area):
python build_index.py --lat 51.5074 --lon -0.1278 --radius 3.0 --resolution 300
```

### Pattern 2: Geolocate with Known Region

```
GUI: Search → Upload photo → Manual → lat=51.5074, lon=-0.1278, radius=3.0 → Run Search
```

### Pattern 3: Geolocate with No Prior Knowledge

```
GUI: Search → Upload photo → AI Coarse → Run Search
# Gemini analyzes signs, architecture, vegetation to estimate region
# Requires: export GEMINI_API_KEY="..."
```

### Pattern 4: Difficult Image (Night / Blur)

```
GUI: Enable Ultra Mode checkbox → Upload → Search
# Activates: LoFTR + descriptor hopping + 100m neighborhood expansion
```

---

## Troubleshooting

### GUI appears blank on macOS

```bash
brew install python-tk@3.11  # match your Python version
```

### CUDA out of memory

- Reduce `max_num_keypoints` in ALIKED: use 512 instead of 1024
- Reduce candidate count from 500 to 200
- Use CPU for matching if GPU VRAM < 4GB

### Low inlier counts / wrong location

- Enable Ultra Mode
- Widen search radius
- Ensure the target area is indexed (check `cosplace_parts/` has `.npz` files)
- Try flipping the query image horizontally (reverses perspective)

### Index build interrupted

Safe to re-run — incremental `.npz` chunks in `cosplace_parts/` are preserved and the builder resumes from the last checkpoint.

### MPS (Mac) errors

```bash
# Force CPU if MPS is unstable:
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### `lightglue` module not found

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### `kornia` not found (Ultra Mode)

```bash
pip install kornia
```

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim visual descriptor | All |
| [ALIKED](https://github.com/naver/alike) | Local keypoints + descriptors | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints + descriptors | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense detector-free matching (Ultra) | All |
| Gemini | AI coarse region estimation (optional) | API |

---

## Key Design Decisions

- **Single unified index**: All cities share one index; radius filter handles scoping
- **Source agnostic**: Works with Mapillary, KartaView, or any street-view provider
- **Flipped descriptor**: Catches reversed camera perspectives automatically
- **Multi-FOV crops** (70°/90°/110°): Handles zoom mismatch between query and indexed view
- **Spatial consensus over raw inliers**: Prevents single high-inlier false positives
- **Descriptor hopping**: Use clean panorama to re-search when query is degraded
```
