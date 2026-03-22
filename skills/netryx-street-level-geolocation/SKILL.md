```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - index street view panoramas
  - use netryx to locate
  - run netryx geolocation pipeline
  - build a netryx index
  - identify location from street photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls Street View panoramas, indexes them as CosPlace embeddings, then uses a three-stage computer vision pipeline (global retrieval → local feature matching → refinement) to match a query image to a location. Sub-50m accuracy, no internet required at search time, runs entirely on local hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

### Optional: Gemini API key for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### macOS tkinter fix (if GUI appears blank)
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

**GPU backends:** CUDA (NVIDIA) → uses ALIKED; MPS (Apple Silicon) → uses DISK; CPU → uses DISK (slow).

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI application (entry point)
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim global descriptors
    └── metadata.npz               # lat/lon, heading, panoid per descriptor
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI has two modes: **Create** (index an area) and **Search** (geolocate a query image).

---

## Workflow

### Step 1 — Create an Index

Index a geographic area before searching. This crawls Street View panoramas and extracts CosPlace fingerprints.

**In GUI:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) — start with `0.5`–`1` for testing
4. Set grid resolution — default `300`, do not change
5. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is **resumable** — if interrupted, re-run and it picks up where it left off.

**For large datasets, use the standalone builder:**
```bash
python build_index.py
```

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: provide approximate center lat/lon + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score displayed on map

**Enable Ultra Mode** for difficult images (night, blur, low texture). Adds LoFTR dense matching, descriptor hopping, and 100m neighborhood expansion. Slower but more robust.

---

## Pipeline Deep-Dive

### Stage 1 — Global Retrieval (CosPlace)
```
Query image → 512-dim CosPlace descriptor
             + flipped image → 512-dim descriptor
→ Cosine similarity against index (single matrix multiply, <1s)
→ Haversine radius filter
→ Top 500–1000 candidates
```

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
```
For each candidate:
  Download Street View panorama (8 tiles, stitched)
  → Rectilinear crop at indexed heading
  → Multi-FOV crops: 70°, 90°, 110°
  → ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
  → LightGlue deep feature matching vs query keypoints
  → RANSAC geometric verification → inlier count
Best match = highest inlier count
Processes 300–500 candidates in 2–5 minutes
```

### Stage 3 — Refinement
```
Top 15 candidates:
  → Heading refinement: ±45° at 15° steps, 3 FOVs
  → Spatial consensus: cluster into 50m cells
  → Confidence scoring: clustering strength + uniqueness ratio
→ Final GPS coordinates
```

### Ultra Mode extras
```
+ LoFTR detector-free dense matching (handles blur/low-contrast)
+ Descriptor hopping: re-search index using matched panorama's descriptor
+ Neighborhood expansion: search all panoramas within 100m of best match
```

---

## Using CosPlace Utilities Directly

```python
# cosplace_utils.py exposes model loading and descriptor extraction

from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (cached after first call)
model = load_cosplace_model()  # auto-detects CUDA / MPS / CPU

# Extract a 512-dim descriptor from any PIL image
img = Image.open("query.jpg").convert("RGB")
descriptor = get_descriptor(model, img)   # returns np.ndarray shape (512,)
print(descriptor.shape)  # (512,)

# Compare two images via cosine similarity
import numpy as np
desc_a = get_descriptor(model, Image.open("a.jpg").convert("RGB"))
desc_b = get_descriptor(model, Image.open("b.jpg").convert("RGB"))
similarity = np.dot(desc_a, desc_b) / (np.linalg.norm(desc_a) * np.linalg.norm(desc_b))
print(f"Cosine similarity: {similarity:.4f}")
```

---

## Working with the Index Programmatically

```python
import numpy as np

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # shape (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)

lats      = meta["lats"]       # shape (N,)
lons      = meta["lons"]       # shape (N,)
headings  = meta["headings"]   # shape (N,)
panoids   = meta["panoids"]    # shape (N,)  — Street View panorama IDs

print(f"Index contains {len(lats):,} panorama views")
```

### Radius-filtered cosine search
```python
from math import radians, sin, cos, sqrt, atan2
import numpy as np

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """Return top_k candidate indices within radius_km of center."""
    descriptors = np.load("index/cosplace_descriptors.npy")
    meta = np.load("index/metadata.npz", allow_pickle=True)
    lats, lons = meta["lats"], meta["lons"]

    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lat, lon)
        for lat, lon in zip(lats, lons)
    ])
    in_radius = np.where(distances <= radius_km)[0]

    if len(in_radius) == 0:
        return []

    # Cosine similarity
    subset = descriptors[in_radius]
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    norms = np.linalg.norm(subset, axis=1, keepdims=True) + 1e-8
    sims = (subset / norms) @ q

    ranked = in_radius[np.argsort(sims)[::-1][:top_k]]
    return ranked.tolist()

# Usage
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image

model = load_cosplace_model()
query_desc = get_descriptor(model, Image.open("query.jpg").convert("RGB"))
candidates = search_index(query_desc, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
print(f"Found {len(candidates)} candidates")
```

---

## LightGlue Feature Matching Example

```python
import torch
from lightglue import LightGlue, SuperPoint, ALIKED, DISK
from lightglue.utils import load_image, rbd
from PIL import Image
import numpy as np

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Choose extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)

def match_images(img_path_a, img_path_b):
    img_a = load_image(img_path_a).to(device)
    img_b = load_image(img_path_b).to(device)

    with torch.no_grad():
        feats_a = extractor.extract(img_a)
        feats_b = extractor.extract(img_b)
        matches_data = matcher({"image0": feats_a, "image1": feats_b})

    feats_a, feats_b, matches_data = [rbd(x) for x in (feats_a, feats_b, matches_data)]
    matched_kps_a = feats_a["keypoints"][matches_data["matches"][..., 0]]
    matched_kps_b = feats_b["keypoints"][matches_data["matches"][..., 1]]

    return matched_kps_a.cpu().numpy(), matched_kps_b.cpu().numpy()

kps_a, kps_b = match_images("query.jpg", "candidate_crop.jpg")
print(f"Matched keypoints: {len(kps_a)}")
```

### RANSAC geometric verification
```python
import cv2
import numpy as np

def count_geometric_inliers(kps_a, kps_b, threshold=3.0):
    """Returns number of RANSAC inliers — higher = better match."""
    if len(kps_a) < 8:
        return 0
    pts_a = kps_a.astype(np.float32)
    pts_b = kps_b.astype(np.float32)
    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, threshold)
    if mask is None:
        return 0
    return int(mask.sum())

inliers = count_geometric_inliers(kps_a, kps_b)
print(f"Geometric inliers: {inliers}")
# Rule of thumb: >30 inliers = strong match, >100 = high confidence
```

---

## Ultra Mode: LoFTR Dense Matching

```python
import kornia
import torch
import cv2
import numpy as np
from kornia.feature import LoFTR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = LoFTR(pretrained="outdoor").eval().to(device)

def loftr_match(img_path_a, img_path_b):
    def preprocess(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        t = torch.from_numpy(img).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)

    img_a = preprocess(img_path_a)
    img_b = preprocess(img_path_b)

    with torch.no_grad():
        result = loftr({"image0": img_a, "image1": img_b})

    kps_a = result["keypoints0"].cpu().numpy()
    kps_b = result["keypoints1"].cpu().numpy()
    confidence = result["confidence"].cpu().numpy()

    # Filter by confidence
    mask = confidence > 0.5
    return kps_a[mask], kps_b[mask]

kps_a, kps_b = loftr_match("blurry_query.jpg", "candidate_crop.jpg")
print(f"LoFTR matches (conf>0.5): {len(kps_a)}")
```

---

## Street View Panorama Stitching

```python
import requests
from PIL import Image
from io import BytesIO
import numpy as np

def download_streetview_panorama(panoid: str, heading: float, fov: float = 90.0,
                                  width: int = 640, height: int = 480) -> Image.Image:
    """
    Download a rectilinear crop from Google Street View.
    Requires a Google Maps Street View Static API key.
    """
    api_key = os.environ["GOOGLE_MAPS_API_KEY"]
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={width}x{height}"
        f"&pano={panoid}"
        f"&heading={heading}"
        f"&fov={fov}"
        f"&pitch=0"
        f"&key={api_key}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# Multi-FOV crops as used by Netryx pipeline
def get_multi_fov_crops(panoid: str, heading: float):
    crops = {}
    for fov in [70, 90, 110]:
        crops[fov] = download_streetview_panorama(panoid, heading, fov=fov)
    return crops
```

---

## Common Patterns

### Full pipeline in code (manual orchestration)
```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import numpy as np

# 1. Load model and query
model = load_cosplace_model()
query = Image.open("mystery_street.jpg").convert("RGB")
query_flipped = query.transpose(Image.FLIP_LEFT_RIGHT)

desc     = get_descriptor(model, query)
desc_fl  = get_descriptor(model, query_flipped)
combined = (desc + desc_fl) / 2   # average both views

# 2. Search index
candidates = search_index(combined, center_lat=48.8566, center_lon=2.3522, radius_km=3.0)

# 3. For each candidate: download crop, match keypoints, RANSAC
# (see match_images + count_geometric_inliers examples above)

# 4. Pick best by inlier count
best_idx = max(candidates, key=lambda i: get_inliers_for_candidate(i))
meta = np.load("index/metadata.npz", allow_pickle=True)
print(f"Location: {meta['lats'][best_idx]:.6f}, {meta['lons'][best_idx]:.6f}")
```

### Checking device / backend
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    feature_extractor = "aliked"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    feature_extractor = "disk"
else:
    device = torch.device("cpu")
    feature_extractor = "disk"

print(f"Using device: {device}, extractor: {feature_extractor}")
```

---

## Configuration Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | `300` | Panorama crawl density — do not change |
| Top-K candidates | `500`–`1000` | Stage 1 retrieval size |
| Heading refinement range | `±45°` | 15° steps, top 15 candidates |
| Heading refinement FOVs | `[70, 90, 110]` | Degrees |
| Spatial consensus cell | `50 m` | Clustering radius |
| Ultra Mode neighborhood | `100 m` | Expansion radius for descriptor hopping |
| RANSAC threshold | `3.0 px` | Inlier reprojection tolerance |
| Strong match threshold | `30` inliers | Rule of thumb for reliable match |

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # match your actual Python version
```

### CUDA out of memory
- Reduce `max_num_keypoints` in ALIKED: `ALIKED(max_num_keypoints=512)`
- Process candidates in smaller batches
- Use CPU or MPS as fallback

### LightGlue import error
```bash
pip install git+https://github.com/cvg/LightGlue.git
# lightglue is NOT on PyPI — must be installed from GitHub
```

### LoFTR not available
```bash
pip install kornia   # Ultra Mode requires kornia for LoFTR
```

### Index search returns 0 results
- Verify `center_lat`/`center_lon` are within the indexed area
- Increase `radius_km`
- Confirm `index/cosplace_descriptors.npy` and `index/metadata.npz` exist
- Re-run `build_index.py` if the auto-build step was skipped

### Poor match confidence
- Enable **Ultra Mode** (LoFTR + descriptor hopping + neighborhood expansion)
- Increase top-K candidates
- Expand search radius
- Ensure the query image has sufficient texture (avoid pure sky/blank wall crops)

### Indexing stalls / interrupted
Re-run the same index creation command — progress is saved incrementally in `cosplace_parts/*.npz` and resumes automatically.

### Gemini AI Coarse mode fails
```bash
export GEMINI_API_KEY="your_key_here"
# Verify key is valid at https://aistudio.google.com
# Note: Manual mode (explicit lat/lon + radius) is recommended over AI Coarse
```

---

## Models Reference

| Model | Task | Paper |
|-------|------|-------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global place recognition descriptor | CVPR 2022 |
| [ALIKED](https://github.com/naver/alike) | Local keypoints — CUDA | IEEE TIP 2023 |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints — MPS/CPU | NeurIPS 2020 |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | ICCV 2023 |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense detector-free matching (Ultra) | CVPR 2021 |
```
