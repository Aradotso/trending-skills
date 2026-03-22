```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - identify location from street photo
  - open source geolocation tool
  - run netryx geolocation pipeline
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, builds a searchable index of visual fingerprints, and matches query images using a three-stage computer vision pipeline (global retrieval → local geometric verification → refinement). No internet presence required for query images — it searches the physical world, not the web.

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

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Environment Variables

```bash
# Optional: Gemini API key for AI-assisted coarse location guessing
export GEMINI_API_KEY="your_key_here"
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

**GPU backends:**
- NVIDIA → CUDA (uses ALIKED for feature extraction)
- Apple Silicon → MPS (uses DISK for feature extraction)
- CPU → supported but slow

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### Step 1: Create an Index

Index a geographic area before searching. The indexer crawls Street View panoramas, extracts 512-dim CosPlace descriptors, and saves them incrementally.

**GUI workflow:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is **resumable** — if interrupted, re-running continues from the last saved chunk.

**For large areas, use the standalone high-performance builder:**

```bash
python build_index.py
```

### Step 2: Search

**GUI workflow:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual cues (signs, architecture, vegetation) to estimate region
4. Click **Run Search** → **Start Full Search**
5. View result on map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # Standalone index builder for large datasets
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search (cosine similarity + haversine radius filter)
    │
    └── Top 500–1000 candidates
    │
    ▼
Per-candidate: Download panorama → Crop at 3 FOVs (70°/90°/110°)
    │
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├── LightGlue deep feature matching
    └── RANSAC geometric verification (inlier count)
    │
    ▼
Heading Refinement (±45° at 15° steps, top 15 candidates)
    │
    ├── Spatial consensus clustering (50m cells)
    └── Confidence scoring (uniqueness ratio + cluster strength)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode

Enable for difficult images (night, motion blur, low texture, low contrast).

**Adds:**
- **LoFTR**: Detector-free dense matching — no keypoints needed, handles blur
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

**When to use Ultra Mode:**
- Image is blurry, dark, or low-contrast
- Standard pipeline returns low confidence (<50 inliers)
- Scene has low texture (plain walls, fog, rain)

---

## Index Design

The index is unified across all indexed cities/areas. Geographic filtering is done at search time via coordinates + radius — no city selection needed.

```
# Index multiple regions into the same index
Paris center (48.8566, 2.3522), radius 5km → adds to index
London center (51.5074, -0.1278), radius 5km → adds to same index

# At search time, radius filter isolates the correct region automatically
Search: center=Paris coords, radius=5km → only returns Paris results
```

**Index files:**
```
index/cosplace_descriptors.npy  # shape: (N, 512), float32
index/metadata.npz              # keys: lat, lon, heading, panoid
```

---

## Using CosPlace Utilities Directly

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
import numpy as np
from PIL import Image

# Load model (auto-detects CUDA/MPS/CPU)
model = load_cosplace_model()

# Extract descriptor from a query image
img = Image.open("query.jpg")
descriptor = extract_descriptor(model, img)          # shape: (512,)

# Extract descriptor from flipped version (catches reversed perspectives)
descriptor_flipped = extract_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT))

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz")
lats, lons = meta["lat"], meta["lon"]

# Cosine similarity search
from numpy.linalg import norm
scores = descriptors @ descriptor / (norm(descriptors, axis=1) * norm(descriptor) + 1e-8)
top_indices = np.argsort(scores)[::-1][:500]

# Haversine radius filter (example: 2km around Paris)
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

center_lat, center_lon = 48.8566, 2.3522
radius_m = 2000

filtered = [
    i for i in top_indices
    if haversine(center_lat, center_lon, lats[i], lons[i]) <= radius_m
]
print(f"Top match: lat={lats[filtered[0]]:.6f}, lon={lons[filtered[0]]:.6f}")
```

---

## LightGlue Feature Matching Example

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Use ALIKED on CUDA, DISK on MPS/CPU (matches Netryx's internal logic)
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)

# Load query and candidate images
query = load_image("query.jpg").to(device)
candidate = load_image("candidate_crop.jpg").to(device)

# Extract features
feats0 = extractor.extract(query)
feats1 = extractor.extract(candidate)

# Match
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]
num_matches = len(matches01["matches"])
print(f"Matches before RANSAC: {num_matches}")

# RANSAC geometric verification
import cv2
import numpy as np

if num_matches >= 4:
    pts0 = matched_kps0.cpu().numpy()
    pts1 = matched_kps1.cpu().numpy()
    _, inlier_mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"Inliers after RANSAC: {inliers}")
```

---

## LoFTR Ultra Mode Example

```python
import torch
import kornia
from kornia.feature import LoFTR
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher = LoFTR(pretrained="outdoor").eval().to(device)

def to_gray_tensor(img_path, size=(640, 480)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return torch.from_numpy(img).float()[None, None] / 255.0

query_t = to_gray_tensor("query.jpg").to(device)
candidate_t = to_gray_tensor("candidate.jpg").to(device)

with torch.no_grad():
    output = matcher({"image0": query_t, "image1": candidate_t})

kps0 = output["keypoints0"].cpu().numpy()
kps1 = output["keypoints1"].cpu().numpy()
confidence = output["confidence"].cpu().numpy()

# Filter by confidence threshold
mask = confidence > 0.5
print(f"LoFTR matches (conf > 0.5): {mask.sum()}")
```

---

## Common Patterns

### Searching a Known City

```python
# Paris — 5km radius
center = (48.8566, 2.3522)
radius_km = 5.0

# London — 3km radius
center = (51.5074, -0.1278)
radius_km = 3.0
```

### Checking GPU Backend

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
    extractor_type = "ALIKED"
elif torch.backends.mps.is_available():
    device = "mps"
    extractor_type = "DISK"
else:
    device = "cpu"
    extractor_type = "DISK"

print(f"Using device: {device}, extractor: {extractor_type}")
```

### Resuming an Interrupted Index Build

Simply re-run the same command — the builder checks `cosplace_parts/` for existing `.npz` chunks and skips already-processed panoramas:

```bash
python test_super.py     # Re-run Create mode with same parameters
# OR for large datasets:
python build_index.py
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank on macOS | Bundled tkinter bug | `brew install python-tk@3.11` |
| `ImportError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ImportError: kornia` | Ultra Mode dependency missing | `pip install kornia` |
| Low confidence results | Heading mismatch or wrong area indexed | Enable Ultra Mode; verify index covers the search area |
| RANSAC returns 0 inliers | Poor candidate or extreme zoom mismatch | Ultra Mode with LoFTR; check multi-FOV crops (70°/90°/110°) |
| MPS device errors on Mac | MPS not supported for all ops | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Index build interrupted | Network / timeout | Re-run — incremental save resumes automatically |
| Gemini AI Coarse not working | Missing API key | `export GEMINI_API_KEY="..."` |
| Out of VRAM | Too many candidates in memory | Reduce candidates to 300; use CPU offloading |

```bash
# MPS fallback for unsupported ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
python test_super.py
```

---

## Key Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| Grid resolution | 300 | Panorama density during indexing — do not change |
| Top candidates | 500–1000 | CosPlace retrieval pool size |
| Heading refinement steps | ±45° at 15° | Applied to top 15 candidates |
| FOV crops | 70°, 90°, 110° | Handles zoom mismatches |
| Spatial clustering cell | 50m | Consensus clustering granularity |
| Neighborhood expansion (Ultra) | 100m | Radius searched around best match |
| RANSAC threshold | 3.0px | Fundamental matrix inlier threshold |
| Confidence threshold (LoFTR) | 0.5 | Filter for LoFTR match confidence |
```
