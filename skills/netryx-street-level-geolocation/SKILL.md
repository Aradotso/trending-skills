---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to precise GPS coordinates using CosPlace, ALIKED/DISK, and LightGlue locally.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - index street view panoramas
  - run netryx geolocation
  - local visual place recognition
  - match photo to map location
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas into a searchable index, then matches query images against that index using a three-stage computer vision pipeline: **CosPlace** (global retrieval) → **ALIKED/DISK + LightGlue** (geometric verification) → **refinement + confidence scoring**. No cloud APIs needed for inference — everything runs on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required — LightGlue must be installed from GitHub
pip install git+https://github.com/cvg/LightGlue.git

# Optional — Ultra Mode (LoFTR dense matching)
pip install kornia
```

### GPU / Platform Notes

| Platform | Extractor used | Notes |
|----------|---------------|-------|
| NVIDIA CUDA | ALIKED (1024 kp) | Best performance |
| Apple MPS (M1–M4) | DISK (768 kp) | Install `python-tk` via brew for GUI |
| CPU | DISK | Works, significantly slower |

**macOS tkinter fix** (blank GUI):
```bash
brew install python-tk@3.11   # match your Python version
```

### Optional — Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"   # free key from aistudio.google.com
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI has two modes selectable at startup: **Create** (index an area) and **Search** (geolocate a photo).

---

## Project Structure

```
netryx/
├── test_super.py            # Main GUI app — indexing + search
├── cosplace_utils.py        # CosPlace model loading & descriptor extraction
├── build_index.py           # Standalone CLI index builder (large datasets)
├── requirements.txt
├── cosplace_parts/          # Raw embedding chunks written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (searchable)
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Key Workflow

### Step 1 — Create an Index

In the GUI select **Create** mode, or run `build_index.py` for large areas. The indexer:
1. Generates a grid of lat/lon points inside the specified radius
2. Fetches Street View panoramas at each point
3. Extracts a 512-dim CosPlace fingerprint per panorama crop
4. Saves chunks to `cosplace_parts/` (resumable if interrupted)
5. Auto-compiles `index/cosplace_descriptors.npy` + `index/metadata.npz`

**Radius sizing guide:**

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Grid resolution default **300** — do not change unless you know what you're doing.

### Step 2 — Search

1. Select **Search** mode in the GUI
2. Upload any street-level JPEG/PNG
3. Choose search method:
   - **Manual**: supply approximate center lat/lon + radius (km)
   - **AI Coarse**: Gemini reads visual cues (signs, architecture) to estimate region — requires `GEMINI_API_KEY`
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score displayed on map

---

## Pipeline Details (for coding against the internals)

### Loading CosPlace and extracting a descriptor

```python
# cosplace_utils.py exposes these helpers
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else \
         "mps"  if torch.backends.mps.is_available() else "cpu"

model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape == (512,)  float32 numpy array
```

### Searching the index manually

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta        = np.load("index/metadata.npz", allow_pickle=True)
lats        = meta["lats"]       # (N,)
lons        = meta["lons"]       # (N,)
headings    = meta["headings"]   # (N,)
panoids     = meta["panoids"]    # (N,)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon,
                 radius_km=2.0, top_k=500):
    """Return top_k candidate indices ranked by cosine similarity within radius."""
    # Radius mask
    dists = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    mask = dists <= radius_km

    # Cosine similarity (descriptors are L2-normalised)
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    d = descriptors[mask]
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    scores = d @ q                         # (M,)

    local_indices = np.where(mask)[0]
    ranked = local_indices[np.argsort(scores)[::-1][:top_k]]
    return ranked, scores[np.argsort(scores)[::-1][:top_k]]

ranked_idx, ranked_scores = search_index(
    descriptor,
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=1.0,
    top_k=500
)
print(f"Top match: panoid={panoids[ranked_idx[0]]} "
      f"lat={lats[ranked_idx[0]]:.6f} lon={lons[ranked_idx[0]]:.6f} "
      f"score={ranked_scores[0]:.4f}")
```

### Flipped-descriptor trick (catches reversed perspectives)

```python
import torchvision.transforms.functional as TF

img_flipped = TF.hflip(img)
desc_flipped = extract_descriptor(model, img_flipped, device=device)

# Merge: take element-wise max of both similarity vectors
scores_normal  = descriptors_normed @ (descriptor / np.linalg.norm(descriptor))
scores_flipped = descriptors_normed @ (desc_flipped / np.linalg.norm(desc_flipped))
scores_merged  = np.maximum(scores_normal, scores_flipped)
```

### LightGlue matching example (Stage 2)

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use ALIKED on CUDA, DISK on MPS/CPU
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
    matcher   = LightGlue(features="aliked").eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)
    matcher   = LightGlue(features="disk").eval().to(device)

def match_images(path_query, path_candidate):
    img0 = load_image(path_query).to(device)
    img1 = load_image(path_candidate).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    kp0 = feats0["keypoints"][matches01["matches"][..., 0]]
    kp1 = feats1["keypoints"][matches01["matches"][..., 1]]
    return kp0, kp1, matches01["matching_scores0"]

kp0, kp1, scores = match_images("query.jpg", "candidate_crop.jpg")
print(f"Matched {len(kp0)} keypoints")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_inliers(kp0_np, kp1_np, threshold=4.0):
    """Return number of RANSAC inliers. Higher = better geometric match."""
    if len(kp0_np) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        kp0_np, kp1_np,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999
    )
    if mask is None:
        return 0
    return int(mask.ravel().sum())

inliers = ransac_inliers(
    kp0.cpu().numpy(),
    kp1.cpu().numpy()
)
print(f"RANSAC inliers: {inliers}")  # >50 = strong match, >100 = very confident
```

### Ultra Mode — LoFTR dense matching

```python
import kornia.feature as KF
import torch, cv2
import numpy as np

def loftr_match(img_path_a, img_path_b, device):
    matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)

    def load_gray(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        return torch.tensor(img / 255.0, dtype=torch.float32)[None, None].to(device)

    with torch.no_grad():
        out = matcher({"image0": load_gray(img_path_a),
                       "image1": load_gray(img_path_b)})

    kp0 = out["keypoints0"].cpu().numpy()
    kp1 = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()

    # Filter by confidence
    mask = conf > 0.5
    return kp0[mask], kp1[mask]
```

---

## Configuration Patterns

### Index stored globally — one index, many cities

The index accumulates all indexed areas. Radius filtering at search time isolates any city:

```python
# Index Paris
search_index(q, center_lat=48.8566, center_lon=2.3522, radius_km=5)

# Index Tokyo — same index file, different search coordinates
search_index(q, center_lat=35.6762, center_lon=139.6503, radius_km=5)
```

No per-city setup needed. Just index with different center coordinates.

### Multi-FOV crops (70°, 90°, 110°)

Netryx internally tests three FOVs to handle zoom mismatches between query and indexed panorama:

```python
FOVS = [70, 90, 110]   # degrees — used in heading refinement loop

# Heading refinement range
HEADING_OFFSETS = range(-45, 46, 15)   # -45° to +45° in 15° steps
TOP_N_REFINE    = 15                    # candidates refined in Stage 3
```

### Spatial consensus clustering (50 m grid)

```python
def cluster_key(lat, lon, cell_m=50):
    """Snap lat/lon to nearest 50m cell for consensus voting."""
    deg_per_m = 1 / 111_320
    return (
        round(lat / (cell_m * deg_per_m)),
        round(lon / (cell_m * deg_per_m))
    )
```

---

## Common Patterns

### Automate batch geolocation

```python
from pathlib import Path
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import json

model  = load_cosplace_model(device="cuda")
results = []

for img_path in Path("photos/").glob("*.jpg"):
    img  = Image.open(img_path).convert("RGB")
    desc = extract_descriptor(model, img, device="cuda")

    ranked_idx, ranked_scores = search_index(
        desc, center_lat=48.8566, center_lon=2.3522, radius_km=2.0
    )
    best = ranked_idx[0]
    results.append({
        "file":       img_path.name,
        "lat":        float(lats[best]),
        "lon":        float(lons[best]),
        "confidence": float(ranked_scores[0]),
        "panoid":     str(panoids[best]),
    })

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Rebuild index from parts (after interrupted indexing)

```bash
python build_index.py
```

`build_index.py` scans `cosplace_parts/*.npz` and rebuilds `index/cosplace_descriptors.npy` + `index/metadata.npz`. Run this any time after adding new parts or recovering from an interrupted crawl.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GUI opens blank on macOS | System tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| CUDA OOM during matching | Too many keypoints | Reduce `max_num_keypoints` (e.g. 512) |
| Indexing stalls / no panoramas found | Area has no Street View coverage | Try different center coords or larger radius |
| Low confidence scores (<30 inliers) | Query too blurry/dark | Enable Ultra Mode; LoFTR handles low-texture scenes |
| Wrong city match | Radius too large | Tighten `radius_km`; use Manual mode not AI Coarse |
| `cosplace_descriptors.npy` not found | Parts not compiled yet | Run `python build_index.py` |
| MPS errors on Apple Silicon | PyTorch MPS edge case | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |

```bash
# Apple Silicon fallback for unsupported MPS ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
python test_super.py
```

### Confidence score interpretation

| Inlier count | Meaning |
|---|---|
| < 20 | Unreliable — likely false match |
| 20–50 | Weak — treat as approximate area |
| 50–100 | Good — usually within 50 m |
| > 100 | Strong — sub-50 m, high confidence |

---

## References

- [CosPlace (CVPR 2022)](https://arxiv.org/abs/2204.02287)
- [ALIKED (IEEE TIP 2023)](https://arxiv.org/abs/2304.03608)
- [DISK (NeurIPS 2020)](https://arxiv.org/abs/2006.13566)
- [LightGlue (ICCV 2023)](https://arxiv.org/abs/2306.13643)
- [LoFTR (CVPR 2021)](https://arxiv.org/abs/2104.00680)
