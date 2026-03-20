```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - netryx geolocation
  - identify location from street view photo
  - osint image geolocation
  - local geolocation pipeline
  - reverse geolocate image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It builds a local index of street-view panoramas and uses a three-stage computer vision pipeline (global retrieval → local feature matching → refinement) entirely on your own hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue matching library
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching (LoFTR)
pip install kornia
```

### Optional: Gemini API for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"   # Get free key at aistudio.google.com
```

### Requirements
- Python 3.9+ (3.10+ recommended)
- GPU: NVIDIA (CUDA, 4GB+ VRAM) or Apple Silicon (MPS) — CPU works but is slow
- RAM: 8GB minimum, 16GB+ recommended
- Storage: 10GB+ (scales with indexed area)

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### Step 1: Create an Index

Index a geographic area before searching. The index stores CosPlace 512-dim fingerprints for every street-view panorama in the area.

**In GUI:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set search radius (start 0.5–1 km for testing)
4. Set grid resolution (default 300, don't change)
5. Click **Create Index**

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — if interrupted, it picks up where it left off.

### Step 2: Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Enter approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py           # Main GUI application + indexing + search
├── cosplace_utils.py       # CosPlace model loading & descriptor extraction
├── build_index.py          # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks (auto-created)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors
    └── metadata.npz               # Coordinates, headings, panoid IDs
```

---

## Pipeline Deep Dive

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dim fingerprint from query image + horizontally flipped version
- Cosine similarity search against all indexed panoramas
- Haversine radius filter restricts to your search area
- Returns top 500–1000 candidates
- **Speed: <1 second** (single matrix multiply)

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads Street View panoramas (8 tiles, stitched)
- Crops at 3 fields of view: 70°, 90°, 110° (handles zoom mismatch)
- Feature extractor selected by hardware:
  - **CUDA** → ALIKED (1024 keypoints)
  - **MPS/CPU** → DISK (768 keypoints)
- LightGlue deep feature matching + RANSAC geometric verification
- **Speed: 2–5 minutes** for 300–500 candidates

### Stage 3 — Refinement
- Heading refinement: ±45° at 15° steps for top 15 candidates
- Spatial consensus: clusters matches into 50m cells
- Confidence scoring: clustering strength + uniqueness ratio

### Ultra Mode (difficult images)
Enable via **Ultra Mode** checkbox in GUI. Adds:
- **LoFTR**: detector-free dense matching for blurry/night images
- **Descriptor hopping**: re-searches index from matched panorama's descriptor
- **Neighborhood expansion**: searches all panoramas within 100m of best match

---

## Code Examples

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image

model, transform, device = get_cosplace_model()

img = Image.open("street_photo.jpg").convert("RGB")
descriptor = get_descriptor(model, transform, img, device)
# descriptor.shape == (512,)
print(f"Descriptor extracted, device: {device}, shape: {descriptor.shape}")
```

### Build index programmatically (large areas)

```python
# Use build_index.py for large datasets (more efficient than GUI)
# Run as standalone script:
import subprocess
subprocess.run([
    "python", "build_index.py",
    "--lat", "48.8566",
    "--lon", "2.3522",
    "--radius", "2.0",       # km
    "--resolution", "300"
])
```

### Load and query the compiled index

```python
import numpy as np
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
headings = meta["headings"]  # (N,)
panoids = meta["panoids"]    # (N,)

# Extract query descriptor
model, transform, device = get_cosplace_model()
img = Image.open("query.jpg").convert("RGB")
query_desc = get_descriptor(model, transform, img, device)  # (512,)

# Cosine similarity search
from numpy.linalg import norm
similarities = descriptors @ query_desc / (norm(descriptors, axis=1) * norm(query_desc))
top_indices = np.argsort(similarities)[::-1][:500]

# Radius filter (haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

center_lat, center_lon = 48.8566, 2.3522
search_radius_m = 3000  # 3 km

candidates = []
for idx in top_indices:
    dist = haversine(center_lat, center_lon, lats[idx], lons[idx])
    if dist <= search_radius_m:
        candidates.append({
            "idx": idx,
            "lat": lats[idx],
            "lon": lons[idx],
            "heading": headings[idx],
            "panoid": panoids[idx],
            "similarity": similarities[idx]
        })

print(f"Top candidate: {candidates[0]}")
```

### Feature matching with LightGlue (Stage 2 pattern)

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

# Select extractor by device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)

# Load and extract features from both images
image0 = load_image("query.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# Match
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

matched_kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
matched_kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]
print(f"Matched keypoints: {len(matched_kpts0)}")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_inliers(kpts0, kpts1, threshold=3.0):
    """Returns number of RANSAC inliers for a candidate match."""
    if len(kpts0) < 8:
        return 0
    pts0 = kpts0.cpu().numpy()
    pts1 = kpts1.cpu().numpy()
    _, mask = cv2.findFundamentalMat(
        pts0, pts1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999
    )
    if mask is None:
        return 0
    return int(mask.sum())

inliers = ransac_inliers(matched_kpts0, matched_kpts1)
print(f"Verified inliers: {inliers}")
# >30 inliers = strong match, >100 = near-certain match
```

### Ultra Mode: LoFTR dense matching

```python
import kornia
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = kornia.feature.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_match(img0_path, img1_path):
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    
    t0 = torch.from_numpy(img0).float()[None, None] / 255.0
    t1 = torch.from_numpy(img1).float()[None, None] / 255.0
    
    with torch.no_grad():
        result = matcher({"image0": t0.to(device), "image1": t1.to(device)})
    
    kpts0 = result["keypoints0"].cpu().numpy()
    kpts1 = result["keypoints1"].cpu().numpy()
    conf = result["confidence"].cpu().numpy()
    
    # Keep high-confidence matches
    mask = conf > 0.5
    return kpts0[mask], kpts1[mask]
```

---

## Index Management

The index is unified — multiple cities/areas can coexist. Search radius + center coordinates act as the filter.

```
# Index Paris and London into the same index:
# Run Create mode centered on Paris → then London
# Search with Paris center → only Paris results
# Search with London center → only London results
```

**Index files:**
- `cosplace_parts/*.npz` — raw chunks written during indexing (incremental)
- `index/cosplace_descriptors.npy` — compiled descriptor matrix
- `index/metadata.npz` — lat/lon/heading/panoid arrays (same row order as descriptors)

Rebuild compiled index from parts:
```bash
python build_index.py --rebuild-only
```

---

## Search Modes

| Mode | When to use | Requirement |
|------|-------------|-------------|
| Manual | You know approximate region | Enter lat/lon + radius |
| AI Coarse | No prior knowledge of location | `GEMINI_API_KEY` env var |
| Ultra Mode | Blurry, night, low-texture images | `pip install kornia` |

---

## Hardware Performance

| Hardware | Stage 2 Speed (300 candidates) |
|----------|-------------------------------|
| NVIDIA GPU (8GB+) | ~2 min |
| Apple M2/M3/M4 (MPS) | ~3–4 min |
| CPU only | ~20–40 min |

---

## Troubleshooting

**GUI appears blank on macOS:**
```bash
brew install python-tk@3.11   # match your Python version
```

**`import lightglue` fails:**
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — only the GitHub version is supported
```

**CUDA out of memory:**
- Reduce `max_num_keypoints` in the ALIKED extractor (try 512)
- Reduce candidate count from 500 to 300 in search settings

**MPS errors on Mac:**
- Ensure macOS 12.3+ and PyTorch 2.0+
- DISK is used automatically on MPS (not ALIKED) — this is expected

**Index search returns no results:**
- Verify your search center/radius actually overlaps indexed areas
- Run `python build_index.py --rebuild-only` to recompile index from parts

**Low confidence score / wrong location:**
- Enable Ultra Mode
- Expand search radius
- Ensure query image is street-level (not aerial, not interior)
- Try manually providing a more accurate center coordinate

**Indexing interrupted:**
- Simply re-run Create mode with identical parameters — it resumes from `cosplace_parts/`

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| CosPlace | Global 512-dim place fingerprint | All |
| ALIKED | Local keypoint extraction | CUDA only |
| DISK | Local keypoint extraction | MPS / CPU |
| LightGlue | Deep feature matching | All |
| LoFTR | Dense detector-free matching (Ultra) | CUDA / CPU |

---

## Key Accuracy Signals

- **>100 RANSAC inliers**: near-certain correct match
- **30–100 inliers**: strong match
- **<30 inliers**: weak — consider Ultra Mode or wider search
- **Spatial consensus**: multiple candidates clustering in same 50m cell = high confidence
- **Uniqueness ratio**: large gap between best and second-best match = reliable result
```
