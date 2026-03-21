---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - use Netryx to locate an image
  - index street view panoramas
  - run Netryx geolocation search
  - set up Netryx locally
  - osint image geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that takes any street-level photograph and returns precise GPS coordinates (sub-50m accuracy). It crawls street-view panoramas into a local index, then uses a three-stage computer vision pipeline — global retrieval (CosPlace) → local feature matching (ALIKED/DISK + LightGlue) → spatial refinement — to match a query image against that index. No cloud API required for searching; only indexing and candidate panorama downloads require internet.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from source)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR support for Ultra Mode
pip install kornia
```

### Platform GPU notes
| Platform | Accelerator | Feature extractor used |
|---|---|---|
| NVIDIA GPU | CUDA | ALIKED (1024 keypoints) |
| Apple Silicon | MPS | DISK (768 keypoints) |
| CPU only | None | DISK (slow) |

### Optional: Gemini API key for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"
```
AI Coarse mode uses Gemini to guess a rough region from visual clues (signs, architecture) when you have zero prior knowledge of where the photo was taken. Manual mode (providing lat/lon + radius) works better in practice.

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11` (match your Python version).

---

## Core Workflow

### 1. Create an Index

Index a geographic area before searching. The indexer crawls Street View panoramas on a grid, extracts CosPlace 512-dim fingerprints, and saves them incrementally to `cosplace_parts/`.

**GUI steps:**
1. Select **Create** mode
2. Enter center lat/lon of the target area
3. Set radius (km) and grid resolution (default 300, don't change)
4. Click **Create Index**

**Indexing time reference:**
| Radius | ~Panoramas | Time (M2 Max) | Disk |
|---|---|---|---|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 h | ~250 MB |
| 5 km | ~30,000 | 8–12 h | ~3 GB |
| 10 km | ~100,000 | 24–48 h | ~7 GB |

Indexing is **resumable** — interrupting and restarting picks up from the last saved chunk.

For large areas, use the standalone high-performance builder:
```bash
python build_index.py
```

### 2. Search for a Photo's Location

**GUI steps:**
1. Select **Search** mode
2. Upload the street-level photo
3. Choose search method:
   - **Manual**: provide center lat/lon + radius (recommended)
   - **AI Coarse**: let Gemini estimate the region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result renders on the map with a confidence score

### 3. Ultra Mode

Enable the **Ultra Mode** checkbox for difficult images (night shots, motion blur, low texture). Adds:
- **LoFTR** dense matching (detector-free, handles blur)
- **Descriptor hopping** (re-searches index using clean matched panorama's descriptor)
- **Neighborhood expansion** (searches all panoramas within 100m of best match)

Ultra Mode is significantly slower but recovers matches the standard pipeline misses.

---

## Project Structure

```
netryx/
├── test_super.py           # Main entry point: GUI + indexing + search pipeline
├── cosplace_utils.py       # CosPlace model loading & descriptor extraction
├── build_index.py          # Standalone CLI index builder for large datasets
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # Stacked 512-dim descriptors for all panoramas
    └── metadata.npz               # lat, lon, heading, panoid per descriptor row
```

---

## Pipeline Deep-Dive

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dim descriptor from the query image
- Also extracts from a horizontally-flipped copy (handles reversed perspectives)
- Cosine similarity search against every descriptor in the index
- Haversine radius filter restricts to the specified area
- Returns top 500–1000 candidates
- Runs in **< 1 second** (single matrix multiply)

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads candidate panoramas from Street View (8 tiles, stitched)
- Crops rectilinear views at the indexed heading
- Generates **3 FOV crops** (70°, 90°, 110°) to handle zoom mismatch
- Extracts local keypoints with ALIKED (CUDA) or DISK (MPS/CPU)
- LightGlue matches query keypoints against candidate keypoints
- RANSAC filters geometrically inconsistent matches (inlier count = match score)
- Processes 300–500 candidates in **2–5 minutes** on modern hardware

### Stage 3 — Refinement
- **Heading refinement**: tests ±45° at 15° steps × 3 FOVs for top 15 candidates
- **Spatial consensus**: clusters matches into 50m cells; clusters beat lone outliers
- **Confidence scoring**: evaluates geographic clustering + uniqueness ratio (best vs. runner-up at a different location)

---

## Code Examples

### Extract a CosPlace descriptor programmatically

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape == (512,)  — float32 numpy array
print("Descriptor shape:", descriptor.shape)
```

### Search the index against a known area (programmatic)

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512) float32
meta        = np.load("index/metadata.npz", allow_pickle=True)
lats        = meta["lats"]     # (N,)
lons        = meta["lons"]     # (N,)
headings    = meta["headings"] # (N,)
panoids     = meta["panoids"]  # (N,)

# Query descriptor (from cosplace_utils)
query_desc = extract_descriptor(model, img, device=device)  # (512,)

# Radius filter
center_lat, center_lon = 48.8566, 2.3522   # Paris
radius_km = 1.0
mask = np.array([haversine_km(center_lat, center_lon, la, lo) <= radius_km
                 for la, lo in zip(lats, lons)])

# Cosine similarity search within radius
filtered_descs = descriptors[mask]             # (M, 512)
sims = filtered_descs @ query_desc             # dot product = cosine sim (if L2-normed)
top_k_local = np.argsort(sims)[::-1][:500]    # top 500 local indices

# Map back to global indices
global_indices = np.where(mask)[0][top_k_local]
print("Top match:", lats[global_indices[0]], lons[global_indices[0]],
      "| heading:", headings[global_indices[0]],
      "| similarity:", sims[top_k_local[0]])
```

### Flip augmentation for better retrieval

```python
from PIL import ImageOps

img_flipped = ImageOps.mirror(img)
desc_orig   = extract_descriptor(model, img,         device=device)
desc_flip   = extract_descriptor(model, img_flipped, device=device)

# Average both descriptors, re-normalize
import numpy as np
combined = (desc_orig + desc_flip) / 2.0
combined /= np.linalg.norm(combined)
```

### LightGlue matching snippet (verification stage)

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="aliked").eval().to(device)

query_img     = load_image("query_photo.jpg").to(device)
candidate_img = load_image("candidate_crop.jpg").to(device)

feats0 = extractor.extract(query_img)
feats1 = extractor.extract(candidate_img)

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]

print(f"Raw matches: {len(matches01['matches'])}")

# RANSAC geometric verification
import cv2
if len(matched_kps0) >= 8:
    pts0 = matched_kps0.cpu().numpy()
    pts1 = matched_kps1.cpu().numpy()
    _, inlier_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"RANSAC inliers: {inlier_count}")
```

### DISK extractor (for MPS/CPU fallback)

```python
from lightglue import DISK

extractor = DISK(max_num_keypoints=768).eval().to(device)
# Usage identical to ALIKED above; pass features="disk" to LightGlue
matcher = LightGlue(features="disk").eval().to(device)
```

### LoFTR dense matching (Ultra Mode)

```python
import kornia.feature as KF
import kornia
import torch
import cv2
import numpy as np

loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def preprocess_for_loftr(img_path, size=(640, 480)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return torch.from_numpy(img).float()[None, None] / 255.0

img0 = preprocess_for_loftr("query_photo.jpg").to(device)
img1 = preprocess_for_loftr("candidate_crop.jpg").to(device)

with torch.no_grad():
    correspondences = loftr({"image0": img0, "image1": img1})

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
conf   = correspondences["confidence"].cpu().numpy()

# Filter by confidence
high_conf = conf > 0.5
mkpts0, mkpts1 = mkpts0[high_conf], mkpts1[high_conf]
print(f"LoFTR matches (conf>0.5): {len(mkpts0)}")
```

---

## Index Management

The index is geography-agnostic. You can index multiple cities into a single unified index:
- Paris (5km radius) + London (10km radius) + Tokyo (3km radius) → one `cosplace_descriptors.npy`
- The radius filter at search time automatically scopes results to the right city
- No city labels or separate files needed

**Index files:**
- `cosplace_parts/*.npz` — raw chunks written during indexing (safe to delete after building)
- `index/cosplace_descriptors.npy` — compiled search index (all descriptors stacked)
- `index/metadata.npz` — parallel arrays: `lats`, `lons`, `headings`, `panoids`

To rebuild the compiled index from parts after adding new areas:
```bash
python build_index.py
```

---

## Confidence Score Interpretation

| Score | Meaning |
|---|---|
| High | Top match has many RANSAC inliers AND geographic clustering of top-N matches around one location |
| Medium | Good inlier count but top matches are geographically scattered |
| Low | Few inliers; result may be incorrect — try Ultra Mode or expand radius |

The uniqueness ratio (best match inliers vs. runner-up inliers at a **different** location) is a strong reliability signal. A ratio > 2× is considered a confident match.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'lightglue'`
LightGlue must be installed from GitHub, not PyPI:
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # match your Python version exactly
```
Then recreate the venv with the Homebrew Python.

### CUDA out of memory
Reduce `max_num_keypoints` in the extractor:
```python
extractor = ALIKED(max_num_keypoints=512).eval().to(device)
```
Or switch to CPU for extraction and only use GPU for LightGlue.

### MPS errors on Apple Silicon
Some kornia/LoFTR operations don't support MPS. Fall back explicitly:
```python
device = torch.device("cpu")  # for LoFTR only
loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)
```

### Index search returns zero candidates
- Verify `center_lat/center_lon` is inside the indexed area
- Increase `radius_km` — the index may be sparser than expected at the edges
- Check that `build_index.py` was run after indexing to compile `cosplace_parts/` into `index/`

### Indexing stalls / resumes incorrectly
The incremental checkpoint is based on already-saved `.npz` filenames in `cosplace_parts/`. Do not rename or move these files between runs.

### Poor match accuracy
1. Enable **Ultra Mode**
2. Increase the candidate pool: edit `top_k` from 500 to 1000 in `test_super.py`
3. Increase index density: re-index with a smaller grid step (lower resolution number = denser)
4. Ensure query image is street-level and facing a consistent direction (not straight up/down)

---

## Key Parameters to Tune

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `top_k` candidates | `test_super.py` | 500 | More candidates = higher recall, slower Stage 2 |
| Grid resolution | GUI / `build_index.py` | 300 | Lower = denser index, more storage |
| Search radius | GUI at search time | user-set | Larger = more candidates, risk false positives |
| `max_num_keypoints` | extractor init | 1024 (ALIKED) / 768 (DISK) | Lower = faster, fewer matches |
| RANSAC threshold | `cv2.findHomography` | 5.0 px | Higher = more permissive, more inliers but more false positives |
| Heading refinement steps | `test_super.py` | ±45° @ 15° | Finer steps find better angle alignment |
| Spatial consensus cell | `test_super.py` | 50m | Smaller = stricter clustering |
