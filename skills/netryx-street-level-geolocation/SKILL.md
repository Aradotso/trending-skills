---
name: netryx-street-level-geolocation
description: Local-first street-level geolocation engine using CosPlace + LightGlue to find GPS coordinates from any street photo
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - visual place recognition locally
  - index street view panoramas
  - match photo to real world location
  - run netryx geolocation
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation pipeline that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, builds a searchable visual index using CosPlace embeddings, then verifies matches geometrically with ALIKED/DISK + LightGlue. Sub-50m accuracy, no cloud dependency, runs entirely on your hardware.

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

# Optional — enables Ultra Mode (LoFTR dense matching)
pip install kornia
```

### Optional: Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### macOS tkinter fix (blank GUI)

```bash
brew install python-tk@3.11   # match your Python version
```

---

## Launch the GUI

```bash
python test_super.py
```

This opens the main application with two modes: **Create** (index an area) and **Search** (geolocate a photo).

---

## Project Structure

```
netryx/
├── test_super.py          # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (searchable)
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area before searching. The system crawls Street View panoramas on a grid, extracts CosPlace descriptors, and saves them.

**Via GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Indexing time reference:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hrs       | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hrs      | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hrs     | ~7 GB      |

Indexing is **resumable** — interrupted runs continue from the last saved chunk.

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to guess the region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on a map

---

## Three-Stage Pipeline

### Stage 1 — Global Retrieval (CosPlace)

Extracts a 512-dimensional descriptor from the query image (plus its horizontal flip) and performs cosine similarity search against the index, filtered by haversine radius.

```python
# cosplace_utils.py pattern
import torch
from cosplace_utils import get_cosplace_model, get_descriptor

model = get_cosplace_model()  # loads CosPlace backbone

descriptor = get_descriptor(model, image_tensor)          # shape: (512,)
flipped_desc = get_descriptor(model, flipped_image_tensor)

# Index search — single matrix multiply, sub-1s for any index size
similarities = cosine_similarity(descriptor, index_descriptors)  # (N,)
```

Returns top 500–1000 candidates ranked by visual similarity.

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

For each candidate panorama: download, crop at 3 FOVs (70°, 90°, 110°), extract keypoints, match with LightGlue, filter with RANSAC.

```python
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

# Feature extractor — ALIKED on CUDA, DISK on MPS/CPU
if torch.cuda.is_available():
    extractor = ALIKED(max_num_keypoints=1024).eval().cuda()
else:
    extractor = DISK(max_num_keypoints=768).eval()   # MPS/CPU fallback

matcher = LightGlue(features='aliked').eval()  # or 'disk'

# Extract features from query
feats0 = extractor.extract(query_image)

# For each candidate crop
feats1 = extractor.extract(candidate_crop)
matches01 = matcher({'image0': feats0, 'image1': feats1})
matches01 = rbd(matches01)  # remove batch dimension

matched_kp0 = feats0['keypoints'][matches01['matches'][..., 0]]
matched_kp1 = feats1['keypoints'][matches01['matches'][..., 1]]

# RANSAC geometric verification
inliers = ransac_filter(matched_kp0, matched_kp1)
inlier_count = inliers.sum()
```

Candidate with the most RANSAC inliers wins.

### Stage 3 — Refinement

- **Heading refinement**: Tests ±45° at 15° steps for top 15 candidates across 3 FOVs
- **Spatial consensus**: Clusters matches into 50m cells — prefers clusters over isolated outliers
- **Confidence scoring**: Evaluates geographic clustering and uniqueness ratio (best vs. runner-up)

---

## Ultra Mode

Enable for difficult images (night, motion blur, low texture):

```python
# Ultra Mode adds three enhancements:

# 1. LoFTR — detector-free dense matching (requires kornia)
from kornia.feature import LoFTR
loftr = LoFTR(pretrained='outdoor')
correspondences = loftr({'image0': query_gray, 'image1': candidate_gray})

# 2. Descriptor hopping — re-search using matched panorama's clean descriptor
matched_pano_desc = get_descriptor(model, high_quality_matched_crop)
new_similarities = cosine_similarity(matched_pano_desc, index_descriptors)

# 3. Neighborhood expansion — search all panoramas within 100m of best match
expanded_candidates = filter_by_radius(index, best_match_coords, radius_m=100)
```

---

## Multi-City Index Pattern

All embeddings live in one unified index. Use coordinates + radius to isolate regions — no city selection needed.

```python
# Index Paris
create_index(center=(48.8566, 2.3522), radius_km=5)

# Index London (appends to same index)
create_index(center=(51.5074, -0.1278), radius_km=5)

# Search only Paris — radius filter handles isolation
results = search(query_image, center=(48.8566, 2.3522), radius_km=5)

# Search only London
results = search(query_image, center=(51.5074, -0.1278), radius_km=5)
```

---

## Standalone Index Builder (Large Datasets)

For areas > 5km radius, use `build_index.py` directly instead of the GUI:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 10 \
  --resolution 300
```

This compiles `cosplace_parts/*.npz` chunks into the final `index/cosplace_descriptors.npy` + `index/metadata.npz`. Run it after any new indexing session to make chunks searchable.

---

## Hardware / Device Behavior

| Feature          | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU      |
|------------------|--------------|---------------------|----------|
| Feature extractor| ALIKED 1024kp| DISK 768kp          | DISK     |
| Speed (300 cands)| ~2 min       | ~3–4 min            | ~15+ min |
| Ultra/LoFTR      | Full support | Partial             | Slow     |
| Recommended VRAM | 8GB+         | Unified 16GB+       | N/A      |

```python
# Device detection pattern used internally
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    extractor_name = 'aliked'
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    extractor_name = 'disk'
else:
    device = torch.device('cpu')
    extractor_name = 'disk'
```

---

## Common Patterns

### Load and query the index programmatically

```python
import numpy as np
import torch
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torchvision.transforms as T

# Load index
descriptors = np.load('index/cosplace_descriptors.npy')   # (N, 512)
meta = np.load('index/metadata.npz', allow_pickle=True)
lats = meta['lats']    # (N,)
lons = meta['lons']    # (N,)
headings = meta['headings']  # (N,)
panoids = meta['panoids']    # (N,)

# Load model
model = get_cosplace_model().eval()

# Prepare query image
transform = T.Compose([T.Resize((512, 512)), T.ToTensor(),
                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
img = Image.open('query.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0)

# Extract descriptor
with torch.no_grad():
    desc = get_descriptor(model, tensor)  # (512,)

# Cosine similarity search
desc_norm = desc / np.linalg.norm(desc)
idx_norm = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
sims = idx_norm @ desc_norm          # (N,)
top_k = np.argsort(sims)[::-1][:500]

print(f"Top match: {lats[top_k[0]]:.6f}, {lons[top_k[0]]:.6f} "
      f"(sim={sims[top_k[0]]:.3f})")
```

### Haversine radius filter

```python
import numpy as np

def haversine_filter(lats, lons, center_lat, center_lon, radius_km):
    """Returns boolean mask of points within radius_km of center."""
    R = 6371.0
    dlat = np.radians(lats - center_lat)
    dlon = np.radians(lons - center_lon)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(center_lat)) * np.cos(np.radians(lats))
         * np.sin(dlon/2)**2)
    dist = 2 * R * np.arcsin(np.sqrt(a))
    return dist <= radius_km

# Usage
mask = haversine_filter(lats, lons,
                         center_lat=48.8566, center_lon=2.3522,
                         radius_km=2.0)
filtered_descriptors = descriptors[mask]
filtered_indices = np.where(mask)[0]
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank/white | macOS tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| LoFTR import fails | kornia missing | `pip install kornia` |
| MPS slow / crashes | Large batch on Apple Silicon | Reduce `max_num_keypoints` to 512 |
| Index search returns 0 results | Radius too small or wrong coordinates | Increase radius, verify lat/lon order |
| Low inlier count (<20) | Query photo too blurry / unusual FOV | Enable Ultra Mode; try wider radius |
| `index/` folder missing or empty` | Index not compiled after creation | Run `python build_index.py` to compile chunks |
| Interrupted indexing resumes wrong area | Stale `cosplace_parts/` chunks | Delete `cosplace_parts/` and restart for clean area |
| CUDA OOM | Too many candidates in memory | Reduce top-K candidates from 1000 → 300 in search settings |
| Gemini AI Coarse returns wrong country | Poor visual clues in image | Switch to Manual mode with known approximate coordinates |

---

## Key Parameters Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | 300 | Higher = denser coverage, don't change |
| Top-K candidates | 500–1000 | From Stage 1 retrieval |
| RANSAC inlier threshold | ~20 | Below this = unreliable match |
| Heading refinement range | ±45° @ 15° steps | Applied to top 15 candidates |
| FOVs tested | 70°, 90°, 110° | Handles zoom mismatches |
| Spatial consensus cell | 50m | Clustering granularity |
| Ultra neighborhood expansion | 100m | Around best match coords |
