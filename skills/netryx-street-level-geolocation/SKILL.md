---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to GPS coordinates using CosPlace + LightGlue computer vision pipeline.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - run netryx search
  - identify location from photo
  - osint image geolocation
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies GPS coordinates from any street-level photograph. It indexes Google Street View panoramas into a searchable fingerprint database and uses a three-stage CV pipeline (CosPlace → ALIKED/DISK + LightGlue → RANSAC refinement) to match a query image to a precise location. Sub-50m accuracy. No landmarks required. Runs entirely on local hardware.

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

# Optional: LoFTR dense matcher for Ultra Mode
pip install kornia
```

**GPU support auto-detected:**
- NVIDIA GPU → CUDA (uses ALIKED, 1024 keypoints)
- Apple Silicon → MPS (uses DISK, 768 keypoints)
- No GPU → CPU (slower)

**Optional Gemini API key** for AI Coarse geolocation mode:
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### 1. Create an Index (Required First Step)

The index stores 512-dim CosPlace fingerprints for every crawled panorama in a geographic area.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon, radius (km), grid resolution (default: 300)
3. Click **Create Index** — saves incrementally to `cosplace_parts/`

**Index size reference:**

| Radius | ~Panoramas | Build Time (M2 Max) | Storage |
|--------|-----------|---------------------|---------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Interrupted builds resume automatically on next run.

**Auto-build compiled index** (runs after Create or manually):
```
cosplace_parts/*.npz  →  index/cosplace_descriptors.npy
                      →  index/metadata.npz
```

### 2. Search for a Location

**Via GUI:**
1. Select **Search** mode
2. Upload street-level photo
3. Choose method:
   - **Manual**: Provide center lat/lon + radius if you know the approximate region
   - **AI Coarse**: Gemini analyzes signs/architecture to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Real-time candidate visualization → GPS result + confidence score on map

---

## Pipeline Architecture

```
Query Image
    │
    ├─ CosPlace 512-dim descriptor
    ├─ Flipped descriptor (handles mirrored perspectives)
    │
    ▼
Cosine similarity search → radius filter (haversine) → Top 500–1000 candidates
    │                                                    (<1 second)
    ▼
Download panoramas → Rectilinear crops at 3 FOVs (70°, 90°, 110°)
    │
    ├─ ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├─ LightGlue deep feature matching
    ├─ RANSAC geometric verification
    │                                                    (2–5 min)
    ▼
Heading refinement: ±45° at 15° steps, top 15 candidates
    │
    ├─ Spatial consensus clustering (50m cells)
    ├─ Confidence scoring (uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode (Difficult Images)

Enable **Ultra Mode** checkbox for night shots, motion blur, low-texture scenes.

Adds three extra passes:
1. **LoFTR** — detector-free dense matching (no keypoints needed, handles blur)
2. **Descriptor hopping** — if best match has <50 inliers, extract CosPlace from the *matched clean panorama* and re-search index
3. **Neighborhood expansion** — searches all panoramas within 100m of best match

Significantly slower; use when standard pipeline returns low confidence.

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (stacked)
    └── metadata.npz               # lat, lon, heading, panoid per descriptor
```

---

## Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = load_cosplace_model(device=device)

img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape → (512,)
```

### Load the Index and Run a Radius-Filtered Cosine Search

```python
import numpy as np

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta        = np.load("index/metadata.npz")
lats        = meta["lats"]      # (N,)
lons        = meta["lons"]      # (N,)
headings    = meta["headings"]  # (N,)
panoids     = meta["panoids"]   # (N,)

# Query descriptor (from extract_descriptor above)
query_vec = descriptor / np.linalg.norm(descriptor)

# Cosine similarity — single matrix multiply
norms     = np.linalg.norm(descriptors, axis=1, keepdims=True)
normed    = descriptors / (norms + 1e-8)
scores    = normed @ query_vec                              # (N,)

# Haversine radius filter
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

center_lat, center_lon = 48.8566, 2.3522   # Paris example
radius_km = 2.0

distances = haversine_km(lats, center_lon, center_lat, center_lon)
# Vectorised version:
distances = haversine_km(lats, lons, center_lat, center_lon)
mask      = distances <= radius_km

filtered_scores  = scores.copy()
filtered_scores[~mask] = -1.0

top_k = 500
top_idx = np.argsort(filtered_scores)[::-1][:top_k]

print("Top match:")
print(f"  lat={lats[top_idx[0]]:.6f}, lon={lons[top_idx[0]]:.6f}")
print(f"  heading={headings[top_idx[0]]}, panoid={panoids[top_idx[0]]}")
print(f"  score={filtered_scores[top_idx[0]]:.4f}")
```

### Run LightGlue Matching Between Query and Candidate Crop

```python
import torch
from PIL import Image
import numpy as np
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Select extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def match_images(path_query: str, path_candidate: str) -> int:
    """Returns number of RANSAC-verified inliers (higher = better match)."""
    img0 = load_image(path_query).to(device)
    img1 = load_image(path_candidate).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
    kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]

    if len(kpts0) < 8:
        return 0

    # RANSAC geometric verification
    import cv2
    pts0 = kpts0.cpu().numpy()
    pts1 = kpts1.cpu().numpy()
    _, inlier_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return inliers

inliers = match_images("query_photo.jpg", "candidate_crop.jpg")
print(f"Verified inliers: {inliers}")   # >30 = good match, >80 = strong match
```

### Use LoFTR (Ultra Mode) for Blurry/Low-Texture Images

```python
import kornia
import torch
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr  = kornia.feature.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_inliers(img0_path: str, img1_path: str) -> int:
    def preprocess(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        tensor = torch.from_numpy(img).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)

    with torch.no_grad():
        result = loftr({"image0": preprocess(img0_path),
                        "image1": preprocess(img1_path)})

    kpts0 = result["keypoints0"].cpu().numpy()
    kpts1 = result["keypoints1"].cpu().numpy()

    if len(kpts0) < 8:
        return 0

    _, mask = cv2.findFundamentalMat(kpts0, kpts1, cv2.RANSAC, 3.0)
    return int(mask.sum()) if mask is not None else 0

score = loftr_inliers("blurry_query.jpg", "candidate_crop.jpg")
print(f"LoFTR inliers: {score}")
```

---

## Multi-Region Index Strategy

All areas share one unified index. Radius filtering at search time handles isolation:

```python
# Index Paris (run once)
# GUI: Create → lat=48.8566, lon=2.3522, radius=5km

# Index London (run once, appends to same index)
# GUI: Create → lat=51.5074, lon=-0.1278, radius=5km

# Search Paris only — London results excluded by radius filter
# GUI: Search → Manual → lat=48.8566, lon=2.3522, radius=5km

# Search London only
# GUI: Search → Manual → lat=51.5074, lon=-0.1278, radius=5km
```

No per-city files — coordinates + radius handle everything automatically.

---

## Common Patterns

### Build Index Headlessly (Large Areas)

Use `build_index.py` for large datasets without the GUI:

```bash
python build_index.py --lat 48.8566 --lon 2.3522 --radius 5 --resolution 300
```

### Confidence Scoring Interpretation

| Inliers | Confidence | Interpretation |
|---------|-----------|----------------|
| >100 | Very High | Near-certain match |
| 50–100 | High | Reliable, typical result |
| 20–50 | Medium | Plausible, verify visually |
| <20 | Low | Uncertain — try Ultra Mode |

### Spatial Consensus Check

```python
from collections import Counter

def cluster_candidates(lats, lons, top_idx, cell_meters=50):
    """Group candidates into 50m cells; prefer largest cluster."""
    cell_deg = cell_meters / 111320.0
    cells = Counter()
    for i in top_idx:
        cell = (round(lats[i] / cell_deg), round(lons[i] / cell_deg))
        cells[cell] += 1
    best_cell, count = cells.most_common(1)[0]
    return best_cell, count
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GUI appears blank on macOS | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ModuleNotFoundError: kornia` | `pip install kornia` (Ultra Mode only) |
| MPS out of memory | Lower keypoint count: `DISK(max_num_keypoints=512)` |
| CUDA OOM | Reduce candidates from 500 to 200 in GUI settings |
| Indexing stalled | Kill process — restart resumes from last `.npz` chunk |
| Low confidence (<20 inliers) | Enable Ultra Mode; widen search radius; check query FOV |
| No panoramas found | Area may lack Street View coverage; try Mapillary/KartaView data |
| AI Coarse mode fails | Check `echo $GEMINI_API_KEY`; use Manual mode instead |
| Wrong region matched | Tighten radius filter; ensure index covers the correct area |

---

## Key Dependencies

```
torch          # PyTorch (CUDA/MPS/CPU)
torchvision
Pillow
numpy
opencv-python  # RANSAC geometric verification
lightglue      # from GitHub (see install)
kornia         # optional, for LoFTR Ultra Mode
requests       # Street View tile fetching
tkinter        # GUI (system package on macOS)
google-generativeai  # optional, for Gemini AI Coarse mode
```
