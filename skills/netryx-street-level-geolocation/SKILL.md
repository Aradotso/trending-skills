```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - identify location from photo
  - use netryx to locate
  - index street view panoramas
  - reverse geolocation from image
  - osint geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes Street View panoramas, extracts visual fingerprints using CosPlace, then verifies matches with ALIKED/DISK keypoints and LightGlue deep feature matching. No internet lookups against photos — it searches the physical world by matching against systematically indexed street-view imagery.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (installed from GitHub, not PyPI)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR dense matching for Ultra Mode
pip install kornia
```

### Platform GPU Support

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU | CUDA | Uses ALIKED (1024 keypoints) — fastest |
| Apple Silicon (M1+) | MPS | Uses DISK (768 keypoints) |
| CPU only | CPU | Works but significantly slower |

### Optional: Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI + indexing + search logic
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim visual fingerprints
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix**: `brew install python-tk@3.11` (match your Python version)

---

## Core Workflow

### Step 1: Index an Area

Indexing crawls Street View panoramas in a geographic area and stores CosPlace descriptors.

In the GUI:
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300 — don't change)
4. Click **Create Index**

Indexing is **incremental** — safe to interrupt and resume.

**Time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

### Step 2: Search

In the GUI:
1. Select **Search** mode
2. Upload a street-level photo
3. Choose **Manual** (provide coordinates + radius) or **AI Coarse** (Gemini guesses region)
4. Click **Run Search** → **Start Full Search**
5. Receive GPS coordinates and confidence score on an interactive map

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace → 512-dim global descriptor
    ├── Flipped copy → 512-dim descriptor (catches reversed perspectives)
    │
    ▼
Index Search (cosine similarity, haversine radius filter)
    → Top 500–1000 candidates
    │
    ▼
For each candidate:
    ├── Download Google Street View panorama (8 tiles, stitched)
    ├── Crop at indexed heading
    ├── Multi-FOV crops: 70°, 90°, 110°
    ├── ALIKED (CUDA) / DISK (MPS/CPU) → local keypoints + descriptors
    └── LightGlue → deep feature matching → RANSAC inliers
    │
    ▼
Heading Refinement: top 15 candidates × ±45° × 3 FOVs
    │
    ├── Spatial consensus clustering (50m cells)
    ├── Confidence scoring (clustering density + uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Using CosPlace Utilities Directly

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA / MPS / CPU)
model = load_cosplace_model()

# Extract 512-dim descriptor from an image
img = Image.open("query.jpg")
descriptor = get_descriptor(model, img)  # returns np.ndarray shape (512,)
print(descriptor.shape)  # (512,)
```

---

## Building the Index Programmatically

```python
# build_index.py is a standalone high-performance index builder
# Run directly for large datasets (faster than GUI indexer)
python build_index.py
```

Or trigger index compilation after crawling:

```python
import numpy as np
import os, glob

# Merge all cosplace_parts/*.npz into the searchable index
parts_dir = "cosplace_parts"
index_dir = "index"
os.makedirs(index_dir, exist_ok=True)

all_descriptors = []
all_lats, all_lons, all_headings, all_panoids = [], [], [], []

for fpath in sorted(glob.glob(f"{parts_dir}/*.npz")):
    data = np.load(fpath, allow_pickle=True)
    all_descriptors.append(data["descriptors"])
    all_lats.extend(data["lats"])
    all_lons.extend(data["lons"])
    all_headings.extend(data["headings"])
    all_panoids.extend(data["panoids"])

descriptors = np.vstack(all_descriptors).astype(np.float32)
np.save(f"{index_dir}/cosplace_descriptors.npy", descriptors)
np.savez(
    f"{index_dir}/metadata.npz",
    lats=np.array(all_lats),
    lons=np.array(all_lons),
    headings=np.array(all_headings),
    panoids=np.array(all_panoids, dtype=object),
)
print(f"Index built: {len(all_lats)} panoramas")
```

---

## Searching the Index Programmatically

```python
import numpy as np
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image

def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in km between two GPS points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")          # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats, lons = meta["lats"], meta["lons"]

# Normalize descriptors for cosine similarity
norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
descriptors_norm = descriptors / np.maximum(norms, 1e-8)

# Load model and encode query
model = load_cosplace_model()
query_img = Image.open("query.jpg")
q_desc = get_descriptor(model, query_img)                         # (512,)
q_desc = q_desc / np.maximum(np.linalg.norm(q_desc), 1e-8)

# Also encode flipped image
q_desc_flip = get_descriptor(model, query_img.transpose(Image.FLIP_LEFT_RIGHT))
q_desc_flip = q_desc_flip / np.maximum(np.linalg.norm(q_desc_flip), 1e-8)

# Cosine similarity scores (take max of normal + flipped)
scores = descriptors_norm @ q_desc
scores_flip = descriptors_norm @ q_desc_flip
scores = np.maximum(scores, scores_flip)

# Radius filter (search within 5km of Paris center)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 5.0
mask = np.array([
    haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
    for i in range(len(lats))
])
scores[~mask] = -1.0

# Top 500 candidates
top_indices = np.argsort(scores)[::-1][:500]
print(f"Top match: lat={lats[top_indices[0]]:.6f}, lon={lons[top_indices[0]]:.6f}")
print(f"Similarity score: {scores[top_indices[0]]:.4f}")
```

---

## Ultra Mode

Enable for difficult images (night, blur, low texture, fog).

In GUI: check **Ultra Mode** before running search.

Ultra Mode adds three enhancements:
1. **LoFTR** — detector-free dense matching (handles blur/low-contrast)
2. **Descriptor hopping** — re-searches index using the matched panorama's clean descriptor if initial match < 50 inliers
3. **Neighborhood expansion** — searches all panoramas within 100m of the best match

Requires `kornia` to be installed for LoFTR:
```bash
pip install kornia
```

---

## Multi-Region Index Strategy

The unified index handles multiple cities naturally — no separate indexes needed:

```python
# All cities go into the same index
# Search is scoped by center coordinates + radius at query time

# Index Paris
# (run GUI Create mode with center=48.8566,2.3522, radius=5km)

# Index London  
# (run GUI Create mode with center=51.5074,-0.1278, radius=5km)

# Index Tokyo
# (run GUI Create mode with center=35.6762,139.6503, radius=5km)

# Search Paris only:  center=48.8566,2.3522, radius=5km
# Search London only: center=51.5074,-0.1278, radius=5km
# No manual city selection — radius filtering does the work
```

---

## Common Patterns

### Check which device will be used

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
    print("Using NVIDIA CUDA — ALIKED extractor (1024 keypoints)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple MPS — DISK extractor (768 keypoints)")
else:
    device = "cpu"
    print("Using CPU — slow but functional")
```

### Verify index health

```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Indexed panoramas : {len(meta['lats'])}")
print(f"Descriptor matrix : {descriptors.shape}")
print(f"Lat range         : {meta['lats'].min():.4f} → {meta['lats'].max():.4f}")
print(f"Lon range         : {meta['lons'].min():.4f} → {meta['lons'].max():.4f}")
assert descriptors.shape[0] == len(meta['lats']), "Index mismatch — rebuild required"
```

### Inspect raw part files

```python
import numpy as np, glob

for f in sorted(glob.glob("cosplace_parts/*.npz"))[:3]:
    d = np.load(f, allow_pickle=True)
    print(f"{f}: {len(d['lats'])} panoramas, descriptors shape {d['descriptors'].shape}")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank on macOS | System tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | LightGlue not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ModuleNotFoundError: kornia` | Ultra Mode dependency missing | `pip install kornia` |
| Low confidence / wrong location | Image too degraded or area not indexed | Enable Ultra Mode; verify area is indexed |
| Index mismatch assertion error | Partial build interrupted | Delete `index/` and re-run `build_index.py` |
| CUDA out of memory | Too many candidates in VRAM | Reduce top-K candidates; lower grid resolution |
| MPS slow | Expected — MPS uses DISK not ALIKED | Normal; DISK is slightly slower than ALIKED |
| Indexing hangs | Street View API rate limiting | Normal back-off behavior; leave running |
| Search returns no candidates | Radius too small or area not indexed | Expand radius or re-index target area |

### Rebuild index from parts (after interrupted indexing)

```bash
python build_index.py
```

This re-merges all `cosplace_parts/*.npz` files into a fresh `index/` without re-downloading any panoramas.

---

## Key Dependencies

```
torch          # Core ML framework
torchvision    # Image transforms
Pillow         # Image loading
numpy          # Array operations
scipy          # Spatial clustering
lightglue      # Deep feature matching (install from GitHub)
kornia         # LoFTR dense matching (Ultra Mode, optional)
tkinter        # GUI (stdlib, but needs system package on macOS)
requests       # Street View tile downloads
```

Full list in `requirements.txt`.
```
