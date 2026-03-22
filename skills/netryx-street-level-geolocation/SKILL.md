---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision pipelines.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - run netryx search
  - visual place recognition locally
  - identify location from street photo
---

# Netryx Street-Level Geolocation Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes street-view panoramas, extracts visual fingerprints with CosPlace, then verifies matches using ALIKED/DISK keypoint extraction and LightGlue deep feature matching — achieving sub-50m accuracy with no internet presence required for the query image.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must install from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR dense matching for Ultra Mode
pip install kornia
```

### Platform GPU Support

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU | CUDA | Best performance; uses ALIKED (1024 keypoints) |
| Apple Silicon (M1+) | MPS | Good performance; uses DISK (768 keypoints) |
| CPU only | CPU | Works but slow; uses DISK |

### Optional: Gemini API for AI Coarse Mode

```bash
export GEMINI_API_KEY="your_key_here"   # Get free key at aistudio.google.com
```

---

## Launch

```bash
# Launch the main GUI (indexing + search)
python test_super.py
```

> **macOS blank GUI fix**: `brew install python-tk@3.11` (match your Python version)

---

## Core Workflow

### Step 1: Create an Index

Index a geographic area before searching. The indexer crawls Street View panoramas, extracts CosPlace 512-dim fingerprints, and saves them to `cosplace_parts/`.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon of the area
3. Set radius (km) and grid resolution (default 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Storage |
|--------|-----------|---------------|---------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — interrupting and restarting picks up where it left off.

**For large datasets, use the standalone high-performance builder:**
```bash
python build_index.py
```

### Step 2: Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate lat/lon + radius (fastest, most accurate)
   - **AI Coarse**: Let Gemini estimate region from visual clues (no prior knowledge needed)
4. Click **Run Search** → **Start Full Search**

---

## Pipeline Architecture

```
Query Image
    │
    ├─ CosPlace descriptor (512-dim fingerprint)
    ├─ Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search (cosine similarity + haversine radius filter)
    │
    └─ Top 500–1000 candidates
    │
    ▼
Download panoramas → Crop at 3 FOVs (70°, 90°, 110°)
    │
    ├─ ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├─ LightGlue deep feature matching
    ├─ RANSAC geometric verification
    │
    ▼
Heading Refinement (±45°, 15° steps, top 15 candidates)
    │
    ├─ Spatial consensus clustering (50m cells)
    ├─ Confidence scoring (uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## File Structure

```
netryx/
├── test_super.py              # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py          # CosPlace model loading + descriptor extraction
├── build_index.py             # Standalone index builder for large datasets
├── requirements.txt
├── cosplace_parts/            # Raw embedding chunks (auto-created during indexing)
│   └── *.npz
└── index/                     # Compiled searchable index (auto-built)
    ├── cosplace_descriptors.npy    # All 512-dim descriptors (matrix)
    └── metadata.npz                # Lat/lon, headings, panorama IDs
```

---

## Code Examples

### Extract a CosPlace Descriptor from Any Image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model, device = load_cosplace_model()

# Extract 512-dim fingerprint from a query image
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device)
# descriptor.shape == (512,)
print(f"Descriptor shape: {descriptor.shape}, device: {device}")
```

### Search the Index Programmatically

```python
import numpy as np

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
panoids = meta["panoids"]
headings = meta["headings"]

# Cosine similarity search
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

model, device = load_cosplace_model()
query = Image.open("query_photo.jpg").convert("RGB")
q_desc = extract_descriptor(model, query, device)  # (512,)

# Normalize and compute similarities
q_norm = q_desc / np.linalg.norm(q_desc)
d_norm = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
similarities = d_norm @ q_norm  # (N,)

# Radius filter (haversine) — restrict to Paris 5km
CENTER_LAT, CENTER_LON, RADIUS_KM = 48.8566, 2.3522, 5.0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

distances = haversine_km(CENTER_LAT, CENTER_LON, lats, lons)
in_radius = distances <= RADIUS_KM

# Get top-500 candidates within radius
masked_sim = np.where(in_radius, similarities, -1.0)
top_indices = np.argsort(masked_sim)[::-1][:500]

print("Top 5 candidate locations:")
for idx in top_indices[:5]:
    print(f"  panoid={panoids[idx]}, lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, "
          f"heading={headings[idx]}, sim={similarities[idx]:.4f}")
```

### Run Full Geolocation on a Single Image (Programmatic)

```python
# Netryx's full pipeline is exposed through the GUI in test_super.py,
# but you can invoke the search logic directly if you understand the internals.
# The main search function is typically triggered via the GUI's "Start Full Search" button.
# For scripted use, patch into test_super.py's search coroutine:

import asyncio
import sys
sys.argv = ["test_super.py"]  # suppress tkinter if headless not needed

# Recommended approach: use subprocess for clean separation
import subprocess
result = subprocess.run(
    ["python", "test_super.py"],
    capture_output=False  # GUI must be interactive
)
```

### Ultra Mode — Dense Matching for Difficult Images

Ultra Mode is toggled via the GUI checkbox. It adds three enhancements for blurry, dark, or low-texture images:

1. **LoFTR** — detector-free dense matcher (requires `kornia`)
2. **Descriptor hopping** — re-searches using clean matched panorama's descriptor
3. **Neighborhood expansion** — searches all panoramas within 100m of best match

```python
# Verify LoFTR is available before enabling Ultra Mode
try:
    import kornia
    from kornia.feature import LoFTR
    loftr = LoFTR(pretrained="outdoor")
    print("LoFTR available — Ultra Mode supported")
except ImportError:
    print("Install kornia for Ultra Mode: pip install kornia")
```

### Check Index Health

```python
import numpy as np
import os

index_dir = "index"
parts_dir = "cosplace_parts"

# Check compiled index
if os.path.exists(f"{index_dir}/cosplace_descriptors.npy"):
    descs = np.load(f"{index_dir}/cosplace_descriptors.npy")
    meta = np.load(f"{index_dir}/metadata.npz", allow_pickle=True)
    print(f"Index: {descs.shape[0]} panoramas indexed")
    print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
    print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
else:
    print("No compiled index found — run Create Index first")

# Count raw parts
if os.path.exists(parts_dir):
    parts = [f for f in os.listdir(parts_dir) if f.endswith(".npz")]
    print(f"Raw embedding chunks: {len(parts)} files in cosplace_parts/")
```

---

## Multi-City Index Pattern

All cities share a single index. The radius filter at search time isolates the correct region — no per-city files needed.

```python
# Index Paris (5km radius)
# → GUI: Create mode, lat=48.8566, lon=2.3522, radius=5

# Index London (10km radius)  
# → GUI: Create mode, lat=51.5074, lon=-0.1278, radius=10

# Index Tel Aviv (3km radius)
# → GUI: Create mode, lat=32.0853, lon=34.7818, radius=3

# All stored in the same index/cosplace_descriptors.npy
# Search with center=Paris + radius=5km → only returns Paris panoramas
# Search with center=London + radius=10km → only returns London panoramas
```

---

## Troubleshooting

### GUI is blank on macOS
```bash
brew install python-tk@3.11   # Match your exact Python version
# Then relaunch:
python test_super.py
```

### LightGlue import error
```bash
# Must install from GitHub, not PyPI
pip uninstall lightglue -y
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory
```python
# Reduce keypoints in the extractor config within test_super.py
# ALIKED default: 1024 keypoints — try reducing to 512
# Look for: extractor = ALIKED(max_num_keypoints=1024)
# Change to: extractor = ALIKED(max_num_keypoints=512)
```

### Indexing stalled / resuming after interruption
```bash
# Safe to Ctrl+C and restart — cosplace_parts/ stores progress incrementally
python test_super.py   # Resume from GUI → Create mode → same coordinates
```

### Poor match accuracy (< 50 inliers on best candidate)
- Enable **Ultra Mode** checkbox for LoFTR + descriptor hopping
- Increase search radius — the correct panorama may be just outside your radius
- Ensure the photo is street-level (not aerial or indoor)
- Try a tighter radius if you have approximate knowledge of the location

### No matches found / empty results
```python
# Verify your index covers the search area
import numpy as np
meta = np.load("index/metadata.npz", allow_pickle=True)
lats, lons = meta["lats"], meta["lons"]

# Check if your search center is within indexed bounds
search_lat, search_lon = 48.8566, 2.3522
lat_ok = lats.min() <= search_lat <= lats.max()
lon_ok = lons.min() <= search_lon <= lons.max()
print(f"Search center in index bounds: lat={lat_ok}, lon={lon_ok}")
print(f"Index bounds: lat [{lats.min():.4f}, {lats.max():.4f}], "
      f"lon [{lons.min():.4f}, {lons.max():.4f}]")
```

### MPS (Apple Silicon) errors
```bash
# Ensure PyTorch with MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
# If False: pip install --upgrade torch torchvision
```

---

## Key Configuration Reference

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| Grid resolution | GUI / Create mode | 300 | Higher = denser coverage; don't change |
| Search radius | GUI / Search mode | User-defined | In km from center point |
| Top candidates | Internal | 500–1000 | Candidates passed to Stage 2 |
| FOV crops | Internal | 70°, 90°, 110° | Handles zoom mismatches |
| Heading refinement range | Internal | ±45° at 15° steps | Applied to top 15 candidates |
| Consensus cell size | Internal | 50m | Spatial clustering grid |
| Ultra neighborhood | Internal | 100m | Expansion radius in Ultra Mode |
| ALIKED keypoints (CUDA) | Internal | 1024 | Reduce to 512 if OOM |
| DISK keypoints (MPS/CPU) | Internal | 768 | |

---

## Models Reference

| Model | Role | Install |
|-------|------|---------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global retrieval (512-dim fingerprint) | Auto via requirements.txt |
| [ALIKED](https://github.com/naver/alike) | Keypoints on CUDA | Auto via requirements.txt |
| [DISK](https://github.com/cvlab-epfl/disk) | Keypoints on MPS/CPU | Auto via LightGlue install |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | `pip install git+https://github.com/cvg/LightGlue.git` |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense matching (Ultra Mode) | `pip install kornia` |
