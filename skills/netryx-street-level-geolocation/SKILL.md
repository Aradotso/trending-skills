---
name: netryx-street-level-geolocation
description: Local-first street-level geolocation engine using CosPlace, ALIKED/DISK, and LightGlue to identify GPS coordinates from street photos with sub-50m accuracy.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - index street view panoramas
  - use netryx to locate
  - run geolocation search on image
  - build a street view index
  - identify location from street photo
---

# Netryx Street-Level Geolocation Engine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes street-view panoramas, extracts visual fingerprints using CosPlace, and matches query images through ALIKED/DISK keypoint extraction and LightGlue deep feature matching — achieving sub-50m accuracy without relying on landmarks or internet image search.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must install from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

**macOS tkinter fix** (if GUI renders blank):
```bash
brew install python-tk@3.11  # match your Python version
```

**Gemini API key** (optional, for AI Coarse location guessing):
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface for all operations: indexing, searching, and viewing results on a map.

---

## Core Workflow

### Step 1 — Create an Index

Index an area by crawling street-view panoramas and storing CosPlace fingerprints.

**In the GUI:**
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is resumable — interrupted runs pick up from where they left off.

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result appears on map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI application (indexing + search)
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## Three-Stage Pipeline

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dim descriptor from the query image (plus a flipped version)
- Cosine similarity search against the full index, filtered by haversine radius
- Returns top 500–1000 candidates in under 1 second (single matrix multiply)

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads panorama tiles from Street View, stitches them, crops at the indexed heading
- Generates multi-FOV crops at 70°, 90°, and 110° to handle zoom mismatches
- Extracts local keypoints: **ALIKED** on CUDA, **DISK** on MPS/CPU
- LightGlue matches keypoints; RANSAC filters geometrically inconsistent matches
- Best candidate = most verified inliers

### Stage 3 — Refinement
- **Heading refinement**: Tests ±45° offsets at 15° steps for top 15 candidates
- **Spatial consensus**: Clusters matches into 50m cells; prefers clusters over outliers
- **Confidence scoring**: Evaluates geographic clustering + uniqueness ratio

---

## Ultra Mode

Enable the **Ultra Mode** checkbox in the GUI for difficult images (night, blur, low texture).

Ultra Mode adds:
- **LoFTR**: Detector-free dense matching — handles blur/low-contrast
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of the best match

Significantly slower but catches matches the standard pipeline misses.

---

## Using CosPlace Utilities Directly

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA / MPS / CPU)
model = load_cosplace_model()

# Extract a 512-dim descriptor from any PIL image
img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img)  # shape: (512,)

print(descriptor.shape)  # torch.Size([512])
```

---

## Building the Index Programmatically

For large areas, use the standalone high-performance builder:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 2.0 \
  --resolution 300
```

This writes chunks to `cosplace_parts/` and compiles them into `index/cosplace_descriptors.npy` and `index/metadata.npz`.

---

## Searching the Index Programmatically

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]
panoids = meta["panoids"]
headings = meta["headings"]

# Load CosPlace and extract query descriptor
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

model = load_cosplace_model()
query_img = Image.open("query.jpg").convert("RGB")
query_desc = extract_descriptor(model, query_img).numpy()  # (512,)

# Radius filter (e.g., 5km around Paris center)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 5.0

mask = np.array([
    haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
    for i in range(len(lats))
])
filtered_descs = descriptors[mask]
filtered_indices = np.where(mask)[0]

# Cosine similarity search
norms = np.linalg.norm(filtered_descs, axis=1, keepdims=True)
filtered_norm = filtered_descs / (norms + 1e-8)
query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
similarities = filtered_norm @ query_norm  # (M,)

# Top-K candidates
top_k = 20
top_local = np.argsort(similarities)[::-1][:top_k]
top_global = filtered_indices[top_local]

for rank, idx in enumerate(top_global):
    print(f"Rank {rank+1}: lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, "
          f"panoid={panoids[idx]}, heading={headings[idx]}, "
          f"similarity={similarities[top_local[rank]]:.4f}")
```

---

## Multi-Index Strategy (Multiple Cities)

Netryx uses a single unified index. All cities share the same `cosplace_descriptors.npy`. Searches are isolated by the center coordinates + radius you provide:

```python
# Index Paris (run first)
# python build_index.py --lat 48.8566 --lon 2.3522 --radius 5.0

# Index London (appends to same index)
# python build_index.py --lat 51.5074 --lon -0.1278 --radius 5.0

# Search only in Paris — London results are excluded by radius filter
center_lat, center_lon = 48.8566, 2.3522
radius_km = 5.0
```

---

## Platform-Specific Behavior

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|--------------|---------------------|-----|
| Feature extractor | ALIKED (1024 kp) | DISK (768 kp) | DISK |
| LoFTR (Ultra) | ✅ Full speed | ✅ Supported | ✅ Slow |
| Recommended VRAM | 8GB+ | 8GB unified | N/A |

```python
import torch

# Device selection used internally by Netryx
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
```

---

## Common Patterns

### Batch-process multiple query images

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import numpy as np
import os

model = load_cosplace_model()
descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

image_dir = "queries/"
results = []

for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
    desc = extract_descriptor(model, img).numpy()
    sims = descriptors @ desc / (
        np.linalg.norm(descriptors, axis=1) * np.linalg.norm(desc) + 1e-8
    )
    best = int(np.argmax(sims))
    results.append({
        "file": fname,
        "lat": float(meta["lats"][best]),
        "lon": float(meta["lons"][best]),
        "confidence": float(sims[best]),
        "panoid": str(meta["panoids"][best]),
    })
    print(f"{fname}: ({results[-1]['lat']:.6f}, {results[-1]['lon']:.6f}) "
          f"conf={results[-1]['confidence']:.4f}")
```

### Check index size and coverage

```python
import numpy as np

meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]
lons = meta["lons"]

print(f"Total indexed panoramas: {len(lats):,}")
print(f"Lat range: {lats.min():.4f} → {lats.max():.4f}")
print(f"Lon range: {lons.min():.4f} → {lons.max():.4f}")

descs = np.load("index/cosplace_descriptors.npy")
print(f"Descriptor matrix shape: {descs.shape}")  # (N, 512)
print(f"Index size on disk: {descs.nbytes / 1e6:.1f} MB")
```

---

## Troubleshooting

**GUI renders blank on macOS**
```bash
brew install python-tk@3.11  # match your exact Python version
```

**`ModuleNotFoundError: lightglue`**
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

**`ModuleNotFoundError: kornia`** (Ultra Mode)
```bash
pip install kornia
```

**Indexing stops mid-way**
- Re-run the same command — indexing is incremental and resumes from `cosplace_parts/`
- Chunks already written are not re-processed

**Low confidence scores / wrong location**
- Enable **Ultra Mode** for degraded images (blur, night, low texture)
- Increase the search radius — the correct location may be outside your current radius
- Try **AI Coarse** mode if you have no prior knowledge of the region

**CUDA out of memory**
```python
# Reduce keypoint count in the extractor config inside test_super.py
# ALIKED default: 1024 keypoints — try 512
extractor = KF.ALIKED(max_num_keypoints=512, ...).to(device)
```

**Index search returns no candidates**
```python
# Verify your search area overlaps with indexed area
import numpy as np
meta = np.load("index/metadata.npz", allow_pickle=True)
print(f"Index covers lats: {meta['lats'].min():.4f} to {meta['lats'].max():.4f}")
print(f"Index covers lons: {meta['lons'].min():.4f} to {meta['lons'].max():.4f}")
```

**MPS (Apple Silicon) errors with ALIKED**
- Netryx automatically falls back to DISK on MPS — no action needed
- If you see MPS tensor errors, ensure PyTorch ≥ 2.0: `pip install --upgrade torch`

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core deep learning runtime |
| `torchvision` | Image transforms |
| `lightglue` (GitHub) | Deep feature matching |
| `kornia` (optional) | LoFTR dense matching (Ultra Mode) |
| `numpy` | Index storage and similarity search |
| `Pillow` | Image loading |
| `requests` | Street View tile downloads |
| `tkinter` | GUI (stdlib, may need system install on macOS) |
