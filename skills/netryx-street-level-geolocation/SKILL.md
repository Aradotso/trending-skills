---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to precise GPS coordinates using computer vision pipelines running entirely on local hardware.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - netryx geolocation
  - index street view panoramas
  - locate where a photo was taken
  - osint image geolocation
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from any street-level photograph. It indexes street-view panoramas into a searchable embedding database, then uses a three-stage computer vision pipeline (global retrieval → local geometric verification → refinement) to match a query image against millions of indexed locations. No cloud API needed for searching — only for panorama data fetching during indexing.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

**Optional — Gemini API for AI Coarse region guessing:**
```bash
export GEMINI_API_KEY="your_key_from_aistudio_google_com"
```

**macOS blank GUI fix:**
```bash
brew install python-tk@3.11   # match your Python version
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 50 GB+ |
| Python | 3.9+ | 3.10+ |

GPU backends: CUDA (NVIDIA) → ALIKED extractor; MPS (Apple Silicon) → DISK extractor; CPU → DISK, slowest.

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface for both indexing and searching.

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace vectors
    └── metadata.npz               # lat/lon, heading, panorama IDs
```

---

## Core Workflow

### Step 1 — Create an Index

Index an area before searching. The indexer crawls street-view panoramas, crops them, and extracts CosPlace fingerprints.

**GUI method:**
1. Select **Create** mode
2. Enter center `lat, lon`
3. Set radius (km) and grid resolution (default 300 — do not change)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Indexing is **resumable** — if interrupted, re-running picks up from the last saved chunk in `cosplace_parts/`.

**For large areas, use the standalone builder:**
```bash
python build_index.py
```

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center `lat, lon` + radius
   - **AI Coarse**: Gemini analyzes visual cues (signs, architecture) to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

**Enable Ultra Mode** for difficult images (night shots, motion blur, low texture). Adds LoFTR dense matching, descriptor hopping, and 100 m neighborhood expansion. Much slower but improves recall significantly.

---

## Multi-City Index

All embeddings live in one unified index. The radius filter handles city separation automatically:

```
# Index Paris
center=(48.8566, 2.3522), radius=5km  →  stored in index

# Index London
center=(51.5074, -0.1278), radius=5km  →  appended to same index

# Search Paris only
search center=(48.8566, 2.3522), radius=5km  →  returns only Paris results

# Search London only
search center=(51.5074, -0.1278), radius=5km  →  returns only London results
```

No city selection or separate databases needed.

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace 512-dim descriptor (original + horizontally flipped)
    │
    ▼
Index cosine similarity search  →  radius-filtered (haversine)
    │
    └── Top 500–1000 candidates  (<1 second, single matrix multiply)
    │
    ▼
For each candidate:
    ├── Download panorama (8 tiles, stitched)
    ├── Crop at indexed heading
    ├── Multi-FOV crops: 70°, 90°, 110°
    ├── ALIKED (CUDA) or DISK (MPS/CPU) → keypoints + descriptors
    └── LightGlue matching → RANSAC geometric verification → inlier count
    │
    (2–5 min for 300–500 candidates)
    │
    ▼
Refinement:
    ├── Heading refinement: ±45° at 15° steps, 3 FOVs, top 15 candidates
    ├── Spatial consensus: cluster matches into 50 m cells
    └── Confidence scoring: clustering density + uniqueness ratio
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

**Ultra Mode additions:**
- LoFTR detector-free dense matcher (handles blur/low contrast)
- Descriptor hopping: re-search index using matched panorama's clean descriptor
- Neighborhood expansion: test all panoramas within 100 m of best match

---

## Using CosPlace Utilities Directly

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model, device = load_cosplace_model()

# Extract descriptor from an image
img = Image.open("query.jpg")
descriptor = extract_descriptor(model, img, device)
# descriptor.shape → (512,)

# Extract descriptor from flipped image (catches reversed perspectives)
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = extract_descriptor(model, img_flipped, device)
```

---

## Loading and Searching the Index

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # shape: (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]       # shape: (N,)
lons = meta["lons"]       # shape: (N,)
headings = meta["headings"]
panoids = meta["panoids"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """
    Search the index for candidates within radius_km of center.
    Returns top_k candidates sorted by cosine similarity.
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    mask = distances <= radius_km
    filtered_idx = np.where(mask)[0]

    if len(filtered_idx) == 0:
        return []

    # Cosine similarity search
    q = query_descriptor.reshape(1, -1)
    sims = cosine_similarity(q, descriptors[filtered_idx])[0]
    ranked = filtered_idx[np.argsort(sims)[::-1][:top_k]]

    return [
        {
            "idx": int(i),
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "heading": float(headings[i]),
            "panoid": str(panoids[i]),
            "similarity": float(sims[np.where(filtered_idx == i)[0][0]])
        }
        for i in ranked
    ]

# Example usage
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

model, device = load_cosplace_model()
img = Image.open("query.jpg")
desc = extract_descriptor(model, img, device)

candidates = search_index(
    query_descriptor=desc,
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    top_k=500
)
print(f"Found {len(candidates)} candidates")
print(f"Top match: {candidates[0]}")
```

---

## Common Patterns

### Pattern: Programmatic search without GUI

```python
# Minimal headless search — retrieval stage only
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import numpy as np

model, device = load_cosplace_model()

def geolocate_retrieval_only(image_path, center_lat, center_lon, radius_km=2.0):
    img = Image.open(image_path).convert("RGB")
    desc = extract_descriptor(model, img, device)

    descriptors = np.load("index/cosplace_descriptors.npy")
    meta = np.load("index/metadata.npz", allow_pickle=True)

    candidates = search_index(desc, center_lat, center_lon, radius_km)
    top = candidates[0] if candidates else None
    return top

result = geolocate_retrieval_only("mystery_street.jpg", 48.8566, 2.3522, radius_km=1.0)
if result:
    print(f"Best retrieval match: ({result['lat']:.6f}, {result['lon']:.6f})")
    print(f"Panorama ID: {result['panoid']}, Heading: {result['heading']}°")
```

### Pattern: Batch index multiple cities

```python
# Build index for multiple cities sequentially
# Run this logic before launching the GUI or build_index.py

cities = [
    {"name": "Paris",   "lat": 48.8566, "lon": 2.3522,   "radius_km": 3.0},
    {"name": "London",  "lat": 51.5074, "lon": -0.1278,  "radius_km": 3.0},
    {"name": "Berlin",  "lat": 52.5200, "lon": 13.4050,  "radius_km": 3.0},
]

# Each city appends to the same cosplace_parts/ directory.
# After all cities indexed, build_index.py compiles them into index/.
# Then search with appropriate center+radius to isolate each city.
for city in cities:
    print(f"Index {city['name']} — center ({city['lat']}, {city['lon']}), radius {city['radius_km']}km")
    # Configure GUI or build_index.py with these params, then run.
```

### Pattern: Check index coverage

```python
import numpy as np

meta = np.load("index/metadata.npz", allow_pickle=True)
lats, lons = meta["lats"], meta["lons"]

print(f"Total indexed panoramas: {len(lats):,}")
print(f"Lat range: {lats.min():.4f} → {lats.max():.4f}")
print(f"Lon range: {lons.min():.4f} → {lons.max():.4f}")
```

---

## Models Reference

| Model | Role | Activated When |
|-------|------|----------------|
| CosPlace | Global 512-dim visual fingerprint | Always (indexing + search retrieval) |
| ALIKED | Local keypoint extraction | Search, CUDA devices |
| DISK | Local keypoint extraction | Search, MPS/CPU devices |
| LightGlue | Deep feature matching + RANSAC | Search, all devices |
| LoFTR | Detector-free dense matching | Ultra Mode only |

---

## Troubleshooting

**GUI is blank on macOS**
```bash
brew install python-tk@3.11   # use your actual Python version
```

**CUDA out of memory**
- Reduce `top_k` candidates passed to Stage 2 (edit in `test_super.py`)
- Disable Ultra Mode
- Use DISK instead of ALIKED (automatic on MPS/CPU, force by removing CUDA availability)

**Zero candidates returned**
- Verify index exists: `ls index/cosplace_descriptors.npy index/metadata.npz`
- Check center coordinates are inside indexed area
- Increase `radius_km`
- Run `build_index.py` if `cosplace_parts/` has chunks but `index/` is empty

**Indexing interrupted / incomplete index**
- Re-run indexing with same parameters — it resumes from last saved chunk in `cosplace_parts/`
- Then recompile: `python build_index.py`

**Gemini AI Coarse mode fails**
```bash
export GEMINI_API_KEY="your_key_here"   # must be set before launching
```
If unavailable, use Manual mode — provide approximate coordinates manually. Manual mode is generally more reliable.

**Low confidence score / wrong location**
- Enable Ultra Mode for difficult images
- Increase indexed area density (smaller grid resolution, larger radius)
- Verify the photo is actually from the indexed area
- Try providing a tighter radius if approximate location is known

**`lightglue` import error**
```bash
pip install git+https://github.com/cvg/LightGlue.git
# NOT pip install lightglue — must be installed from GitHub
```
