```markdown
---
name: netryx-street-level-geolocation
description: Use Netryx to geolocate street-level images locally using CosPlace, ALIKED/DISK, and LightGlue computer vision pipeline
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - set up Netryx geolocation
  - index street view panoramas
  - run netryx search
  - street level geolocation pipeline
  - build a location index with netryx
  - identify location from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates from street-level photographs. It crawls Street View panoramas, builds a searchable index of visual fingerprints, and matches query images using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature extraction (ALIKED/DISK), and deep feature matching (LightGlue). Sub-50m accuracy, no landmarks needed, runs entirely on local hardware.

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

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

**GPU support:**
- NVIDIA: CUDA (ALIKED extractor, 1024 keypoints)
- Apple Silicon: MPS (DISK extractor, 768 keypoints)
- CPU: Supported but slow

**Optional — Gemini API for AI Coarse location guessing:**
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

> macOS blank GUI fix: `brew install python-tk@3.11`

---

## Core Workflow

### 1. Create an Index

Index a geographic area by crawling Street View panoramas and extracting CosPlace fingerprints.

In the GUI:
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Index is saved incrementally to `cosplace_parts/` — safe to interrupt and resume.

**Index size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk |
|--------|-----------|---------------|------|
| 0.5 km | ~500      | 30 min        | ~60 MB |
| 1 km   | ~2,000    | 1–2 hrs       | ~250 MB |
| 5 km   | ~30,000   | 8–12 hrs      | ~3 GB |
| 10 km  | ~100,000  | 24–48 hrs     | ~7 GB |

### 2. Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual**: Provide known approximate coordinates + radius
   - **AI Coarse**: Gemini analyzes visual cues to guess region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI, indexing, search
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors
    └── metadata.npz               # Coordinates, headings, panoid IDs
```

---

## Using the Index Programmatically

### Extract a CosPlace Descriptor

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
model = load_cosplace_model(device=device)

# Extract 512-dim fingerprint from an image
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
print(descriptor.shape)  # (512,)
```

### Search the Index

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
headings = meta["headings"]
panoids = meta["panoids"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km=2.0, top_k=500):
    """
    Returns top_k candidates sorted by cosine similarity within radius.
    query_descriptor: np.ndarray shape (512,)
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lat, lon)
        for lat, lon in zip(lats, lons)
    ])
    mask = distances <= radius_km
    filtered_idx = np.where(mask)[0]

    if len(filtered_idx) == 0:
        return []

    # Cosine similarity
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db = descriptors[filtered_idx]
    db_norms = np.linalg.norm(db, axis=1, keepdims=True) + 1e-8
    db_normed = db / db_norms
    sims = db_normed @ q

    top_local = np.argsort(-sims)[:top_k]
    top_global = filtered_idx[top_local]

    results = []
    for i, idx in enumerate(top_global):
        results.append({
            "rank": i + 1,
            "panoid": panoids[idx],
            "lat": float(lats[idx]),
            "lon": float(lons[idx]),
            "heading": float(headings[idx]),
            "similarity": float(sims[top_local[i]]),
        })
    return results

# Example usage
results = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_km=1.5)
for r in results[:5]:
    print(f"Rank {r['rank']}: panoid={r['panoid']} lat={r['lat']:.6f} lon={r['lon']:.6f} sim={r['similarity']:.4f}")
```

### Flipped Descriptor Search (Improves Recall)

```python
import torchvision.transforms.functional as TF

def extract_with_flip(model, img, device):
    """Extract and average descriptor from original + horizontally flipped image."""
    d1 = extract_descriptor(model, img, device=device)
    img_flipped = TF.hflip(img)
    d2 = extract_descriptor(model, img_flipped, device=device)
    combined = (d1 + d2) / 2.0
    return combined / (np.linalg.norm(combined) + 1e-8)

descriptor = extract_with_flip(model, img, device)
results = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
```

---

## Pipeline Stages

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dim descriptor from the query image (+ flipped version)
- Cosine similarity search against all indexed panoramas
- Radius (haversine) filter limits to target area
- Returns top 500–1000 candidates
- Runtime: <1 second (matrix multiply)

### Stage 2 — Local Feature Matching (ALIKED/DISK + LightGlue)
- Downloads panorama tiles from Street View; stitches them
- Crops at 3 FOVs: 70°, 90°, 110° (handles zoom mismatch)
- Extracts keypoints with ALIKED (CUDA) or DISK (MPS/CPU)
- Matches keypoints with LightGlue
- RANSAC filters geometrically inconsistent matches
- Runtime: 2–5 min for 300–500 candidates

### Stage 3 — Refinement
- Heading refinement: ±45° at 15° steps for top 15 candidates
- Spatial consensus: clusters matches into 50m cells; prefers clusters over outliers
- Confidence scoring: geographic clustering + uniqueness ratio

### Ultra Mode (Optional)
Enable for blurry/night/low-texture images:
- **LoFTR**: Detector-free dense matching (handles degraded images)
- **Descriptor hopping**: Re-searches index using matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

---

## Multi-City Index Pattern

All embeddings share one unified index — no per-city management needed:

```python
# Index Paris
search_index(desc, center_lat=48.8566, center_lon=2.3522, radius_km=5.0)

# Index Tel Aviv  
search_index(desc, center_lat=32.0853, center_lon=34.7818, radius_km=5.0)

# Searching Paris only returns Paris results; Tel Aviv only returns Tel Aviv results
# No city selection — coordinates + radius handle scoping automatically
```

---

## Building the Index from CLI

For large datasets, use the standalone high-performance builder:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --grid 300
```

This writes chunks to `cosplace_parts/` and compiles them to `index/cosplace_descriptors.npy` + `index/metadata.npz`.

---

## Common Patterns

### Batch Index Multiple Cities

```python
import subprocess

cities = [
    ("Paris",    48.8566,  2.3522,  5.0),
    ("London",   51.5074, -0.1278,  5.0),
    ("New York", 40.7128, -74.0060, 5.0),
]

for name, lat, lon, radius in cities:
    print(f"Indexing {name}...")
    subprocess.run([
        "python", "build_index.py",
        "--lat", str(lat),
        "--lon", str(lon),
        "--radius", str(radius),
        "--grid", "300"
    ])
    print(f"{name} done.")
```

### Confidence Scoring Heuristic

```python
def compute_confidence(results, top_n=10):
    """
    Simple confidence: ratio of best similarity to mean of remaining top-N.
    Higher ratio = more unique / confident match.
    """
    if len(results) < 2:
        return 0.0
    best = results[0]["similarity"]
    rest_mean = np.mean([r["similarity"] for r in results[1:top_n]])
    return float(best / (rest_mean + 1e-8))

conf = compute_confidence(results)
print(f"Confidence ratio: {conf:.2f}")  # >1.3 is typically a strong match
```

### Quick Environment Check

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

# Feature extractor chosen automatically:
# CUDA  → ALIKED (1024 keypoints, best accuracy)
# MPS   → DISK  (768 keypoints, Apple Silicon optimized)
# CPU   → DISK  (slow, functional)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| macOS GUI is blank | `brew install python-tk@3.11` (or your Python version) |
| `ImportError: lightglue` | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ImportError: kornia` | `pip install kornia` (only needed for Ultra Mode / LoFTR) |
| `GEMINI_API_KEY` not found | `export GEMINI_API_KEY="..."` before launching |
| Index search returns 0 results | Radius too small, or area not indexed — expand radius or re-index |
| Very low confidence scores | Try Ultra Mode; image may be blurry/night/low-texture |
| Indexing interrupted | Safe to re-run — resumes from last checkpoint in `cosplace_parts/` |
| Out of VRAM | Reduce top-K candidates or use MPS/CPU path |
| Wrong match despite high inliers | Enable spatial consensus (built into Stage 3); check heading refinement |

---

## Model References

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim visual fingerprint | Any |
| [ALIKED](https://github.com/naver/alike) | Local keypoints + descriptors | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints + descriptors | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | Any |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Detector-free dense matching (Ultra) | Any |
```
