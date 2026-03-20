---
name: netryx-street-level-geolocation
description: Local-first street-level geolocation engine using CosPlace, ALIKED/DISK, and LightGlue to identify GPS coordinates from any street photo
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - identify location from photo
  - netryx geolocation
  - build street view index
  - visual place recognition pipeline
  - osint geolocation from image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a local-first geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, builds a searchable visual index using CosPlace embeddings, then matches query images via ALIKED/DISK keypoints + LightGlue to achieve sub-50m accuracy — entirely on your own hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

### Optional: Gemini API for AI Coarse region detection

```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### GPU / platform auto-detection

| Platform | Feature extractor | Notes |
|----------|------------------|-------|
| NVIDIA CUDA | ALIKED (1024 kp) | Fastest |
| Apple MPS (M1+) | DISK (768 kp) | Good performance |
| CPU | DISK | Slow but functional |

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank window fix:** `brew install python-tk@3.11` (match your Python version)

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI + indexing + search entry point
├── cosplace_utils.py      # CosPlace model loading & descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors
    └── metadata.npz               # coords, headings, panoid IDs
```

---

## Core Pipeline Overview

```
Query Image
  │
  ├─ CosPlace → 512-dim global descriptor
  ├─ Flipped CosPlace descriptor (catches reversed perspectives)
  │
  ▼
Cosine similarity search against index (radius-filtered via haversine)
  │
  ├─ Top 500–1000 candidate panoramas
  │
  ▼
Download panoramas → 3 FOV crops (70°, 90°, 110°)
  │
  ├─ ALIKED/DISK: extract local keypoints
  ├─ LightGlue: deep feature matching
  ├─ RANSAC: geometric verification (filter false matches)
  │
  ▼
Heading refinement (±45°, 15° steps, top 15 candidates)
  │
  ├─ Spatial consensus clustering (50m cells)
  ├─ Confidence scoring
  │
  ▼
📍 GPS Coordinates + Confidence Score
```

---

## Step 1: Create an Index

Index an area before searching. The index stores CosPlace fingerprints for every crawled street-view panorama.

**In the GUI:**
1. Select **Create** mode
2. Enter center lat/lon of the area
3. Set radius (km) and grid resolution (default 300 — don't change)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Indexing is **incremental** — interrupting and restarting resumes from the last saved chunk.

**Multiple cities in one index:** Index Paris, then London, then Tokyo — all go into the same index. The radius filter at search time restricts results to your specified area automatically.

---

## Step 2: Search

**In the GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual** — provide approximate center coordinates + radius
   - **AI Coarse** — Gemini analyzes visual cues (signs, architecture) to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Results appear on the map with confidence score

---

## Using the CosPlace Utilities Programmatically

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA / MPS / CPU)
model = load_cosplace_model()

# Extract a 512-dim descriptor from any image
img = Image.open("query_photo.jpg")
descriptor = get_descriptor(model, img)  # shape: (512,)
print(descriptor.shape)  # (512,)
```

---

## Building the Index from Python

```python
# Standalone high-performance indexer for large areas
# Run directly for scripted / headless indexing
import subprocess

subprocess.run([
    "python", "build_index.py",
    "--lat", "48.8566",
    "--lon", "2.3522",
    "--radius", "2.0",       # km
    "--resolution", "300",   # grid density (don't change)
    "--output", "index/"
])
```

Or import and call directly if `build_index.py` exposes a function:

```python
# Check build_index.py for the exposed API — typical pattern:
from build_index import build_index

build_index(
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    grid_resolution=300,
    output_dir="index/"
)
```

---

## Searching the Index Programmatically

```python
import numpy as np
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]        # (N,)
lons = meta["lons"]        # (N,)
headings = meta["headings"]
panoids = meta["panoids"]

# Extract query descriptor
model = load_cosplace_model()
query_img = Image.open("query.jpg")
query_desc = get_descriptor(model, query_img)              # (512,)

# Also extract flipped descriptor (catches reversed perspectives)
query_flipped = get_descriptor(model, query_img.transpose(Image.FLIP_LEFT_RIGHT))

# Cosine similarity search
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(b, a) / (norm(b, axis=1) * norm(a))

sims = np.maximum(cosine_sim(query_desc, descriptors),
                  cosine_sim(query_flipped, descriptors))

# Haversine radius filter (example: 5km around Paris center)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 5.0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

dists = haversine_km(center_lat, center_lon, lats, lons)
mask = dists <= radius_km

# Top 500 candidates within radius
filtered_indices = np.where(mask)[0]
top_idx = filtered_indices[np.argsort(sims[filtered_indices])[::-1][:500]]

print(f"Top match: lat={lats[top_idx[0]]:.6f}, lon={lons[top_idx[0]]:.6f}")
print(f"Similarity: {sims[top_idx[0]]:.4f}")
print(f"Panoid: {panoids[top_idx[0]]}, Heading: {headings[top_idx[0]]}")
```

---

## Ultra Mode

Enable for difficult images (night, blur, low texture). Adds:
- **LoFTR** — detector-free dense matching (requires `kornia`)
- **Descriptor hopping** — re-searches from matched panorama's clean descriptor
- **Neighborhood expansion** — checks all panoramas within 100m of best match

In GUI: check **Ultra Mode** before clicking **Start Full Search**.

Programmatically (if exposed):

```python
# Ultra Mode is controlled via the GUI checkbox; to trigger via code
# look for an `ultra_mode=True` parameter in the search function inside test_super.py
# Example pattern (verify against actual source):
results = run_search(
    query_image_path="blurry_night_photo.jpg",
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    ultra_mode=True
)
```

---

## Common Patterns

### Pattern 1: Batch geolocate multiple images

```python
import os
from pathlib import Path

image_dir = Path("images_to_geolocate/")
results = []

model = load_cosplace_model()
descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

for img_path in image_dir.glob("*.jpg"):
    img = Image.open(img_path)
    desc = get_descriptor(model, img)
    sims = cosine_sim(desc, descriptors)
    best_idx = np.argmax(sims)
    results.append({
        "image": img_path.name,
        "lat": float(meta["lats"][best_idx]),
        "lon": float(meta["lons"][best_idx]),
        "confidence": float(sims[best_idx]),
        "panoid": str(meta["panoids"][best_idx])
    })
    print(f"{img_path.name}: ({results[-1]['lat']:.5f}, {results[-1]['lon']:.5f}) conf={results[-1]['confidence']:.3f}")
```

### Pattern 2: Export results to GeoJSON

```python
import json

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {"image": r["image"], "confidence": r["confidence"]}
        }
        for r in results
    ]
}

with open("results.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
```

### Pattern 3: Confidence thresholding

```python
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.55

for r in results:
    if r["confidence"] >= HIGH_CONFIDENCE:
        print(f"✅ HIGH: {r['image']} → {r['lat']:.5f}, {r['lon']:.5f}")
    elif r["confidence"] >= MEDIUM_CONFIDENCE:
        print(f"⚠️  MED:  {r['image']} → {r['lat']:.5f}, {r['lon']:.5f}")
    else:
        print(f"❌ LOW:  {r['image']} — try Ultra Mode or expand radius")
```

---

## Troubleshooting

### GUI appears blank (macOS)
```bash
brew install python-tk@3.11   # match your Python version
```

### `ModuleNotFoundError: lightglue`
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### `ModuleNotFoundError: kornia` (Ultra Mode)
```bash
pip install kornia
```

### CUDA out of memory
- Reduce candidates: lower the top-K from 500 to 200
- Use DISK instead of ALIKED (fewer keypoints)
- Close other GPU processes

### Index search returns 0 candidates
- Verify `center_lat`/`center_lon` are within indexed area
- Increase `radius_km` — index may be offset from expected center
- Check `index/metadata.npz` lat/lon ranges:
  ```python
  meta = np.load("index/metadata.npz", allow_pickle=True)
  print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
  print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
  ```

### Indexing interrupted / partial index
Re-run the same **Create Index** command — it resumes from the last saved `cosplace_parts/*.npz` chunk automatically.

### Low accuracy results
1. Enable **Ultra Mode**
2. Increase index density — re-index with a smaller radius and higher grid resolution
3. Expand search radius — the correct panorama may be just outside your radius
4. Try both Manual and AI Coarse modes — AI Coarse may place you in the wrong country

### MPS (Apple Silicon) slower than expected
DISK on MPS is used instead of ALIKED. This is intentional — ALIKED has compatibility issues on MPS. Performance is still significantly faster than CPU.

---

## Key Configuration Reference

| Parameter | Where | Default | Notes |
|-----------|-------|---------|-------|
| Grid resolution | GUI / `build_index.py` | 300 | Don't change — controls panorama density |
| Top-K candidates | `test_super.py` | 500–1000 | Lower = faster, less accurate |
| Heading refinement range | `test_super.py` | ±45°, 15° steps | Applied to top 15 candidates |
| Spatial cluster cell size | `test_super.py` | 50m | Consensus clustering resolution |
| Ultra Mode neighborhood | `test_super.py` | 100m | Expansion radius around best match |
| Multi-FOV crops | `test_super.py` | 70°, 90°, 110° | Handles zoom mismatches |
