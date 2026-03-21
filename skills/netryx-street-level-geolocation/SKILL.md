```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source locally-hosted street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - reverse image geolocation
  - identify location from street view photo
  - netryx geolocation
  - build street view index
  - osint photo location
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It works by indexing street-view panoramas into a searchable embedding database, then matching query photos against that index using a three-stage computer vision pipeline: global retrieval (CosPlace), local feature matching (ALIKED/DISK + LightGlue), and spatial refinement.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (install from source)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode (LoFTR dense matching)
pip install kornia
```

### Platform-Specific Notes

- **NVIDIA GPU**: CUDA auto-detected; uses ALIKED (1024 keypoints)
- **Apple Silicon (M1/M2/M3/M4)**: MPS auto-detected; uses DISK (768 keypoints)
- **CPU-only**: Works but significantly slower; uses DISK
- **macOS blank GUI fix**: `brew install python-tk@3.11` (replace 3.11 with your Python version)

### Optional: Gemini API for AI Coarse Mode

```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles both index creation and searching.

---

## Core Workflow

### Step 1: Create an Index

Index an area before searching. Indexing crawls street-view panoramas and extracts CosPlace fingerprints.

**In GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (km) — start with 0.5–1km for testing
4. Set grid resolution (default: 300, don't change)
5. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|------------|---------------|------------|
| 0.5 km | ~500       | 30 min        | ~60 MB     |
| 1 km   | ~2,000     | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000    | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000   | 24–48 hours   | ~7 GB      |

Indexing is **resumable** — interrupting and restarting continues from where it stopped.

### Step 2: Search

**In GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate coordinates + radius (recommended)
   - **AI Coarse**: Gemini analyzes visual clues to guess region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result shows on map with GPS coordinates and confidence score

---

## Project Structure

```
netryx/
├── test_super.py          # Main application: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large datasets)
├── requirements.txt       # Dependencies
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## The Three-Stage Pipeline

### Stage 1: Global Retrieval (CosPlace)
- Extracts 512-dim fingerprint from query image + horizontally flipped version
- Cosine similarity search against entire index
- Haversine radius filter applied
- Returns top 500–1000 candidates
- Speed: **< 1 second**

### Stage 2: Local Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads panorama tiles from street view (8 tiles, stitched)
- Generates rectilinear crops at 3 FOVs: 70°, 90°, 110°
- Extracts local keypoints via ALIKED (CUDA) or DISK (MPS/CPU)
- LightGlue deep feature matching between query and candidate keypoints
- RANSAC filters geometrically inconsistent matches
- Speed: **2–5 minutes for 300–500 candidates**

### Stage 3: Refinement
- Heading refinement: ±45° at 15° steps across 3 FOVs for top 15 candidates
- Spatial consensus: clusters matches into 50m cells
- Confidence scoring: geographic clustering + uniqueness ratio

---

## Ultra Mode

Enable for difficult images (night, blur, low texture):

```
☑ Ultra Mode  (checkbox in GUI)
```

Ultra Mode adds:
- **LoFTR**: Detector-free dense matching (handles blur/low contrast)
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

Requires `kornia`: `pip install kornia`

---

## Working with the Index Programmatically

### Extract a CosPlace Descriptor

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model, device = get_cosplace_model()

# Extract descriptor from an image
img = Image.open("query_photo.jpg")
descriptor = get_descriptor(model, img, device)  # shape: (512,)
print(f"Descriptor shape: {descriptor.shape}")
```

### Search the Index Manually

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]    # (N,)
lons = meta["lons"]    # (N,)
panoids = meta["panoids"]  # (N,)
headings = meta["headings"]  # (N,)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """Search the index with radius filtering."""
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    mask = distances <= radius_km
    
    if mask.sum() == 0:
        return []
    
    # Cosine similarity search
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db = descriptors[mask]
    db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-8)
    sims = db_norm @ q
    
    # Top-k candidates
    indices_in_mask = np.where(mask)[0]
    top_local = np.argsort(-sims)[:top_k]
    top_global = indices_in_mask[top_local]
    
    results = []
    for idx in top_global:
        results.append({
            "lat": float(lats[idx]),
            "lon": float(lons[idx]),
            "panoid": str(panoids[idx]),
            "heading": float(headings[idx]),
            "similarity": float(sims[top_local[np.where(top_global == idx)[0][0]]])
        })
    
    return results

# Usage
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image

model, device = get_cosplace_model()
img = Image.open("mystery_photo.jpg")
desc = get_descriptor(model, img, device)

candidates = search_index(
    query_descriptor=desc,
    center_lat=48.8566,   # Paris
    center_lon=2.3522,
    radius_km=5.0,
    top_k=500
)

print(f"Found {len(candidates)} candidates")
print(f"Top match: {candidates[0]}")
```

### Build Index from Parts (Standalone)

For large areas, use the high-performance builder:

```bash
python build_index.py
```

This compiles `cosplace_parts/*.npz` into the unified `index/` directory. Run after manual data collection or after interrupting/resuming indexing.

### Manually Add Embeddings to Index Parts

```python
import numpy as np
import os

def save_embedding_chunk(descriptors, lats, lons, panoids, headings, chunk_id):
    """Save a batch of embeddings as a cosplace_parts chunk."""
    os.makedirs("cosplace_parts", exist_ok=True)
    np.savez_compressed(
        f"cosplace_parts/part_{chunk_id:06d}.npz",
        descriptors=np.array(descriptors, dtype=np.float32),
        lats=np.array(lats, dtype=np.float64),
        lons=np.array(lons, dtype=np.float64),
        panoids=np.array(panoids),
        headings=np.array(headings, dtype=np.float32)
    )

# Example: add custom panorama embeddings
save_embedding_chunk(
    descriptors=[desc],           # list of (512,) arrays
    lats=[48.8584],
    lons=[2.2945],
    panoids=["custom_pano_001"],
    headings=[90.0],
    chunk_id=99999
)
```

---

## Multi-Region Indexing

All regions go into the **same unified index**. The radius filter isolates them at search time.

```python
# Index Paris (5km radius) — stored in same index as everything else
# Search Paris:
paris_results = search_index(desc, 48.8566, 2.3522, radius_km=5.0)

# Index London separately (same index file)
# Search London:
london_results = search_index(desc, 51.5074, -0.1278, radius_km=5.0)

# Searches don't cross-contaminate — radius filter handles separation
```

---

## Common Patterns

### Pattern: Full Geolocation Pipeline (Headless)

```python
"""
Run the full geolocation pipeline from Python (no GUI).
This calls into the same functions test_super.py uses.
"""
import sys
sys.path.insert(0, ".")

from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import numpy as np

def geolocate_image(image_path, center_lat, center_lon, radius_km=2.0):
    """
    Stage 1 only (fast retrieval). For full pipeline including
    LightGlue verification, use the GUI or extend with Stage 2 calls
    from test_super.py.
    """
    model, device = get_cosplace_model()
    
    # Load and extract descriptor
    img = Image.open(image_path).convert("RGB")
    desc = get_descriptor(model, img, device)
    
    # Also try flipped version
    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    desc_flipped = get_descriptor(model, img_flipped, device)
    
    # Load index
    descriptors = np.load("index/cosplace_descriptors.npy")
    meta = np.load("index/metadata.npz", allow_pickle=True)
    
    # Search with both descriptors, merge results
    results_normal = search_index(desc, center_lat, center_lon, radius_km)
    results_flipped = search_index(desc_flipped, center_lat, center_lon, radius_km)
    
    # Combine and deduplicate by panoid
    all_results = {r["panoid"]: r for r in results_normal + results_flipped}
    ranked = sorted(all_results.values(), key=lambda x: -x["similarity"])
    
    return ranked[:10]  # Top 10 candidates for Stage 2 verification

results = geolocate_image("test_photo.jpg", 48.8566, 2.3522, radius_km=3.0)
for r in results:
    print(f"  {r['lat']:.6f}, {r['lon']:.6f} | sim={r['similarity']:.3f} | pano={r['panoid']}")
```

### Pattern: Batch Indexing from Custom Image Source

```python
"""
Index panoramas from a custom source (e.g., Mapillary, KartaView)
instead of the default provider. Netryx is source-agnostic.
"""
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import numpy as np
import requests
from io import BytesIO

model, device = get_cosplace_model()

def index_custom_panorama(pano_url, lat, lon, heading, chunk_id, pano_id):
    """Download a panorama and add it to the index."""
    response = requests.get(pano_url, timeout=30)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    desc = get_descriptor(model, img, device)
    
    save_embedding_chunk(
        descriptors=[desc],
        lats=[lat],
        lons=[lon],
        panoids=[pano_id],
        headings=[heading],
        chunk_id=chunk_id
    )
    return desc

# Example: index a list of panoramas
panoramas = [
    {"url": "https://example.com/pano1.jpg", "lat": 48.860, "lon": 2.350, "heading": 0, "id": "pano_001"},
    {"url": "https://example.com/pano2.jpg", "lat": 48.861, "lon": 2.351, "heading": 90, "id": "pano_002"},
]

for i, pano in enumerate(panoramas):
    index_custom_panorama(
        pano_url=pano["url"],
        lat=pano["lat"],
        lon=pano["lon"],
        heading=pano["heading"],
        chunk_id=i,
        pano_id=pano["id"]
    )

# After indexing, rebuild the compiled index
import subprocess
subprocess.run(["python", "build_index.py"])
```

### Pattern: Check Index Stats

```python
import numpy as np

def index_stats():
    """Print statistics about the current index."""
    try:
        descriptors = np.load("index/cosplace_descriptors.npy")
        meta = np.load("index/metadata.npz", allow_pickle=True)
        lats = meta["lats"]
        lons = meta["lons"]
        
        print(f"Total panoramas indexed: {len(descriptors):,}")
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Lat range: {lats.min():.4f} to {lats.max():.4f}")
        print(f"Lon range: {lons.min():.4f} to {lons.max():.4f}")
        print(f"Index size: {descriptors.nbytes / 1e6:.1f} MB")
    except FileNotFoundError:
        print("No compiled index found. Run build_index.py after creating parts.")

index_stats()
```

### Pattern: Verify LightGlue Installation

```python
def verify_installation():
    """Check all required components are installed correctly."""
    checks = {}
    
    # LightGlue
    try:
        from lightglue import LightGlue, ALIKED, DISK
        checks["lightglue"] = "✓"
    except ImportError:
        checks["lightglue"] = "✗ Run: pip install git+https://github.com/cvg/LightGlue.git"
    
    # CosPlace utils
    try:
        from cosplace_utils import get_cosplace_model
        checks["cosplace_utils"] = "✓"
    except ImportError:
        checks["cosplace_utils"] = "✗ cosplace_utils.py not found"
    
    # Ultra Mode (optional)
    try:
        import kornia
        checks["kornia (Ultra Mode)"] = "✓"
    except ImportError:
        checks["kornia (Ultra Mode)"] = "○ Optional: pip install kornia"
    
    # Device detection
    import torch
    if torch.cuda.is_available():
        checks["device"] = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        checks["device"] = "MPS (Apple Silicon)"
    else:
        checks["device"] = "CPU (no GPU detected)"
    
    for k, v in checks.items():
        print(f"  {k}: {v}")

verify_installation()
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11  # Replace 3.11 with your Python version
# Then re-activate venv and run again
```

### `ImportError: No module named 'lightglue'`
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Note: package is 'lightglue' (lowercase) imported as `from lightglue import ...`
```

### `FileNotFoundError: index/cosplace_descriptors.npy`
The compiled index doesn't exist yet. Either:
1. Run indexing through the GUI (Create mode), or
2. If you have `cosplace_parts/` populated, run: `python build_index.py`

### No candidates found for search area
- The search radius doesn't overlap your indexed area
- Index was created for a different region
- Verify: run `index_stats()` to check lat/lon coverage of your index

### Very low similarity scores (< 0.3) for all candidates
- Query image is from a region not in the index — expand radius or re-index
- Image is severely distorted, cropped, or synthetic
- Try Ultra Mode for degraded images

### CUDA out of memory
```python
# In test_super.py or your custom code, reduce batch size or use:
import torch
torch.cuda.empty_cache()
# Or switch to CPU: set device = torch.device("cpu")
```

### Indexing stops mid-way
Indexing is resumable. Simply restart the same Create Index job with identical parameters — it skips already-processed panoramas and continues from the last checkpoint.

### Stage 2 (LightGlue) is very slow on CPU
- Expected: 2–5 min on GPU, can be 30–60+ min on CPU for 500 candidates
- Reduce candidates by shrinking search radius
- Use a GPU machine or Apple Silicon for production use

---

## Key Models Reference

| Model | Role | Paper |
|-------|------|-------|
| CosPlace | Global 512-dim descriptor (retrieval) | CVPR 2022 |
| ALIKED | Local keypoints on CUDA | IEEE TIP 2023 |
| DISK | Local keypoints on MPS/CPU | NeurIPS 2020 |
| LightGlue | Deep feature matching | ICCV 2023 |
| LoFTR | Dense matching, Ultra Mode only | CVPR 2021 |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | No | AI Coarse mode (Gemini blind region detection) |
```
