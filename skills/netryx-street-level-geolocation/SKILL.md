```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, an open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - match photo to location
  - osint geolocation tool
  - reverse geolocate image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that takes any street-level photograph and returns precise GPS coordinates (sub-50m accuracy). It crawls Street View panoramas into a local index, then uses a three-stage computer vision pipeline (global retrieval → local feature matching → geometric verification) to identify the exact location. No cloud API required for searching — runs entirely on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git  # Required
pip install kornia  # Optional: Ultra Mode (LoFTR dense matching)
```

### Optional: Gemini API key (AI Coarse mode only)
```bash
export GEMINI_API_KEY="your_key_here"
```

### macOS tkinter fix (if GUI renders blank)
```bash
brew install python-tk@3.11  # Match your Python version
```

---

## Launch the GUI

```bash
python test_super.py
```

This is the single entry point for all functionality: indexing, searching, and visualization.

---

## Core Concepts

### The Index
- All panorama embeddings are stored in a single unified index (`index/cosplace_descriptors.npy` + `index/metadata.npz`)
- You index geographic areas by center coordinate + radius
- Searching is always scoped by coordinates + radius — no city selection needed
- Multiple regions (Paris, London, Tokyo) can coexist in the same index

### Pipeline Stages
1. **Stage 1 — Global Retrieval**: CosPlace 512-dim descriptor → cosine similarity search → top 500–1000 candidates (< 1 second)
2. **Stage 2 — Local Verification**: Download panorama tiles → ALIKED/DISK keypoints → LightGlue matching → RANSAC filtering (2–5 minutes for 300–500 candidates)
3. **Stage 3 — Refinement**: Heading sweep ±45°, spatial clustering in 50m cells, confidence scoring

### Device Selection (automatic)
| Hardware | Feature Extractor |
|----------|------------------|
| NVIDIA CUDA | ALIKED (1024 keypoints) |
| Apple MPS (M1+) | DISK (768 keypoints) |
| CPU fallback | DISK (768 keypoints, slower) |

---

## Step-by-Step Workflow

### Step 1: Create an Index for an Area

In the GUI:
1. Select **Create** mode
2. Enter center latitude/longitude (e.g., `48.8566, 2.3522` for Paris)
3. Set radius (km) — start with `0.5` for testing
4. Set grid resolution — use `300` (default, don't change)
5. Click **Create Index**

Indexing is incremental — safe to interrupt and resume.

**Time/size estimates:**
| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

### Step 2: Search with a Photo

In the GUI:
1. Select **Search** mode
2. Upload a street-level photo (JPEG/PNG)
3. Choose search method:
   - **Manual**: Enter known approximate coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Real-time candidate scanning visualization appears
6. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Code Examples

### Extract a CosPlace Descriptor from an Image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (downloads weights on first run)
model = load_cosplace_model()
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Extract descriptor from a query image
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device)  # shape: (512,)
print(f"Descriptor shape: {descriptor.shape}")
```

### Load the Index and Do a Cosine Similarity Search

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load prebuilt index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]       # (N,)
lons = meta["lons"]       # (N,)
headings = meta["headings"]
panoids = meta["panoids"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """
    Returns indices of top_k candidates within radius, ranked by cosine similarity.
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    in_radius = np.where(distances <= radius_km)[0]
    
    if len(in_radius) == 0:
        return []

    # Cosine similarity (descriptors should already be L2-normalized)
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    subset = descriptors[in_radius]
    subset_normed = subset / (np.linalg.norm(subset, axis=1, keepdims=True) + 1e-8)
    
    sims = subset_normed @ q  # (M,)
    ranked = np.argsort(sims)[::-1][:top_k]
    
    return [(in_radius[i], sims[i]) for i in ranked]

# Usage
candidates = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_km=1.0)
for idx, score in candidates[:5]:
    print(f"panoid={panoids[idx]}, lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, score={score:.4f}")
```

### Use Flipped Descriptor for Robustness

```python
from PIL import Image, ImageOps
import numpy as np

def get_robust_descriptor(model, image_path, device):
    """Extract both normal and horizontally-flipped descriptors, return max-pooled."""
    img = Image.open(image_path).convert("RGB")
    img_flip = ImageOps.mirror(img)
    
    desc = extract_descriptor(model, img, device)
    desc_flip = extract_descriptor(model, img_flip, device)
    
    # Element-wise max across both views
    combined = np.maximum(desc, desc_flip)
    combined /= np.linalg.norm(combined) + 1e-8
    return combined
```

### Standalone Index Building (Large Datasets)

For large areas (5km+ radius), use `build_index.py` directly instead of the GUI:

```bash
# Build index from existing cosplace_parts/ chunks
python build_index.py
```

This compiles all `cosplace_parts/*.npz` chunks into the unified `index/` directory.

---

## Configuration Reference

All configuration happens through the GUI or by editing `test_super.py` constants:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Grid resolution | `300` | Meters between sampled panorama grid points — don't change |
| Top candidates | `500–1000` | Number of CosPlace candidates passed to Stage 2 |
| Heading refinement steps | `15°` | Step size for ±45° heading sweep |
| Heading refinement candidates | Top 15 | How many matches get heading-refined |
| Spatial cluster cell | `50m` | Grid cell size for consensus clustering |
| Ultra Mode neighborhood | `100m` | Expansion radius around best match |
| Ultra Mode descriptor hop threshold | `50 inliers` | Below this → try descriptor hopping |

---

## Ultra Mode

Enable via the **Ultra Mode** checkbox in the GUI. Use for:
- Night photos
- Motion-blurred images
- Low-texture scenes (fog, overcast, featureless walls)

Ultra Mode adds:
1. **LoFTR** — detector-free dense matching (handles blur/low contrast)
2. **Descriptor hopping** — re-searches index using the matched panorama's clean descriptor
3. **Neighborhood expansion** — searches all panoramas within 100m of best match

**Requires:** `pip install kornia`

---

## Common Patterns

### Pattern: Index Multiple Cities into One Index

```bash
# In GUI: Create mode → index Paris (48.8566, 2.3522, 5km radius)
# In GUI: Create mode → index London (51.5074, -0.1278, 5km radius)
# In GUI: Create mode → index Tel Aviv (32.0853, 34.7818, 3km radius)
# All go into the same index/
# Search by specifying the city's coordinates + radius — no conflicts
```

### Pattern: OSINT Workflow (Unknown Location)

1. Enable **AI Coarse** mode (set `GEMINI_API_KEY`)
2. Gemini analyzes signage, architecture, vegetation, vehicles to estimate country/city
3. System auto-selects search center + radius
4. Run full pipeline → result

### Pattern: Known Region Workflow

1. Use **Manual** mode
2. Enter approximate center (you know it's "somewhere in central Paris")
3. Set radius to cover uncertainty (e.g., 5km)
4. Run search

### Pattern: Verify a Match Programmatically

```python
# After getting a result, verify confidence
def interpret_confidence(inliers, cluster_size, uniqueness_ratio):
    """
    inliers: RANSAC-verified keypoint matches for best candidate
    cluster_size: how many top-k candidates cluster at same location
    uniqueness_ratio: best_inliers / second_best_inliers (different location)
    """
    if inliers > 100 and uniqueness_ratio > 2.0:
        return "HIGH — very reliable"
    elif inliers > 50 and cluster_size >= 3:
        return "MEDIUM — likely correct, verify visually"
    elif inliers > 20:
        return "LOW — treat as approximate, use Ultra Mode"
    else:
        return "INSUFFICIENT — expand search radius or use Ultra Mode"
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11  # Replace 3.11 with your Python version
```

### CUDA out of memory
- Reduce `top_k` candidates in Stage 2
- Use `DISK` instead of `ALIKED` (fewer keypoints)
- Restart the process — models accumulate on VRAM between searches

### No candidates found / zero results
- Verify the area is actually indexed (`cosplace_parts/` should contain `.npz` files)
- Increase search radius
- Check that `index/cosplace_descriptors.npy` exists — if not, the auto-build step may have failed; run `python build_index.py` manually

### Very low inlier counts (< 20) on good images
- The indexed heading may not match query viewing direction → enable heading refinement (already on by default)
- Try Ultra Mode for LoFTR + neighborhood expansion
- The area may not be densely indexed — re-index with smaller grid resolution or larger radius

### Indexing stalls / crashes mid-way
- Safe to restart — indexing is incremental, resumes from last saved chunk in `cosplace_parts/`
- Check network connectivity (requires internet to fetch Street View tiles during indexing)

### ImportError: No module named 'lightglue'
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### MPS (Apple Silicon) errors
```bash
# Ensure PyTorch nightly or >= 2.0 with MPS support
pip install --upgrade torch torchvision
```

---

## Models Reference

| Model | Source | Used For |
|-------|--------|----------|
| CosPlace | `gmberton/cosplace` | Global 512-dim descriptor (Stage 1) |
| ALIKED | `naver/alike` | Keypoints on CUDA (Stage 2) |
| DISK | `cvlab-epfl/disk` | Keypoints on MPS/CPU (Stage 2) |
| LightGlue | `cvg/LightGlue` | Deep feature matching (Stage 2) |
| LoFTR | `zju3dv/LoFTR` | Dense matching, Ultra Mode only |

All models download weights automatically on first use.
```
