```markdown
---
name: netryx-street-level-geolocation
description: Use Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from any street photo using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - run netryx geolocation
  - build a street view index
  - search netryx index
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a local-first geolocation engine that identifies the exact GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a visual index using CosPlace descriptors, then matches a query image through a three-stage pipeline: global retrieval → local geometric verification (ALIKED/DISK + LightGlue) → spatial refinement. Sub-50m accuracy. No internet landmark matching — it searches the physical world.

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

### Optional: Gemini API for AI Coarse mode

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

### 1. Create an Index (crawl + embed an area)

In the GUI:
1. Select **Create** mode
2. Enter center lat/lon of the area to index
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Index is saved incrementally to `cosplace_parts/` — safe to interrupt and resume.

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

### 2. Search (geolocate a photo)

In the GUI:
1. Select **Search** mode
2. Upload street-level photo
3. Choose search method:
   - **Manual**: provide known approximate center lat/lon + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py          # Main entry point — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Pipeline Deep Dive

### Stage 1 — Global Retrieval (CosPlace)

```python
# cosplace_utils.py handles this
# Each panorama → 512-dim descriptor
# Query → descriptor + flipped descriptor
# Cosine similarity against full index
# Radius filter (haversine) → top 500–1000 candidates
# Runtime: <1 second (single matrix multiply)
```

### Stage 2 — Local Geometric Verification

```python
# For each candidate:
# 1. Download Street View panorama (8 tiles, stitched)
# 2. Crop at indexed heading angle
# 3. Generate multi-FOV crops: 70°, 90°, 110°
# 4. Extract keypoints: ALIKED (CUDA) or DISK (MPS/CPU)
# 5. LightGlue deep feature matching vs query keypoints
# 6. RANSAC geometric verification → inlier count
# Runtime: 2–5 minutes for 300–500 candidates
```

### Stage 3 — Refinement

```python
# Heading refinement: top 15 candidates × ±45° offsets (15° steps) × 3 FOVs
# Spatial consensus: cluster matches into 50m cells
# Confidence scoring: clustering quality + uniqueness ratio (best vs runner-up)
```

### Ultra Mode (optional, for hard images)

```python
# Adds:
# - LoFTR: detector-free dense matching (handles blur/low-contrast)
# - Descriptor hopping: re-search index from matched panorama's descriptor
# - Neighborhood expansion: search all panoramas within 100m of best match
```

---

## Using CosPlace Utils Directly

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model = get_cosplace_model()

# Extract 512-dim descriptor from any image
img = Image.open("street_photo.jpg").convert("RGB")
descriptor = get_descriptor(model, img)  # shape: (512,)

# Also extract flipped version for robustness
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = get_descriptor(model, img_flipped)
```

---

## Building Index Programmatically

```python
# Use build_index.py for large-scale or headless indexing
# Run standalone for high-performance batch processing:
python build_index.py \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 2.0 \
    --resolution 300
```

---

## Multi-Area Index Strategy

The index is unified — all cities share one index file. Search uses coordinates + radius to filter.

```python
# Index Paris
# GUI: Create, lat=48.8566, lon=2.3522, radius=5km

# Index London  
# GUI: Create, lat=51.5074, lon=-0.1278, radius=5km

# Both stored in same cosplace_parts/ and index/
# Search Paris: center=48.8566,2.3522 radius=5km → only Paris results
# Search London: center=51.5074,-0.1278 radius=5km → only London results
```

---

## Hardware & Device Behavior

| Feature          | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU      |
|------------------|---------------|---------------------|----------|
| Feature extractor | ALIKED 1024kp | DISK 768kp          | DISK     |
| Speed            | Fastest       | Fast                | Slow     |
| Min VRAM         | 4GB           | Shared RAM          | N/A      |
| LoFTR (Ultra)    | ✅            | ✅                  | ✅ (slow)|

---

## Models Reference

| Model | Role | Source |
|-------|------|--------|
| CosPlace | Global visual place recognition (512-dim descriptor) | `gmberton/cosplace` |
| ALIKED | Local keypoint extraction (CUDA) | `naver/alike` |
| DISK | Local keypoint extraction (MPS/CPU) | `cvlab-epfl/disk` |
| LightGlue | Deep feature matching | `cvg/LightGlue` |
| LoFTR | Detector-free dense matching (Ultra Mode) | `zju3dv/LoFTR` via kornia |

---

## Common Patterns

### Geolocate a known-region photo (fastest path)

```
1. Index the target region (one-time setup)
2. GUI → Search → Manual → enter approximate lat/lon + 2km radius
3. Upload photo → Run Search → Start Full Search
4. Read coordinates from result map
```

### Fully blind geolocation (no prior knowledge)

```
1. Set GEMINI_API_KEY environment variable
2. GUI → Search → AI Coarse
3. Upload photo — Gemini identifies region from visual clues
4. System auto-sets search center + radius
5. Run Search → Start Full Search
```

### Difficult image (night / blur / low texture)

```
1. Enable Ultra Mode checkbox before searching
2. Adds LoFTR dense matching + descriptor hopping + 100m neighborhood expansion
3. Significantly slower but catches matches standard pipeline misses
```

---

## Troubleshooting

**GUI appears blank on macOS**
```bash
brew install python-tk@3.11  # match your Python version
```

**LightGlue import error**
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install from PyPI — must use GitHub source
```

**CUDA out of memory**
- Reduce candidate count in GUI search settings
- Use DISK instead of ALIKED (fewer keypoints)
- Disable Ultra Mode

**Index search returns no results**
- Verify the search center lat/lon matches an indexed area
- Check that `index/cosplace_descriptors.npy` and `index/metadata.npz` exist
- If only `cosplace_parts/` exists, the index hasn't been compiled — re-run indexing to trigger auto-build

**Slow indexing**
- Use `build_index.py` standalone script instead of GUI for large areas
- Index is saved incrementally — interrupting and resuming is safe

**Low confidence / wrong location**
- Increase search radius
- Enable Ultra Mode
- Try Manual mode with a tighter radius if you have approximate knowledge of location
- Ensure the indexed area includes the photo's actual location (check coverage)

**MPS (Apple Silicon) errors**
```bash
# Ensure PyTorch MPS build is installed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or use the standard pip install which includes MPS on macOS
```
```
