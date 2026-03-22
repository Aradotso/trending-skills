---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - build a street view index
  - reverse geolocate image
  - find where a photo was taken
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls street-view panoramas, indexes them as CosPlace visual fingerprints, then matches query images using a three-stage pipeline: global retrieval → local geometric verification → spatial refinement. Sub-50m accuracy, no internet presence of the target location required.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (deep feature matcher)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11   # match your Python version
```

### Optional: Gemini API for AI Coarse geolocation
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

**GPU backends:**
- NVIDIA → CUDA (uses ALIKED, 1024 keypoints)
- Apple Silicon → MPS (uses DISK, 768 keypoints)
- CPU → works, significantly slower

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It covers both **Create** (indexing) and **Search** modes.

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area before searching. The indexer crawls Street View panoramas and extracts 512-dim CosPlace fingerprints.

**In the GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|------------|---------------|------------|
| 0.5 km | ~500       | 30 min        | ~60 MB     |
| 1 km   | ~2,000     | 1–2 hrs       | ~250 MB    |
| 5 km   | ~30,000    | 8–12 hrs      | ~3 GB      |
| 10 km  | ~100,000   | 24–48 hrs     | ~7 GB      |

Indexing is **resumable** — if interrupted, it picks up from where it left off.

**For large-scale indexing, use the standalone builder:**
```bash
python build_index.py
```

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual**: Enter approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on a map

---

## Project Structure

```
netryx/
├── test_super.py           # Main app: GUI, indexing, search pipeline
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone high-perf index builder (large datasets)
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (searchable)
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## How the Pipeline Works

### Stage 1 — Global Retrieval (CosPlace)
```
Query image → 512-dim CosPlace descriptor
             + flipped image descriptor
             → cosine similarity vs. entire index
             → haversine radius filter
             → top 500–1000 candidates
```
Runs in under 1 second (single matrix multiply).

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
```
For each candidate:
  Download panorama (8 tiles, stitched)
  → Rectilinear crop at indexed heading
  → Multi-FOV crops: 70°, 90°, 110°
  → ALIKED (CUDA) or DISK (MPS/CPU) keypoints
  → LightGlue matching vs. query keypoints
  → RANSAC inlier filtering
```
Processes 300–500 candidates in 2–5 minutes depending on hardware.

### Stage 3 — Refinement
- **Heading refinement**: ±45° at 15° steps × 3 FOVs for top 15 candidates
- **Spatial consensus**: Cluster matches into 50m cells; prefer clusters over outliers
- **Confidence scoring**: Geographic clustering + uniqueness ratio (best vs. runner-up)

### Ultra Mode
Enable via the **Ultra Mode** checkbox for night shots, blur, or low-texture images:
- **LoFTR**: Detector-free dense matcher (handles blur/low-contrast)
- **Descriptor hopping**: Re-searches index using the matched panorama's descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

---

## Index Architecture

The index is **city-agnostic** — all areas share one unified index, filtered at query time by coordinates + radius:

```python
# Conceptual search flow (inside test_super.py)
# 1. Extract query descriptor
query_desc = cosplace_model(query_image)           # shape: (512,)
flipped_desc = cosplace_model(flip(query_image))   # shape: (512,)

# 2. Load index
all_descs = np.load("index/cosplace_descriptors.npy")  # shape: (N, 512)
metadata = np.load("index/metadata.npz")

# 3. Radius filter (haversine)
center_lat, center_lon = 48.8566, 2.3522
radius_km = 2.0
mask = haversine_filter(metadata["lats"], metadata["lons"],
                        center_lat, center_lon, radius_km)

# 4. Cosine similarity search
filtered_descs = all_descs[mask]
sims = filtered_descs @ query_desc  # cosine similarity (normalized)
top_k_idx = np.argsort(sims)[::-1][:500]
candidates = metadata[mask][top_k_idx]
```

Multi-city indexing example:
```
Index Paris (48.8566, 2.3522, r=5km) → adds ~30k panoramas
Index London (51.5074, -0.1278, r=5km) → adds ~30k more

Search Paris: center=48.8566,2.3522, radius=5km → only Paris results
Search London: center=51.5074,-0.1278, radius=5km → only London results
```

---

## Working with CosPlace Descriptors

```python
# cosplace_utils.py usage
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (downloads weights on first run)
model = load_cosplace_model(device="cuda")  # or "mps", "cpu"

# Extract descriptor from any street image
img = Image.open("query.jpg")
descriptor = extract_descriptor(model, img, device="cuda")
# descriptor.shape → (512,) normalized float32

# Also extract flipped (catches reversed perspectives)
descriptor_flipped = extract_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT), device="cuda")
```

---

## Building a Custom Index Programmatically

```python
# High-level flow mirroring build_index.py
import numpy as np
from pathlib import Path

# After indexing, parts are saved as:
# cosplace_parts/part_000.npz, part_001.npz, ...
# Each .npz contains: descriptors, lats, lons, headings, panoids

def compile_index(parts_dir="cosplace_parts", out_dir="index"):
    Path(out_dir).mkdir(exist_ok=True)
    all_descs, all_lats, all_lons, all_headings, all_panoids = [], [], [], [], []

    for part_file in sorted(Path(parts_dir).glob("*.npz")):
        data = np.load(part_file)
        all_descs.append(data["descriptors"])
        all_lats.append(data["lats"])
        all_lons.append(data["lons"])
        all_headings.append(data["headings"])
        all_panoids.append(data["panoids"])

    np.save(f"{out_dir}/cosplace_descriptors.npy",
            np.vstack(all_descs).astype(np.float32))
    np.savez(f"{out_dir}/metadata.npz",
             lats=np.concatenate(all_lats),
             lons=np.concatenate(all_lons),
             headings=np.concatenate(all_headings),
             panoids=np.concatenate(all_panoids))
    print(f"Index compiled: {len(all_lats)} panorama views")
```

---

## Common Patterns

### Pattern 1: OSINT — Known region, unknown exact location
```
Mode: Manual
Center: <rough city center lat/lon>
Radius: 5–10 km
Ultra Mode: OFF (try first)
```

### Pattern 2: Blind geolocation — No prior knowledge
```
Mode: AI Coarse (requires GEMINI_API_KEY)
Gemini analyzes: signs, architecture, vegetation, license plates
→ Returns estimated region → feeds into Manual search
```

### Pattern 3: Difficult images (night/blur/low texture)
```
Mode: Manual or AI Coarse
Ultra Mode: ON
  - LoFTR handles blur where ALIKED/DISK fail
  - Descriptor hopping re-anchors from clean matched panorama
  - Neighborhood expansion ±100m around best match
```

### Pattern 4: Large area indexing (city-scale)
```bash
# Use the standalone builder instead of GUI for long-running jobs
python build_index.py
# Resumable — safe to Ctrl+C and restart
```

---

## Troubleshooting

### GUI appears blank (macOS)
```bash
brew install python-tk@3.11   # or python-tk@3.10, match your Python version
```

### LightGlue import error
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Must be installed from GitHub, not PyPI
```

### LoFTR not available (Ultra Mode disabled)
```bash
pip install kornia
# kornia provides the LoFTR implementation used in Ultra Mode
```

### CUDA out of memory
- Reduce candidate count in the GUI (lower top-K)
- Use DISK instead of ALIKED (fewer keypoints: 768 vs 1024)
- Ensure no other GPU processes are running

### Low confidence / wrong result
1. Increase search radius — the correct location may be outside your radius
2. Enable Ultra Mode
3. Try flipping the query image horizontally (catches mirrored perspectives)
4. Increase grid resolution when re-indexing the area (denser panorama coverage)
5. Verify the area is actually indexed — check `cosplace_parts/` is non-empty

### Index search returns 0 candidates
- The area may not be indexed yet — run **Create Index** first
- Check that `index/cosplace_descriptors.npy` and `index/metadata.npz` exist
- Confirm center coordinates + radius actually overlap your indexed region

### Indexing stalls / no panoramas found
- Street View API access may be rate-limited — wait and retry
- Some areas have sparse or no Street View coverage
- Check internet connectivity (indexing requires Street View tile downloads)

---

## Models Reference

| Model | Role | Backend |
|-------|------|---------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim visual fingerprint | CUDA / MPS / CPU |
| [ALIKED](https://github.com/naver/alike) | Local keypoint extraction | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoint extraction | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | CUDA / MPS / CPU |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense matching (Ultra Mode) | via `kornia` |

---

## Data Flow Summary

```
CREATE MODE:
  Grid points (lat/lon)
  → Street View panorama tiles (8 tiles → stitched)
  → CosPlace descriptor extraction
  → cosplace_parts/*.npz (saved incrementally)
  → Auto-compiled to index/cosplace_descriptors.npy + metadata.npz

SEARCH MODE:
  Query image
  → CosPlace descriptor + flipped descriptor
  → Cosine similarity search (radius-filtered)
  → Top 500 candidates
  → Download panoramas → multi-FOV crops (70°/90°/110°)
  → ALIKED/DISK keypoints → LightGlue matching → RANSAC
  → Heading refinement (±45°) → Spatial consensus clustering
  → GPS coordinates + confidence score
```
