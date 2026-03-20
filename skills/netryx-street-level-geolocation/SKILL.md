```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - identify location from street view photo
  - netryx geolocation
  - reverse geolocate image
  - find where a photo was taken
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls and indexes street-view panoramas, extracts visual fingerprints using CosPlace, then verifies matches using ALIKED/DISK keypoint extraction and LightGlue feature matching — all running on your own hardware with no cloud dependency.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # Required
pip install kornia                                        # Optional: Ultra Mode (LoFTR)
```

### Optional: Gemini API for AI Coarse mode

```bash
export GEMINI_API_KEY="your_key_here"
```

### macOS tkinter fix (if GUI appears blank)

```bash
brew install python-tk@3.11   # match your Python version
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface — all indexing and searching is done through it.

---

## Project Structure

```
netryx/
├── test_super.py          # Main application — GUI, indexing, search
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim visual fingerprints
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Core Workflow

### Step 1 — Create an Index

Before searching, index the area of interest. This crawls street-view panoramas and stores CosPlace fingerprints.

**In GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius (start 0.5–1 km for testing, 5–10 km for production)
4. Set grid resolution (default 300 — do not change)
5. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

Indexing is incremental — safe to interrupt and resume.

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini infers region from visual cues (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result displays on map with GPS coordinates + confidence score

### Ultra Mode

Enable **Ultra Mode** checkbox for difficult images (night, blur, low texture). Adds:
- LoFTR dense matching (requires `kornia`)
- Descriptor hopping (re-searches index using matched panorama's descriptor)
- Neighborhood expansion (searches within 100m of best match)

---

## The Three-Stage Pipeline

### Stage 1 — Global Retrieval (CosPlace)

```
Query image → 512-dim CosPlace descriptor
             + flipped image descriptor
             → cosine similarity vs. entire index
             → radius-filtered (haversine)
             → top 500–1000 candidates
```

Runs in under 1 second (single matrix multiplication).

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)

```
For each candidate:
  Download GSV panorama (8 tiles, stitched)
  → Crop at indexed heading
  → Multi-FOV crops: 70°, 90°, 110°
  → ALIKED (CUDA) / DISK (MPS/CPU): extract keypoints
  → LightGlue: match keypoints vs. query
  → RANSAC: filter geometrically inconsistent matches
  → Score = verified inlier count
```

Processes 300–500 candidates in 2–5 minutes depending on hardware.

### Stage 3 — Refinement

```
Top 15 candidates:
  → Heading refinement: ±45° at 15° steps × 3 FOVs
  → Spatial consensus: cluster into 50m cells
  → Confidence scoring: clustering density + uniqueness ratio
  → Final GPS + confidence score
```

---

## Platform-Specific Feature Extractor Selection

| Platform       | Extractor       | Keypoints |
|----------------|-----------------|-----------|
| CUDA (NVIDIA)  | ALIKED          | 1024      |
| MPS (Mac M1+)  | DISK            | 768       |
| CPU            | DISK            | 768       |

Netryx auto-detects the platform — no manual configuration needed.

---

## Index Architecture — Multi-City Support

All city indexes share a single unified index. Search is scoped by coordinates + radius at query time:

```
# Index Paris
Create mode → lat=48.8566, lon=2.3522, radius=5km → cosplace_parts/*.npz

# Index London  
Create mode → lat=51.5074, lon=-0.1278, radius=5km → cosplace_parts/*.npz

# Both merge into the same index/
# Search Paris only:  center=48.8566,2.3522  radius=5km
# Search London only: center=51.5074,-0.1278 radius=10km
```

---

## Using `cosplace_utils.py` Directly

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model, transform, device = load_cosplace_model()

# Extract descriptor from an image
img = Image.open("query.jpg").convert("RGB")
descriptor = get_descriptor(img, model, transform, device)
# descriptor.shape → (512,)

# Extract from flipped image too (catches reversed perspectives)
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = get_descriptor(img_flipped, model, transform, device)
```

---

## Using `build_index.py` for Large Datasets

For large area indexing (5 km+), use the standalone builder instead of the GUI:

```bash
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5000 \
  --resolution 300
```

This compiles all `cosplace_parts/*.npz` chunks into the searchable `index/` directory.

---

## Common Patterns

### Pattern 1: OSINT — Known approximate location

```
Scenario: You have a photo from "somewhere in Paris"
1. Index Paris (radius 5–10 km around city center)
2. Search mode → Manual
3. Center: 48.8566, 2.3522 | Radius: 10 km
4. Upload photo → Run Search
```

### Pattern 2: Conflict/event geolocation — Unknown region

```
Scenario: Photo with no location metadata, unknown country
1. Enable AI Coarse mode (requires GEMINI_API_KEY)
2. Gemini analyzes signs, architecture, vegetation → returns probable region
3. System auto-sets center + radius from Gemini output
4. Proceed with standard search pipeline
```

### Pattern 3: Difficult image (night, blur, low contrast)

```
Scenario: Night photo or motion-blurred image fails standard search
1. Enable Ultra Mode checkbox
2. Runs LoFTR (detector-free) in addition to ALIKED/DISK
3. If <50 inliers: descriptor hopping re-searches using matched panorama
4. Neighborhood expansion: checks all panoramas within 100m of best match
```

### Pattern 4: Incremental indexing (interrupted run)

```
If indexing is interrupted, just re-run with same parameters.
The system checks cosplace_parts/ for existing chunks and skips already-indexed panoramas.
```

---

## Troubleshooting

### GUI appears blank on macOS

```bash
brew install python-tk@3.11   # match your exact Python version
```

### `ModuleNotFoundError: No module named 'lightglue'`

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### LoFTR / Ultra Mode not available

```bash
pip install kornia
```

### CUDA out of memory

- Reduce keypoint count by editing the ALIKED config in `test_super.py`
- Or use CPU mode (slower but no VRAM limit)

### Index search returns no candidates

- Your search radius may be too small or the area hasn't been indexed
- Check `cosplace_parts/` directory exists and contains `.npz` files
- Run **Create Index** for the target area first

### Very low confidence scores (<30%)

- Try Ultra Mode
- Increase search radius — the correct panorama may be at the boundary
- The area may have sparse street-view coverage

### Panorama download failures

- Netryx requires internet access during search (to fetch GSV tiles)
- Check network connectivity
- If specific panoramas 404, they may have been removed from GSV — the pipeline skips them

---

## Hardware Recommendations

| Setup              | Indexing Speed | Search Speed |
|--------------------|---------------|--------------|
| NVIDIA RTX 3080+   | Fastest       | 2–3 min      |
| Apple M2/M3/M4 Max | Fast          | 3–5 min      |
| NVIDIA GTX 1060    | Moderate      | 5–8 min      |
| CPU only           | Slow          | 15–30 min    |

Minimum: 4 GB GPU VRAM, 8 GB RAM  
Recommended: 8 GB+ VRAM, 16 GB RAM, SSD for index storage

---

## Key Models Reference

| Model     | Role                          | Hardware  |
|-----------|-------------------------------|-----------|
| CosPlace  | Global visual fingerprinting  | All       |
| ALIKED    | Local keypoint extraction     | CUDA only |
| DISK      | Local keypoint extraction     | MPS/CPU   |
| LightGlue | Deep feature matching         | All       |
| LoFTR     | Dense matching (Ultra Mode)   | All       |
```
