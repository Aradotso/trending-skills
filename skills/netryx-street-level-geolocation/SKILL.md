```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a locally-hosted open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo locally
  - find GPS coordinates from an image
  - run Netryx geolocation
  - index street view panoramas
  - street level geolocation with computer vision
  - use Netryx to locate an image
  - build a geolocation index for a city
  - open source GeoGuessr-style geolocation
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies the precise GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a searchable visual index using CosPlace (global descriptors), then verifies matches with ALIKED/DISK keypoints and LightGlue deep feature matching. No cloud APIs required after setup — everything runs on your hardware.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (install from GitHub, not PyPI)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11   # match your Python version
```

### Optional: Gemini API key for AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"   # get free key at aistudio.google.com
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the main entry point for all operations: indexing, searching, and viewing results on a map.

---

## Core Workflow

### Step 1 — Build an Index

You must index an area before you can search it. Indexing crawls street-view panoramas, extracts 512-dim CosPlace fingerprints, and saves them to disk.

**GUI method:**
1. Select **Create** mode
2. Enter center lat/lon, radius (km), grid resolution (default: 300)
3. Click **Create Index**

**Indexing time estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — if interrupted, it continues from the last saved chunk on restart.

All indexed areas live in the **same unified index**. Multiple cities can coexist; the radius filter at search time isolates the right area automatically.

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide center lat/lon + radius (use when you know the rough region)
   - **AI Coarse**: Gemini analyzes visual cues (signs, architecture, vegetation) to estimate the region — requires `GEMINI_API_KEY`
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on a live map

---

## Project Structure

```
netryx/
├── test_super.py           # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone high-performance index builder (large datasets)
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks (auto-created during indexing)
└── index/
    ├── cosplace_descriptors.npy    # All 512-dim descriptors (matrix)
    └── metadata.npz                # Coordinates, headings, panorama IDs
```

---

## Three-Stage Pipeline (How It Works)

### Stage 1 — Global Retrieval (CosPlace)
- Extracts a 512-dim descriptor from the query image (+ horizontally flipped version)
- Compares against all index entries via cosine similarity (single matrix multiply, <1 second)
- Applies haversine radius filter to restrict to the specified area
- Returns top 500–1000 visual candidates

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
- Downloads each candidate panorama (8 tiles, stitched)
- Crops at 3 fields of view: 70°, 90°, 110° (handles zoom mismatch)
- Extracts local keypoints:
  - **CUDA**: ALIKED (1024 keypoints)
  - **MPS/CPU**: DISK (768 keypoints)
- LightGlue matches query keypoints vs. candidate keypoints
- RANSAC filters geometrically inconsistent matches
- Processes 300–500 candidates in **2–5 minutes** depending on hardware

### Stage 3 — Refinement
- **Heading refinement**: Tests ±45° at 15° steps × 3 FOVs for top 15 candidates
- **Spatial consensus**: Clusters matches into 50m cells; prefers clusters over lone outliers
- **Confidence scoring**: Evaluates clustering density + uniqueness ratio (best vs. runner-up)

### Ultra Mode (optional checkbox)
Enable for blurry, night, or low-texture images. Adds:
- **LoFTR**: Detector-free dense matching (no keypoints needed)
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

---

## Code Examples

### Extract a CosPlace descriptor programmatically

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-selects CUDA > MPS > CPU)
model, device = load_cosplace_model()

# Extract descriptor from an image file
img = Image.open("query.jpg").convert("RGB")
descriptor = get_descriptor(model, img, device)  # shape: (512,)

print(f"Descriptor shape: {descriptor.shape}")
print(f"Device used: {device}")
```

### Search the index against a query descriptor

```python
import numpy as np

# Load pre-built index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]        # (N,)
lons = meta["lons"]        # (N,)
headings = meta["headings"]
panoids = meta["panoids"]

# Cosine similarity search
query_desc = descriptor / np.linalg.norm(descriptor)
index_norms = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
similarities = index_norms @ query_desc                    # (N,)

# Radius filter using haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

center_lat, center_lon = 48.8566, 2.3522   # Paris
radius_m = 2000                             # 2 km

distances = haversine(center_lat, center_lon, lats, lons)
in_radius = distances < radius_m

# Rank candidates within radius
masked_sim = np.where(in_radius, similarities, -1)
top_indices = np.argsort(masked_sim)[::-1][:500]

for i in top_indices[:10]:
    print(f"panoid={panoids[i]}  lat={lats[i]:.6f}  lon={lons[i]:.6f}  "
          f"heading={headings[i]}  similarity={similarities[i]:.4f}")
```

### Build an index programmatically (large area, no GUI)

```python
# Use build_index.py for large-scale indexing without the GUI
# Run from the project root:
import subprocess
subprocess.run([
    "python", "build_index.py",
    "--lat", "48.8566",
    "--lon", "2.3522",
    "--radius", "5.0",       # km
    "--resolution", "300",
    "--output", "cosplace_parts/"
])
```

### Run LightGlue matching between two crops

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load extractor and matcher
extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

# Load images (LightGlue expects tensors in [0,1], shape C×H×W)
query = load_image("query.jpg").to(device)
candidate = load_image("candidate_crop.jpg").to(device)

# Extract features
feats0 = extractor.extract(query)
feats1 = extractor.extract(candidate)

# Match
with torch.no_grad():
    matches01 = matcher({"image0": feats0, "image1": feats1})

# Unpack
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
kpts0 = feats0["keypoints"]
kpts1 = feats1["keypoints"]
matches = matches01["matches"]           # (M, 2) index pairs
scores = matches01["matching_scores0"]   # (M,)

matched0 = kpts0[matches[:, 0]]
matched1 = kpts1[matches[:, 1]]
print(f"Matched keypoints: {len(matches)}")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def verify_matches_ransac(kpts0, kpts1, matches, threshold=3.0):
    """
    Returns number of RANSAC inliers — higher = better geometric match.
    kpts0, kpts1: numpy arrays of shape (N, 2)
    matches: numpy array of shape (M, 2)
    """
    if len(matches) < 8:
        return 0

    src_pts = kpts0[matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kpts1[matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    if mask is None:
        return 0

    inliers = int(mask.sum())
    return inliers

# Usage
kpts0_np = matched0.cpu().numpy()
kpts1_np = matched1.cpu().numpy()
match_indices = matches.cpu().numpy()
inliers = verify_matches_ransac(kpts0_np, kpts1_np, match_indices)
print(f"RANSAC inliers: {inliers}")
# >50 inliers = strong match, >100 = very confident
```

### Spatial consensus clustering

```python
import numpy as np
from collections import defaultdict

def cluster_candidates(candidates, cell_size_m=50):
    """
    candidates: list of dicts with keys 'lat', 'lon', 'inliers'
    Returns the cluster center with the highest total inlier count.
    """
    def to_cell(lat, lon):
        # ~50m per cell at most latitudes
        cell_lat = round(lat / (cell_size_m / 111320), 0)
        cell_lon = round(lon / (cell_size_m / (111320 * np.cos(np.radians(lat)))), 0)
        return (cell_lat, cell_lon)

    clusters = defaultdict(list)
    for c in candidates:
        cell = to_cell(c["lat"], c["lon"])
        clusters[cell].append(c)

    best_cell = max(clusters, key=lambda k: sum(x["inliers"] for x in clusters[k]))
    best_group = clusters[best_cell]

    total_inliers = sum(x["inliers"] for x in best_group)
    avg_lat = np.mean([x["lat"] for x in best_group])
    avg_lon = np.mean([x["lon"] for x in best_group])

    return {"lat": avg_lat, "lon": avg_lon, "inliers": total_inliers, "count": len(best_group)}

result = cluster_candidates(top_candidates)
print(f"Best cluster: {result['lat']:.6f}, {result['lon']:.6f} "
      f"({result['count']} matches, {result['inliers']} total inliers)")
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| Grid resolution | 300 | Sampling density for panorama crawl. Don't change unless you know why. |
| Retrieval candidates | 500–1000 | How many CosPlace results feed into Stage 2 |
| FOVs tested | 70°, 90°, 110° | Handles zoom differences between query and indexed view |
| Heading refinement | ±45° at 15° steps | Applied to top 15 candidates |
| Spatial cluster cell | 50m | Cell size for consensus clustering |
| RANSAC threshold | 3.0px | Reprojection error for inlier acceptance |
| Ultra neighborhood | 100m | Expansion radius for neighborhood search in Ultra Mode |

---

## Platform-Specific Notes

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|--------------|---------------------|-----|
| Feature extractor | ALIKED (1024 kp) | DISK (768 kp) | DISK (768 kp) |
| LoFTR (Ultra Mode) | ✅ Full speed | ⚠️ Slow | ⚠️ Very slow |
| Stage 2 speed | Fastest | Fast | Usable |
| Min VRAM | 4GB | 4GB unified | N/A |

Auto-detection order: CUDA → MPS → CPU. No manual config needed.

---

## Common Patterns

### Pattern: Automating headless search (no GUI)

The GUI in `test_super.py` can be bypassed by calling its pipeline functions directly. Study the `run_search()` function in `test_super.py` and instantiate the pipeline components:

```python
# Pseudocode — adapt to actual function signatures in test_super.py
from test_super import GeolocalizationPipeline

pipeline = GeolocalizationPipeline(
    index_dir="index/",
    parts_dir="cosplace_parts/"
)

result = pipeline.search(
    query_image_path="my_photo.jpg",
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    ultra_mode=False
)

print(f"Predicted: {result.lat:.6f}, {result.lon:.6f}")
print(f"Confidence: {result.confidence:.2f}")
```

### Pattern: Multi-city index strategy

```python
# Index multiple cities into one unified index
# Run indexing sequentially — all chunks land in cosplace_parts/
# The index auto-merges on next search or explicit build_index call

areas = [
    (48.8566, 2.3522, 3.0),    # Paris, 3km radius
    (51.5074, -0.1278, 3.0),   # London, 3km radius
    (40.7128, -74.0060, 3.0),  # NYC, 3km radius
]

for lat, lon, radius in areas:
    print(f"Indexing {lat}, {lon} ...")
    # Run via GUI or build_index.py for each area
    # All output goes to cosplace_parts/ — no collision
```

### Pattern: Evaluating match confidence

```python
def confidence_score(best_inliers, second_best_inliers, cluster_count):
    """
    Heuristic confidence score (0.0 – 1.0).
    - High inliers + big uniqueness gap + cluster > 1 = high confidence
    """
    if second_best_inliers == 0:
        uniqueness = 1.0
    else:
        uniqueness = min(1.0, (best_inliers - second_best_inliers) / best_inliers)

    cluster_bonus = min(0.2, cluster_count * 0.05)
    inlier_score = min(0.8, best_inliers / 200.0)

    return round(inlier_score + uniqueness * 0.3 + cluster_bonus, 3)

# Example
score = confidence_score(best_inliers=142, second_best_inliers=38, cluster_count=4)
print(f"Confidence: {score}")   # e.g. 0.761
```

---

## Troubleshooting

### GUI is blank on macOS
```bash
brew install python-tk@3.11   # match your exact Python version
```

### `ModuleNotFoundError: No module named 'lightglue'`
LightGlue must be installed from GitHub, not PyPI:
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### LoFTR not available (Ultra Mode fails)
```bash
pip install kornia
```

### Index not found / empty results
- Check that `index/cosplace_descriptors.npy` and `index/metadata.npz` exist
- If only `cosplace_parts/` exists, the index hasn't been compiled yet — run a search or `build_index.py` to trigger auto-build
- Confirm your search radius actually overlaps indexed areas

### CUDA out of memory
- Reduce `max_num_keypoints` in ALIKED (default 1024 → try 512)
- Reduce the number of Stage 2 candidates (modify top-k in retrieval step)
- Disable Ultra Mode

### MPS errors on Apple Silicon
```bash
# Ensure PyTorch nightly with MPS support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Search returns wrong location (low confidence)
1. Enable **Ultra Mode** for difficult images
2. Expand search radius — your initial center estimate may be off
3. Verify the area is indexed (index the suspected region first)
4. Try the AI Coarse mode to get a better center estimate (requires `GEMINI_API_KEY`)

### Indexing is very slow
- Use `build_index.py` (standalone, no GUI overhead) for large areas
- Indexing is I/O and API-bound — a faster internet connection helps significantly
- Do not change grid resolution (300) unless you understand the implications

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `test_super.py` | Everything: GUI, indexing pipeline, search pipeline, result display |
| `cosplace_utils.py` | `load_cosplace_model()`, `get_descriptor()` |
| `build_index.py` | CLI tool for headless large-area indexing |
| `index/cosplace_descriptors.npy` | Compiled descriptor matrix — load with `np.load()` |
| `index/metadata.npz` | Per-panorama metadata: lat, lon, heading, panoid |
| `cosplace_parts/*.npz` | Raw chunks written during indexing; compiled into `index/` |
```
