```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - netryx geolocation
  - identify location from street view photo
  - reverse geolocate image
  - build street view index
  - osint geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It indexes street-view panoramas, extracts visual fingerprints, and uses a three-stage CV pipeline (global retrieval → geometric verification → refinement) entirely on your hardware — no cloud APIs required for inference.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must be installed from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching
pip install kornia
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11   # match your Python version
```

### Gemini API key (optional — AI Coarse mode only)
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Quick Start

### Launch the GUI
```bash
python test_super.py
```

The GUI has two modes: **Create** (index an area) and **Search** (geolocate a photo).

---

## Core Workflow

### Step 1 — Index an Area (Create Mode)

Build a searchable index of street-view panoramas for a geographic region.

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Center lat/lon | target area | e.g. `48.8566, 2.3522` for Paris center |
| Radius | 1–5 km | Start small (0.5 km) for testing |
| Grid resolution | 300 | Don't change this default |

**Indexing time vs. coverage:**

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Indexing is **resumable** — if interrupted, restarting picks up where it left off from `cosplace_parts/`.

### Step 2 — Search (Search Mode)

1. Upload a street-level photo
2. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
3. Click **Run Search** → **Start Full Search**
4. Result: GPS coordinates + confidence score on map

---

## Project Structure

```
netryx/
├── test_super.py           # Main entry point: GUI + indexing + search pipeline
├── cosplace_utils.py       # CosPlace model loader and descriptor extractor
├── build_index.py          # Standalone high-performance index builder (large areas)
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## Pipeline Architecture

```
Query Image
    │
    ├─ CosPlace 512-dim descriptor extraction
    ├─ Flipped image descriptor (catches reversed perspectives)
    │
    ▼
Index cosine similarity search (radius-filtered via haversine)
    │  → Top 500–1000 candidates
    │
    ▼
Per candidate:
    ├─ Download Street View panorama (8 tiles, stitched)
    ├─ Rectilinear crop at indexed heading
    ├─ Multi-FOV crops: 70°, 90°, 110°
    ├─ ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    ├─ LightGlue deep feature matching
    └─ RANSAC geometric verification → inlier count
    │
    ▼
Heading Refinement (top 15 candidates, ±45° at 15° steps, 3 FOVs)
    │
    ├─ Spatial consensus clustering (50m cells)
    ├─ Confidence scoring (clustering + uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode

Enable the **Ultra Mode** checkbox for difficult images (night, blur, low texture, heavy compression).

Ultra Mode adds three enhancements:

1. **LoFTR dense matching** — detector-free, handles blur/low-contrast where keypoint detectors fail
2. **Descriptor hopping** — if best match has <50 inliers, extracts CosPlace from the *matched clean panorama* and re-searches; recovers from degraded query descriptors
3. **Neighborhood expansion** — searches all panoramas within 100m of best match; correct location often one node away

Ultra Mode is significantly slower; use for hard cases only.

---

## Platform Behavior

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|--------------|---------------------|-----|
| Feature extractor | ALIKED (1024 kp) | DISK (768 kp) | DISK |
| Speed | Fastest | Fast | Slow |
| Min VRAM | 4 GB | 4 GB unified | N/A |

---

## Programmatic Usage Examples

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape → (512,)
```

### Search an existing index programmatically

```python
import numpy as np

# Load the compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
headings = meta["headings"]  # (N,)
panoids = meta["panoids"]    # (N,)

def haversine_filter(query_lat, query_lon, lats, lons, radius_km):
    """Return boolean mask of entries within radius_km of query point."""
    R = 6371.0
    dlat = np.radians(lats - query_lat)
    dlon = np.radians(lons - query_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(query_lat))
         * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a)) <= radius_km

def cosine_search(query_descriptor, descriptors, mask, top_k=500):
    """Return top_k indices sorted by cosine similarity within mask."""
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db = descriptors[mask]
    db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-8)
    sims = db_norm @ q
    masked_indices = np.where(mask)[0]
    top_local = np.argsort(sims)[::-1][:top_k]
    return masked_indices[top_local], sims[top_local]

# Example: search Paris area
query_lat, query_lon = 48.8566, 2.3522
radius_km = 2.0

mask = haversine_filter(query_lat, query_lon, lats, lons, radius_km)
top_indices, scores = cosine_search(descriptor, descriptors, mask, top_k=500)

for i, (idx, score) in enumerate(zip(top_indices[:5], scores[:5])):
    print(f"#{i+1}: panoid={panoids[idx]}, lat={lats[idx]:.6f}, "
          f"lon={lons[idx]:.6f}, heading={headings[idx]}, score={score:.4f}")
```

### Build index for a large area (CLI alternative)

```bash
# build_index.py is the high-performance standalone indexer
python build_index.py \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 5.0 \
    --resolution 300
```

### Using both normal and flipped descriptors (improves recall)

```python
import PIL.Image as Image
from cosplace_utils import load_cosplace_model, extract_descriptor
import torch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
model = load_cosplace_model(device=device)

img = Image.open("query.jpg").convert("RGB")
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

desc_normal  = extract_descriptor(model, img, device=device)
desc_flipped = extract_descriptor(model, img_flipped, device=device)

# Use max similarity across both descriptors
sims_normal  = descriptors_norm @ (desc_normal  / np.linalg.norm(desc_normal))
sims_flipped = descriptors_norm @ (desc_flipped / np.linalg.norm(desc_flipped))
sims_combined = np.maximum(sims_normal, sims_flipped)
```

### LightGlue matching snippet

```python
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

device = torch.device("cuda")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="aliked").eval().to(device)

image0 = load_image("query.jpg").to(device)
image1 = load_image("candidate_crop.jpg").to(device)

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]
print(f"Matched keypoints: {len(kpts0)}")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_inliers(kpts0, kpts1, threshold=3.0):
    """
    Returns number of RANSAC inliers for a set of matched keypoints.
    kpts0, kpts1: numpy arrays of shape (N, 2)
    """
    if len(kpts0) < 4:
        return 0
    _, mask = cv2.findFundamentalMat(
        kpts0.astype(np.float32),
        kpts1.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999
    )
    if mask is None:
        return 0
    return int(mask.sum())

inliers = ransac_inliers(kpts0.cpu().numpy(), kpts1.cpu().numpy())
print(f"Inliers after RANSAC: {inliers}")
```

---

## Index Management

The index is **unified and location-agnostic**. You can index multiple cities into the same index:

```
# Index Paris
python test_super.py  →  Create, 48.8566, 2.3522, radius=5km

# Index London (same index)
python test_super.py  →  Create, 51.5074, -0.1278, radius=5km

# Search is scoped by coordinates + radius — no city selection needed
# Searching Paris center with radius=5km returns only Paris results
```

**Index files:**
- `cosplace_parts/*.npz` — raw chunks, written incrementally during crawl
- `index/cosplace_descriptors.npy` — compiled descriptor matrix
- `index/metadata.npz` — lat, lon, heading, panoid arrays

Auto-build compiles parts → index automatically before search.

---

## Common Patterns

### Pattern: Test with a small index first
```
radius=0.5km, resolution=300 → ~500 panoramas, ~30 min
Good for verifying pipeline works before committing to large crawl.
```

### Pattern: Confidence score interpretation
- **High confidence**: strong spatial clustering + high uniqueness ratio (best >> runner-up)
- **Low confidence**: try Ultra Mode, or expand search radius if approximate location is uncertain

### Pattern: AI Coarse → Manual refinement
```
1. Use AI Coarse to identify country/city from photo clues (signs, architecture, vegetation)
2. Switch to Manual with the identified city center + 5km radius
3. Run full search — much faster than global search
```

### Pattern: Night/blurry images
```
1. Enable Ultra Mode checkbox
2. Use Manual mode with a known approximate location
3. Ultra Mode LoFTR handles blur; descriptor hopping recovers from degraded query
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank | macOS bundled tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | LightGlue not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `ModuleNotFoundError: kornia` | Ultra Mode dependency missing | `pip install kornia` |
| CUDA OOM | VRAM < 4GB | Reduce `max_num_keypoints` or use CPU |
| Indexing stalls at same point | Street View API rate limit | Wait and restart; index resumes automatically |
| Low inlier counts (<20) | FOV mismatch, wrong perspective | Enable Ultra Mode; try heading refinement manually |
| AI Coarse fails | Missing or invalid Gemini key | `export GEMINI_API_KEY="..."` or use Manual mode |
| Index search returns 0 results | Radius too small or wrong coordinates | Increase radius; verify center lat/lon |
| `cosplace_descriptors.npy` not found | Index not compiled yet | Run auto-build or re-index the area |

### Verify GPU is being used

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS: ", torch.backends.mps.is_available())
# Expected: True for your platform
```

### Check index integrity

```python
import numpy as np

desc = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Descriptors: {desc.shape}")        # (N, 512)
print(f"Lats range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
print(f"Total indexed panoramas: {len(meta['lats'])}")
assert desc.shape[0] == len(meta['lats']), "Index mismatch — rebuild index"
```

---

## Models Reference

| Model | Role | Hardware |
|-------|------|----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim visual fingerprint | All |
| [ALIKED](https://github.com/naver/alike) | Local keypoints + descriptors | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints + descriptors | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | All |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Dense detector-free matching (Ultra) | All (slow on CPU) |

---

## Hardware Recommendations

- **Best**: Apple Silicon M1+ (MPS) or NVIDIA GPU ≥8GB VRAM
- **Minimum**: 4GB GPU VRAM, 8GB RAM, 10GB storage
- **Production indexing**: 16GB+ RAM, 50GB+ storage, broadband internet
- **CPU-only**: Functional but stage 2 will take 10–30× longer
```
