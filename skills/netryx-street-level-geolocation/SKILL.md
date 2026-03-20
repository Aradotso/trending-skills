```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view area
  - run netryx search
  - geolocation from photo locally
  - osint geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the precise GPS coordinates of any street-level photograph. It crawls street-view panoramas, indexes them as visual fingerprints, and matches query images using a three-stage computer vision pipeline (global retrieval → geometric verification → refinement). Sub-50m accuracy. Runs entirely on local hardware.

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

### Platform Notes
- **NVIDIA GPU (CUDA)**: Uses ALIKED (1024 keypoints) — fastest, highest accuracy
- **Mac Apple Silicon (MPS)**: Uses DISK (768 keypoints) — excellent on M1/M2/M3/M4
- **CPU only**: Works but 5–10× slower than GPU

### Optional: Gemini API for AI Coarse Mode
```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11   # match your Python version
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. All workflows (indexing, searching, results) run through it.

---

## Core Workflow

### Step 1: Create an Index for a Geographic Area

Before any search, you must index the target area. This crawls street-view panoramas and stores CosPlace embeddings.

**GUI steps:**
1. Select **Create** mode
2. Enter center `latitude, longitude`
3. Set `radius` in km (start with `0.5`–`1` for testing)
4. Set grid resolution (default `300` — do not change)
5. Click **Create Index**

**Index size estimates:**

| Radius | ~Panoramas | Time (M2 Max) | Disk |
|--------|-----------|---------------|------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

Indexing is **incremental** — interrupting and restarting resumes from where it left off.

**All cities share one index.** Index Paris, then London, then Tokyo — all go into the same index files. The radius filter at search time isolates results to the correct city automatically.

---

### Step 2: Search

**GUI steps:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coordinates + radius (recommended)
   - **AI Coarse**: Let Gemini infer the region from visual clues (no coordinates needed)
4. Click **Run Search** → **Start Full Search**
5. Result shown on map with confidence score and GPS coordinates

**Ultra Mode** (checkbox): Enable for difficult images (night, blur, low texture). Adds LoFTR dense matching, descriptor hopping, and 100m neighborhood expansion. Slower but more robust.

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone index builder (large datasets / headless)
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
│   └── *.npz
└── index/                 # Compiled searchable index
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace 512-dim descriptor + flipped variant
    ▼
Index Search (cosine similarity, haversine radius filter)
    │
    ├── Top 500 candidates
    ▼
Download Panoramas → 3 FOV crops (70°, 90°, 110°)
    │
    ├── ALIKED (CUDA) / DISK (MPS/CPU) keypoint extraction
    ├── LightGlue deep feature matching
    ├── RANSAC geometric verification (inlier counting)
    ▼
Heading Refinement (±45° in 15° steps, top 15 candidates)
    │
    ├── Spatial consensus clustering (50m cells)
    ├── Confidence scoring (clustering + uniqueness ratio)
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (cached after first call)
model, transform = get_cosplace_model()

# Load your query image
image = Image.open("query.jpg").convert("RGB")

# Extract 512-dim descriptor
descriptor = get_descriptor(image, model, transform)
print(descriptor.shape)  # torch.Size([512])
```

### Search the Index Programmatically

```python
import numpy as np
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import math

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def search_index(query_image_path, center_lat, center_lon, radius_km, top_k=500):
    # Load index
    descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512) float32
    meta = np.load("index/metadata.npz", allow_pickle=True)
    lats = meta["lats"]
    lons = meta["lons"]
    headings = meta["headings"]
    panoids = meta["panoids"]

    # Extract query descriptor
    model, transform = get_cosplace_model()
    image = Image.open(query_image_path).convert("RGB")
    q_desc = get_descriptor(image, model, transform).numpy()

    # Also try flipped version
    image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    q_desc_flip = get_descriptor(image_flip, model, transform).numpy()

    # Cosine similarity (descriptors are L2-normalized)
    sims = descriptors @ q_desc
    sims_flip = descriptors @ q_desc_flip
    combined_sims = np.maximum(sims, sims_flip)

    # Radius filter
    in_radius = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i]) <= radius_km
        for i in range(len(lats))
    ])

    combined_sims[~in_radius] = -1  # exclude out-of-radius

    # Top-k candidates
    top_indices = np.argsort(combined_sims)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        if combined_sims[idx] < 0:
            break
        candidates.append({
            "panoid": panoids[idx],
            "lat": lats[idx],
            "lon": lons[idx],
            "heading": headings[idx],
            "similarity": float(combined_sims[idx]),
        })

    return candidates


# Usage
candidates = search_index(
    query_image_path="my_photo.jpg",
    center_lat=48.8566,
    center_lon=2.3522,
    radius_km=2.0,
    top_k=500
)
print(f"Found {len(candidates)} candidates")
print(f"Top match: {candidates[0]}")
```

### Build Index Headlessly (Large Areas)

For large-scale indexing without the GUI, use `build_index.py`:

```bash
python build_index.py \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 5.0 \
    --grid 300
```

This saves chunks to `cosplace_parts/` and auto-compiles the final `index/` on completion.

### Check Index Stats

```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Total panoramas indexed: {len(descriptors):,}")
print(f"Descriptor shape: {descriptors.shape}")          # (N, 512)
print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
```

### Verify GPU / Device Detection

```python
import torch

if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS (Apple Silicon) available")
else:
    print("CPU only — expect slower performance")
```

---

## Configuration Reference

All configuration is done through the GUI or by modifying constants in `test_super.py`. Key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | `300` | Panorama density. Do not change. |
| Top candidates (Stage 1) | `500–1000` | More = slower but more thorough |
| Heading refinement range | `±45°` at `15°` steps | Applied to top 15 candidates |
| FOV crops | `70°, 90°, 110°` | Multi-scale matching |
| Spatial clustering cell | `50m` | Consensus grouping size |
| Ultra Mode neighborhood | `100m` | Expansion radius after best match |
| LoFTR activation | Ultra Mode only | Requires `kornia` installed |

---

## Multi-City Index Strategy

```
Index once, search many:

  python test_super.py → Create → Paris (48.8566, 2.3522, 5km)
  python test_super.py → Create → London (51.5074, -0.1278, 5km)
  python test_super.py → Create → Tokyo  (35.6762, 139.6503, 5km)

  All stored in the SAME index/.

  Search Paris:  center=(48.8566, 2.3522), radius=5km  → only Paris results
  Search London: center=(51.5074, -0.1278), radius=5km → only London results
```

No per-city index files. No city selection dropdown. Coordinates + radius handle isolation automatically.

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # or python-tk@3.10 depending on your version
# Then relaunch:
python test_super.py
```

### LightGlue import error
```bash
# Must install from GitHub, not PyPI:
pip install git+https://github.com/cvg/LightGlue.git
```

### LoFTR / Ultra Mode not working
```bash
pip install kornia
# Verify:
python -c "import kornia; print(kornia.__version__)"
```

### CUDA out of memory
- Reduce batch size in `test_super.py` (lower `top_k` candidates)
- Use DISK instead of ALIKED (fewer keypoints): forced automatically on MPS, manually switchable in code
- Ensure no other GPU processes are running

### Index build interrupted
No action needed. Restart the same Create Index job with identical parameters — it resumes from the last saved chunk in `cosplace_parts/`.

### Low confidence / wrong result
1. Enable **Ultra Mode** (adds LoFTR + descriptor hopping + neighborhood expansion)
2. Increase search radius (the area may not be fully indexed)
3. Verify the area is indexed: check `index/metadata.npz` lat/lon ranges cover your search area
4. Try **AI Coarse** mode if you don't know the approximate location

### Slow indexing
- Indexing requires internet (fetches Street View panoramas) — use broadband
- Large radii (5–10km) are expected to take many hours; this is normal
- `build_index.py` is faster than the GUI for large areas (no UI overhead)

### Verify installation end-to-end
```python
# Quick smoke test
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
import numpy as np

model, transform = get_cosplace_model()
img = Image.new("RGB", (640, 480), color=(128, 128, 128))
desc = get_descriptor(img, model, transform).numpy()
assert desc.shape == (512,), f"Expected (512,), got {desc.shape}"
assert abs(np.linalg.norm(desc) - 1.0) < 0.01, "Descriptor not L2-normalized"
print("✓ CosPlace working correctly")
```

---

## Models Reference

| Model | Role | Activated |
|-------|------|-----------|
| [CosPlace](https://github.com/gmberton/cosplace) | Global 512-dim place fingerprint | Always |
| [ALIKED](https://github.com/naver/alike) | Local keypoints (1024) | CUDA only |
| [DISK](https://github.com/cvlab-epfl/disk) | Local keypoints (768) | MPS / CPU |
| [LightGlue](https://github.com/cvg/LightGlue) | Deep feature matching | Always |
| [LoFTR](https://github.com/zju3dv/LoFTR) | Detector-free dense matching | Ultra Mode only |

Models are downloaded automatically on first use and cached locally.
```
