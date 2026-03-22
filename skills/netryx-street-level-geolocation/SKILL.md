```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - use netryx to locate
  - index street view panoramas
  - visual place recognition locally
  - find where a photo was taken
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the precise GPS coordinates of any street-level photograph. It crawls street-view panoramas, builds a searchable index of visual fingerprints, and matches query images using a three-stage computer vision pipeline: global retrieval (CosPlace) → local feature matching (ALIKED/DISK + LightGlue) → spatial refinement.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git

# Optional: Ultra Mode dense matching
pip install kornia
```

### macOS tkinter fix (if GUI appears blank)
```bash
brew install python-tk@3.11  # Match your Python version
```

### Optional: Gemini AI Coarse mode
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI application — entry point
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks (auto-created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim visual fingerprints
    └── metadata.npz               # Lat/lon, headings, panorama IDs
```

---

## Launch the GUI

```bash
python test_super.py
```

The GUI is the primary interface. It handles both indexing and searching.

---

## Core Workflow

### Step 1: Build an Index

An index must exist before searching. The index crawls street-view panoramas in a geographic area and extracts CosPlace descriptors.

**In the GUI:**
1. Select **Create** mode
2. Enter center coordinates (lat, lon)
3. Set radius in km (start with 0.5–1 km for testing)
4. Set grid resolution (default: 300 — do not change)
5. Click **Create Index**

**Index size estimates:**

| Radius | ~Panoramas | Build Time (M2 Max) | Index Size |
|--------|-----------|---------------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **resumable** — interrupting and restarting picks up where it left off.

**For large areas, use the standalone builder directly:**
```bash
python build_index.py
```

### Step 2: Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose method:
   - **Manual**: Provide approximate center lat/lon + radius
   - **AI Coarse**: Gemini analyzes visual clues (signs, architecture) to auto-guess the region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on a map

---

## Pipeline Architecture

```
Query Image
    │
    ├─ CosPlace → 512-dim descriptor
    ├─ Flipped CosPlace → 512-dim descriptor (handles reversed perspectives)
    │
    ▼
Index Search (cosine similarity, haversine radius filter)
    │
    └─ Top 500–1000 candidates
    │
    ▼
For each candidate:
    ├─ Download panorama (8 tiles, stitched)
    ├─ Crop at indexed heading
    ├─ Generate multi-FOV crops (70°, 90°, 110°)
    ├─ ALIKED (CUDA) / DISK (MPS/CPU) → keypoints + descriptors
    ├─ LightGlue → feature matches
    └─ RANSAC → geometric inliers
    │
    ▼
Heading Refinement (top 15 candidates, ±45° at 15° steps, 3 FOVs)
    │
    ├─ Spatial consensus clustering (50m cells)
    └─ Confidence scoring
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Platform-Specific Behavior

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|--------------|---------------------|-----|
| Feature extractor | ALIKED (1024 kp) | DISK (768 kp) | DISK |
| Typical speed | Fastest | Fast | Slow |
| Ultra Mode (LoFTR) | ✅ | ✅ | ✅ (very slow) |

---

## Ultra Mode

Enable the **Ultra Mode** checkbox in the GUI for difficult images (night shots, motion blur, low texture).

Ultra Mode adds:
- **LoFTR** — detector-free dense matching (handles blur/low-contrast)
- **Descriptor hopping** — re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion** — searches all panoramas within 100m of the best match

Ultra Mode is significantly slower but catches matches the standard pipeline misses.

---

## Multi-Area Indexing

All areas are stored in a single unified index. The radius filter isolates regions at search time — no city selection needed.

```
# Index Paris (5km radius around Eiffel Tower)
center_lat=48.8584, center_lon=2.2945, radius=5km → index/

# Index London (10km radius around Trafalgar Square)
center_lat=51.5080, center_lon=-0.1281, radius=10km → same index/

# Search Paris only:
search(center_lat=48.8584, center_lon=2.2945, radius=5km)

# Search London only:
search(center_lat=51.5080, center_lon=-0.1281, radius=10km)
```

---

## Working with CosPlace Descriptors Directly

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA/MPS/CPU)
model = load_cosplace_model()

# Extract descriptor from a query image
img = Image.open("query_photo.jpg")
descriptor = get_descriptor(model, img)  # Returns torch.Tensor, shape (512,)

# Extract descriptor from flipped image (improves recall for reversed views)
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = get_descriptor(model, img_flipped)
```

---

## Loading and Querying the Index Directly

```python
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load compiled index
descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512) float32
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]       # (N,) float64
lons = meta["lons"]       # (N,) float64
headings = meta["headings"]  # (N,) float32
panoids = meta["panoids"]    # (N,) str

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km, top_k=500):
    """
    query_descriptor: np.ndarray shape (512,), L2-normalized
    Returns indices of top_k candidates within radius, sorted by cosine similarity.
    """
    # Radius filter
    distances = np.array([
        haversine_km(center_lat, center_lon, lats[i], lons[i])
        for i in range(len(lats))
    ])
    in_radius = np.where(distances <= radius_km)[0]

    if len(in_radius) == 0:
        return []

    # Cosine similarity (descriptors should be L2-normalized)
    subset = descriptors[in_radius]  # (M, 512)
    q = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    sims = subset @ q  # (M,)

    # Top-k
    top_local = np.argsort(sims)[::-1][:top_k]
    return in_radius[top_local]

# Example usage
import torch
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image

model = load_cosplace_model()
img = Image.open("mystery_photo.jpg")
desc = get_descriptor(model, img).cpu().numpy()

candidates = search_index(desc, center_lat=48.8584, center_lon=2.2945, radius_km=2.0)
print(f"Found {len(candidates)} candidates")
for idx in candidates[:5]:
    print(f"  panoid={panoids[idx]}, lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, heading={headings[idx]:.1f}°")
```

---

## Raw Index Chunks (cosplace_parts/)

During indexing, data is saved incrementally as `.npz` chunks:

```python
import numpy as np
import os

# Inspect a chunk
chunk = np.load("cosplace_parts/chunk_0000.npz", allow_pickle=True)
print(chunk.files)          # ['descriptors', 'lats', 'lons', 'headings', 'panoids']
print(chunk['descriptors'].shape)  # (batch_size, 512)

# Manually rebuild the compiled index from parts
parts_dir = "cosplace_parts"
all_descs, all_lats, all_lons, all_headings, all_panoids = [], [], [], [], []

for fname in sorted(os.listdir(parts_dir)):
    if fname.endswith(".npz"):
        part = np.load(os.path.join(parts_dir, fname), allow_pickle=True)
        all_descs.append(part['descriptors'])
        all_lats.append(part['lats'])
        all_lons.append(part['lons'])
        all_headings.append(part['headings'])
        all_panoids.append(part['panoids'])

descriptors = np.vstack(all_descs).astype(np.float32)
os.makedirs("index", exist_ok=True)
np.save("index/cosplace_descriptors.npy", descriptors)
np.savez("index/metadata.npz",
         lats=np.concatenate(all_lats),
         lons=np.concatenate(all_lons),
         headings=np.concatenate(all_headings),
         panoids=np.concatenate(all_panoids))
print(f"Index built: {len(descriptors)} panoramas")
```

---

## Common Patterns

### Pattern 1: Batch geolocation of multiple images

```python
from cosplace_utils import load_cosplace_model, get_descriptor
from PIL import Image
import numpy as np

model = load_cosplace_model()
descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
center_lat, center_lon, radius_km = 48.8584, 2.2945, 3.0

for path in image_paths:
    img = Image.open(path)
    desc = get_descriptor(model, img).cpu().numpy()
    candidates = search_index(desc, center_lat, center_lon, radius_km, top_k=100)
    best = candidates[0] if len(candidates) > 0 else None
    if best is not None:
        print(f"{path}: lat={meta['lats'][best]:.6f}, lon={meta['lons'][best]:.6f}")
    else:
        print(f"{path}: No match found in area")
```

### Pattern 2: Check what's in your index

```python
import numpy as np

meta = np.load("index/metadata.npz", allow_pickle=True)
lats, lons = meta["lats"], meta["lons"]

print(f"Total panoramas indexed: {len(lats)}")
print(f"Lat range: {lats.min():.4f} → {lats.max():.4f}")
print(f"Lon range: {lons.min():.4f} → {lons.max():.4f}")
print(f"Approx center: {lats.mean():.4f}, {lons.mean():.4f}")
```

### Pattern 3: Verify LightGlue is available

```python
try:
    from lightglue import LightGlue, ALIKED, DISK
    from lightglue.utils import load_image, rbd
    print("LightGlue installed correctly")
except ImportError:
    print("Run: pip install git+https://github.com/cvg/LightGlue.git")
```

### Pattern 4: Verify device detection

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU (slow)")
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11  # Replace 3.11 with your Python version
```

### LightGlue import fails
```bash
pip install git+https://github.com/cvg/LightGlue.git
# Do NOT install via pip install lightglue — that is a different package
```

### LoFTR / Ultra Mode fails
```bash
pip install kornia
# kornia ships LoFTR — required for Ultra Mode
```

### Index search returns 0 candidates
- Check that your search center coordinates and radius actually overlap with indexed area
- Run the "check what's in your index" snippet above to verify coverage
- Ensure `index/cosplace_descriptors.npy` and `index/metadata.npz` both exist (run indexing to completion or manually rebuild from `cosplace_parts/`)

### Indexing interrupted / partial index
Indexing resumes automatically. Just re-run Create Index with the same parameters. Completed chunks in `cosplace_parts/` are skipped.

### Out of VRAM during search
- Reduce `top_k` candidates (default 500 → try 200)
- Disable Ultra Mode
- On CUDA: set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

### Low match confidence / wrong location
- Increase search radius — the correct area may be underindexed
- Enable Ultra Mode for degraded images
- Try AI Coarse mode if you have `GEMINI_API_KEY` set and don't know the region
- Ensure the query photo is street-level (not aerial, satellite, or indoor)

### Gemini AI Coarse mode not working
```bash
export GEMINI_API_KEY="your_key_from_aistudio_google_com"
# Get free key at: https://aistudio.google.com
```

---

## Key Models Reference

| Model | Role | Loaded on |
|-------|------|-----------|
| CosPlace (512-dim) | Global visual fingerprint | CUDA / MPS / CPU |
| ALIKED | Local keypoints (CUDA only) | CUDA |
| DISK | Local keypoints (fallback) | MPS / CPU |
| LightGlue | Deep feature matching | CUDA / MPS / CPU |
| LoFTR (kornia) | Dense matching, Ultra Mode | CUDA / MPS / CPU |

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `GEMINI_API_KEY` | AI Coarse geolocation mode | No |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memory tuning | No |
```
