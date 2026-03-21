---
name: netryx-street-level-geolocation
description: Use Netryx, a local-first open-source street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue computer vision models.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - identify location from street view photo
  - run netryx geolocation
  - build a street view index
  - osint image geolocation
  - local geolocation without google lens
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls street-view panoramas, indexes them as 512-dim CosPlace fingerprints, and matches a query image through a three-stage pipeline: global retrieval → geometric verification → refinement. No cloud API required for searching — only for initial indexing (Street View panorama downloads).

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # required
pip install kornia                                        # optional: Ultra Mode (LoFTR)
```

### Optional: Gemini API key (AI Coarse location guessing)
```bash
export GEMINI_API_KEY="your_key_here"   # from aistudio.google.com
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

This opens the full GUI with Create (indexing) and Search modes.

---

## Project Structure

```
netryx/
├── test_super.py           # Main entry point — GUI + indexing + search
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/         # Raw .npz embedding chunks (written during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (searchable)
    └── metadata.npz               # lat/lon, headings, panorama IDs
```

---

## Core Workflow

### Step 1 — Create an Index

Index a geographic area before searching. The system crawls Street View panoramas in a grid, extracts CosPlace fingerprints, and saves them incrementally (resumable if interrupted).

**Via GUI:**
1. Select **Create** mode
2. Enter center `latitude, longitude`
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2,000    | 1–2 hours     | ~250 MB    |
| 5 km   | ~30,000   | 8–12 hours    | ~3 GB      |
| 10 km  | ~100,000  | 24–48 hours   | ~7 GB      |

**Via standalone builder (large datasets):**
```bash
python build_index.py
```

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: provide center lat/lon + radius (recommended)
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on a map

---

## Pipeline Internals

### Stage 1 — Global Retrieval (CosPlace)
```python
# cosplace_utils.py pattern
import torch
from cosplace_utils import get_cosplace_model, get_descriptor

model = get_cosplace_model()           # loads ResNet18/50 + GeM pooling
descriptor = get_descriptor(model, image_tensor)   # returns (512,) float32 tensor

# Index search = cosine similarity matrix multiplication
# flipped image also searched to catch reversed perspectives
# radius filter via haversine distance on metadata lat/lon
# returns top 500–1000 candidates
```

### Stage 2 — Geometric Verification (ALIKED/DISK + LightGlue)
```python
# Platform-adaptive feature extraction
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ALIKED used on CUDA (1024 keypoints)
# DISK used on MPS/CPU (768 keypoints)
# LightGlue matches keypoints between query and each candidate crop
# RANSAC filters to geometrically consistent inliers
# Candidate with most verified inliers = best match
```

### Stage 3 — Refinement
- **Heading refinement**: tests ±45° offsets at 15° steps across 3 FOVs (70°, 90°, 110°) for top 15 candidates
- **Spatial consensus**: clusters matches into 50m cells; prefers cluster over lone high-inlier outlier
- **Confidence scoring**: geographic clustering + uniqueness ratio vs. runner-up

---

## Ultra Mode

Enable for difficult images (night, motion blur, low texture). Adds:

- **LoFTR**: detector-free dense matching (requires `kornia`)
- **Descriptor hopping**: re-searches index using CosPlace descriptor from the matched panorama (clean image) instead of degraded query
- **Neighborhood expansion**: searches all panoramas within 100m of best match

Enable via the **Ultra Mode** checkbox in the GUI before running search.

---

## Multi-City Indexing

All cities share one unified index. Radius filter handles isolation automatically:

```
# Index Paris → Tel Aviv → London — all into the same index/
# Search Paris:  center=(48.8566, 2.3522), radius=5  → only Paris results
# Search London: center=(51.5074, -0.1278), radius=10 → only London results
```

No per-city configuration needed — coordinates + radius is the only selector.

---

## Hardware Requirements & GPU Selection

| Platform | Feature Extractor | Notes |
|----------|------------------|-------|
| NVIDIA CUDA (4GB+ VRAM) | ALIKED (1024 kp) | Fastest, recommended |
| Apple Silicon MPS | DISK (768 kp) | Good performance on M1–M4 |
| CPU | DISK | Works, significantly slower |

```python
# Netryx auto-selects device — no manual config needed
# Verify your device will be used:
import torch
print(torch.cuda.is_available())        # NVIDIA
print(torch.backends.mps.is_available()) # Apple Silicon
```

---

## Common Patterns

### Check index health
```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Index size: {descriptors.shape}")          # (N, 512)
print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
print(f"Panoramas: {len(np.unique(meta['panoids']))}")
```

### Manual cosine similarity search (scripting without GUI)
```python
import numpy as np
import torch
from cosplace_utils import get_cosplace_model, get_descriptor
from PIL import Image
from torchvision import transforms

# Load index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)

# Load model and extract query descriptor
model = get_cosplace_model()
model.eval()

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("query.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    query_desc = get_descriptor(model, tensor).numpy()   # (512,)

# Cosine similarity
desc_norm = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
query_norm = query_desc / np.linalg.norm(query_desc)
scores = desc_norm @ query_norm                          # (N,)

# Radius filter (haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

center_lat, center_lon = 48.8566, 2.3522
radius_m = 2000
lats, lons = meta["lats"], meta["lons"]
dists = haversine(center_lat, center_lon, lats, lons)
in_radius = dists < radius_m

# Top candidates within radius
scores[~in_radius] = -1
top_idx = np.argsort(scores)[::-1][:500]

print("Top match:")
print(f"  lat={lats[top_idx[0]]:.6f}, lon={lons[top_idx[0]]:.6f}")
print(f"  score={scores[top_idx[0]]:.4f}")
print(f"  panoid={meta['panoids'][top_idx[0]]}")
```

### Rebuild searchable index from parts
```python
# If index/ is stale but cosplace_parts/ has new data, rebuild:
import subprocess
subprocess.run(["python", "build_index.py"])

# Or trigger from GUI: it auto-builds on first search if index is missing
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank | macOS tkinter bug | `brew install python-tk@3.11` |
| `No module named 'lightglue'` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| `No module named 'kornia'` | Ultra Mode dependency | `pip install kornia` |
| Search returns wrong city | Radius too large + no filter | Set tighter radius, use Manual mode |
| Low inliers (<30) on valid image | Heading mismatch | Enable Ultra Mode for heading refinement |
| Indexing stalls | Network timeout | Re-run — indexing is resumable from last saved chunk |
| MPS out of memory | Too many candidates | Reduce top-K candidates or use CPU fallback |
| `GEMINI_API_KEY` not found | Env var not set | `export GEMINI_API_KEY=your_key` or use Manual mode instead |
| Index search returns 0 results | Radius too small / wrong coords | Verify center coords overlap your indexed area |

### Verify GPU is active
```python
import torch
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Netryx will use: {device}")
# cpu is valid but expect 5–10x slower feature extraction
```

### Check cosplace_parts integrity
```python
import numpy as np
import os

parts_dir = "cosplace_parts"
total = 0
for f in sorted(os.listdir(parts_dir)):
    if f.endswith(".npz"):
        data = np.load(os.path.join(parts_dir, f))
        n = data["descriptors"].shape[0]
        total += n
        print(f"{f}: {n} descriptors")
print(f"Total: {total} panorama views indexed")
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Model inference backbone |
| `lightglue` (GitHub) | Deep feature matching |
| `kornia` (optional) | LoFTR dense matching (Ultra Mode) |
| `numpy` | Index storage and similarity search |
| `Pillow` | Image loading and crop generation |
| `tkinter` | GUI (stdlib, may need brew upgrade on macOS) |
