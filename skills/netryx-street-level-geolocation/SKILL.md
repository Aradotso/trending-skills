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
  - reverse geolocate image
  - find where this photo was taken
  - osint geolocation from photo
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted, open-source geolocation engine that identifies precise GPS coordinates from any street-level photograph. It crawls and indexes street-view panoramas, extracts visual fingerprints using CosPlace, then verifies matches using ALIKED/DISK keypoint extraction and LightGlue feature matching — all running on your own hardware with no external API dependencies for search.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (must install from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### Platform GPU Notes

| Platform | Backend | Feature Extractor |
|---|---|---|
| NVIDIA GPU | CUDA | ALIKED (1024 keypoints) |
| Apple Silicon (M1+) | MPS | DISK (768 keypoints) |
| CPU only | CPU | DISK (slower) |

### Optional: Gemini API Key (AI Coarse Mode)

```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix:** `brew install python-tk@3.11` (match your Python version)

---

## Project Structure

```
netryx/
├── test_super.py          # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Raw .npz embedding chunks (created during indexing)
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim CosPlace descriptors
    └── metadata.npz               # Coordinates, headings, panorama IDs
```

---

## Core Workflow

### Step 1 — Build an Index (required before any search)

In the GUI, select **Create** mode and configure:

| Setting | Description | Recommended |
|---|---|---|
| Center lat/lon | Center of area to index | Target city center |
| Radius | Coverage radius | 0.5–1km to start |
| Grid resolution | Sampling density | 300 (don't change) |

Click **Create Index**. The index saves incrementally — safe to interrupt and resume.

**Indexing time estimates:**

| Radius | Panoramas | Time (M2 Max) | Index Size |
|---|---|---|---|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hr | ~250 MB |
| 5 km | ~30,000 | 8–12 hr | ~3 GB |
| 10 km | ~100,000 | 24–48 hr | ~7 GB |

### Step 2 — Search

1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center lat/lon + radius
   - **AI Coarse**: Gemini analyzes visual clues to estimate region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Three-Stage Pipeline

### Stage 1: Global Retrieval (CosPlace)

- Extracts a 512-dim descriptor from query image (+ horizontally flipped version)
- Cosine similarity search against entire index
- Haversine radius filter applied
- Returns top 500–1000 candidates
- Runs in **< 1 second** (single matrix multiply)

### Stage 2: Local Geometric Verification (ALIKED/DISK + LightGlue)

For each candidate:
- Downloads 8 Google Street View tiles → stitches panorama
- Crops at indexed heading, 3 FOVs (70°, 90°, 110°)
- ALIKED or DISK extracts local keypoints
- LightGlue matches keypoints to query
- RANSAC filters geometrically consistent matches (inliers)
- Processes 300–500 candidates in **2–5 minutes**

### Stage 3: Refinement

- Heading refinement: tests ±45° at 15° steps across 3 FOVs on top 15 candidates
- Spatial consensus: clusters matches into 50m cells; clusters beat lone outliers
- Confidence scoring: evaluates geographic clustering + uniqueness ratio

---

## Ultra Mode

Enable the **Ultra Mode** checkbox in the GUI for:

- **LoFTR**: Detector-free dense matching (handles blur, low texture, night images)
- **Descriptor hopping**: Re-searches index using the matched panorama's clean descriptor
- **Neighborhood expansion**: Searches all panoramas within 100m of best match

Use when standard pipeline gives low-confidence results or the image is degraded.

---

## Multi-Region Index Pattern

The index is unified — all cities/areas share a single index file. The radius filter at search time handles separation:

```python
# Index Paris
# Center: 48.8566, 2.3522 — radius 5km

# Index London  
# Center: 51.5074, -0.1278 — radius 5km

# Search only Paris results:
# center=48.8566,2.3522  radius=5km → only returns Paris panoramas

# Search only London results:
# center=51.5074,-0.1278 radius=5km → only returns London panoramas
```

No city selection needed — coordinate + radius fully isolates the search area.

---

## Code Examples

### Extract a CosPlace Descriptor Programmatically

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (auto-detects CUDA / MPS / CPU)
model, device = load_cosplace_model()

# Extract descriptor from an image file
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device)
# descriptor.shape → torch.Size([512])
```

### Build Index Programmatically (Large Areas)

```python
# Use the standalone high-performance builder for large datasets
# Run from project root:
import subprocess
subprocess.run([
    "python", "build_index.py",
    "--lat", "48.8566",
    "--lon", "2.3522",
    "--radius", "5000",       # meters
    "--output-dir", "./index"
])
```

### Load and Query the Index Directly

```python
import numpy as np
import torch
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

# Load pre-built index
descriptors = np.load("index/cosplace_descriptors.npy")   # (N, 512)
meta = np.load("index/metadata.npz", allow_pickle=True)
lats = meta["lats"]      # (N,)
lons = meta["lons"]      # (N,)
headings = meta["headings"]  # (N,)
panoids = meta["panoids"]    # (N,)

# Extract query descriptor
model, device = load_cosplace_model()
query_img = Image.open("query.jpg").convert("RGB")
query_desc = extract_descriptor(model, query_img, device)  # torch.Tensor [512]

# Cosine similarity search
desc_tensor = torch.from_numpy(descriptors).float()
query_norm = torch.nn.functional.normalize(query_desc.unsqueeze(0), dim=1)
db_norm = torch.nn.functional.normalize(desc_tensor, dim=1)
similarities = (query_norm @ db_norm.T).squeeze(0)  # (N,)

# Radius filter using haversine
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

center_lat, center_lon = 48.8566, 2.3522
radius_km = 5.0

mask = torch.tensor([
    haversine_km(center_lat, center_lon, float(lats[i]), float(lons[i])) <= radius_km
    for i in range(len(lats))
])

similarities[~mask] = -1.0  # exclude out-of-radius

# Top 500 candidates
top_k = torch.topk(similarities, k=min(500, mask.sum().item()))
top_indices = top_k.indices.tolist()
top_scores = top_k.values.tolist()

for idx, score in zip(top_indices[:5], top_scores[:5]):
    print(f"lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, "
          f"heading={headings[idx]}, panoid={panoids[idx]}, sim={score:.4f}")
```

### Flip-Augmented Retrieval (Catches Reversed Perspectives)

```python
import torchvision.transforms.functional as TF

query_img = Image.open("query.jpg").convert("RGB")
query_img_flipped = TF.hflip(query_img)

desc_orig = extract_descriptor(model, query_img, device)
desc_flip = extract_descriptor(model, query_img_flipped, device)

# Average both descriptors for combined similarity
desc_combined = (desc_orig + desc_flip) / 2
desc_combined = torch.nn.functional.normalize(desc_combined, dim=0)
```

### Verify a Candidate with LightGlue

```python
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)

# Load query and candidate panorama crop
query = load_image("query.jpg").to(device)
candidate = load_image("candidate_crop.jpg").to(device)

# Extract features
feats0 = extractor.extract(query)
feats1 = extractor.extract(candidate)

# Match
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]

# RANSAC verification
import cv2
import numpy as np

if len(kpts0) >= 8:
    pts0 = kpts0.cpu().numpy()
    pts1 = kpts1.cpu().numpy()
    _, inlier_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"Verified inliers: {inlier_count}")
```

---

## Common Patterns

### Pattern: Index a Small Test Area First

Always start with a 0.5km radius to validate your setup before indexing large areas:

```
Center: target location lat/lon
Radius: 0.5 km
Grid resolution: 300
```

### Pattern: Use Manual Mode Over AI Coarse

For OSINT/conflict work, Manual mode with known approximate coordinates is faster and more reliable than AI Coarse (which uses Gemini and may be imprecise):

```
Center lat/lon: your best estimate of where the photo is from
Radius: 2–5 km (wider = more candidates = slower but safer)
```

### Pattern: Ultra Mode for Degraded Images

Enable Ultra Mode when:
- Image is blurry, low-resolution, or heavily compressed
- Night photography or low-contrast scenes
- Standard pipeline confidence score is below 0.5
- Heavily cropped or zoomed-in images with few features

### Pattern: Expanding Search Radius Iteratively

```
Attempt 1: radius=1km   → no match
Attempt 2: radius=3km   → no match  
Attempt 3: radius=5km   → match found at confidence 0.82
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: lightglue` | LightGlue not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| GUI appears blank on macOS | Bundled tkinter bug | `brew install python-tk@3.11` |
| Low confidence scores (<0.4) | Index doesn't cover photo location, or wrong radius | Widen radius or re-index the correct area |
| Ultra Mode fails with LoFTR error | kornia not installed | `pip install kornia` |
| Indexing hangs or crashes | Street View API rate limit or network issue | Index saves incrementally — restart, it resumes |
| CUDA out of memory | VRAM too small for batch | Reduce `max_num_keypoints` in extractor init |
| MPS errors on Apple Silicon | PyTorch MPS backend bug | `export PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Wrong match in urban areas | Similar-looking streets (e.g. Haussmanian Paris) | Enable Ultra Mode + reduce radius to narrow area |
| `index/cosplace_descriptors.npy` not found | Index not compiled yet | Run search once with Create mode, or run `build_index.py` |

### Environment Variable Reference

```bash
export GEMINI_API_KEY="..."              # Required for AI Coarse mode only
export PYTORCH_ENABLE_MPS_FALLBACK=1    # Fix MPS backend issues on Mac
```

---

## Hardware Recommendations

| Use Case | Minimum | Recommended |
|---|---|---|
| Testing (0.5km area) | 8GB RAM, any GPU | M1 Mac or GTX 1080 |
| City district (5km) | 16GB RAM, 6GB VRAM | M2 Max or RTX 3080 |
| Full city (10km+) | 32GB RAM, 8GB VRAM | M3 Max or RTX 4090 |
| Index storage per city | 250MB (1km) | 7GB (10km) |

---

## Key Concepts Summary

- **Index first, search later** — you must crawl and index an area before searching it
- **One unified index** — all cities share `cosplace_descriptors.npy`; radius filter separates them at search time
- **Sub-50m accuracy** on standard pipeline; sub-20m possible with Ultra Mode
- **No landmarks required** — works on featureless residential streets
- **Source agnostic** — designed for Mapillary, KartaView, or any street-view provider
- **Confidence score** — values above 0.7 are high confidence; below 0.4 indicates uncertain match
```
