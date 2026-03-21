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
  - index street view panoramas
  - reverse geolocation from photo
  - osint geolocation tool
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies exact GPS coordinates from any street-level photograph. It crawls street-view panoramas, indexes them using CosPlace visual fingerprints, then matches query images using ALIKED/DISK keypoint extraction and LightGlue deep feature matching — achieving sub-50m accuracy with no internet presence required for the target location.

---

## Installation

```bash
git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
cd Netryx-OpenSource-Next-Gen-Street-Level-Geolocation

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Required: LightGlue (install from GitHub)
pip install git+https://github.com/cvg/LightGlue.git

# Optional: LoFTR for Ultra Mode (difficult/blurry images)
pip install kornia
```

### macOS tkinter fix (blank GUI)
```bash
brew install python-tk@3.11  # Match your Python version
```

### Gemini API Key (optional — AI Coarse location mode)
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.10+ |
| GPU VRAM | 4GB | 8GB+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 50GB+ |

**GPU backends:**
- NVIDIA → CUDA (uses ALIKED, 1024 keypoints)
- Apple Silicon → MPS (uses DISK, 768 keypoints)
- CPU → works but slow

---

## Launch the GUI

```bash
python test_super.py
```

This opens the full application: index creation, search, real-time visualization, and map display.

---

## Core Workflow

### Step 1 — Create an Index (crawl and fingerprint an area)

In the GUI:
1. Select **Create** mode
2. Enter center latitude/longitude
3. Set radius (km) and grid resolution (default: 300)
4. Click **Create Index**

Indexing times:

| Radius | ~Panoramas | Time (M2 Max) | Index Size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Indexing is **incremental** — safe to interrupt and resume.

**Index output structure:**
```
cosplace_parts/          # Raw embedding chunks per crawl batch
index/
├── cosplace_descriptors.npy   # All 512-dim CosPlace fingerprints
└── metadata.npz               # Lat/lon, headings, panorama IDs
```

### Step 2 — Search (geolocate a photo)

In the GUI:
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Provide approximate center coords + radius
   - **AI Coarse**: Gemini analyzes visual clues (signs, vegetation, architecture) to suggest region
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Pipeline Architecture

```
Query Image
    │
    ├── CosPlace descriptor (512-dim)
    ├── Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search — cosine similarity, haversine radius filter
    │
    └── Top 500–1000 candidates
    │
    ▼
For each candidate:
    ├── Download Street View panorama (8 tiles, stitched)
    ├── Rectilinear crop at indexed heading
    ├── Multi-FOV crops: 70°, 90°, 110°
    ├── ALIKED (CUDA) or DISK (MPS/CPU) keypoint extraction
    └── LightGlue matching + RANSAC verification
    │
    ▼
Heading Refinement
    ├── ±45° offsets at 15° steps, top 15 candidates, 3 FOVs
    │
    ▼
Spatial Consensus (50m clustering)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Project Structure

```
netryx/
├── test_super.py          # Main app — GUI, indexing, search pipeline
├── cosplace_utils.py      # CosPlace model loading + descriptor extraction
├── build_index.py         # Standalone high-performance index builder
├── requirements.txt
├── cosplace_parts/        # Incremental embedding chunks
└── index/
    ├── cosplace_descriptors.npy
    └── metadata.npz
```

---

## Key Code Patterns

### Extract a CosPlace descriptor from an image

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

# Load model (auto-selects CUDA/MPS/CPU)
model, device = load_cosplace_model()

# Extract 512-dim fingerprint
img = Image.open("street_photo.jpg")
descriptor = extract_descriptor(model, img, device)  # shape: (512,)

# Also extract flipped version to catch reversed perspectives
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
descriptor_flipped = extract_descriptor(model, img_flipped, device)
```

### Search the index manually (radius-filtered cosine similarity)

```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def search_index(query_descriptor, center_lat, center_lon, radius_m, top_k=500):
    # Load index
    descriptors = np.load("index/cosplace_descriptors.npy")  # (N, 512)
    meta = np.load("index/metadata.npz", allow_pickle=True)
    lats = meta["lats"]
    lons = meta["lons"]
    headings = meta["headings"]
    panoids = meta["panoids"]

    # Radius filter
    distances = np.array([
        haversine(center_lat, center_lon, lat, lon)
        for lat, lon in zip(lats, lons)
    ])
    in_radius = distances <= radius_m

    # Cosine similarity search
    filtered_descs = descriptors[in_radius]
    q = query_descriptor / np.linalg.norm(query_descriptor)
    db = filtered_descs / np.linalg.norm(filtered_descs, axis=1, keepdims=True)
    scores = db @ q

    # Top-k candidates
    top_indices = np.argsort(scores)[::-1][:top_k]
    original_indices = np.where(in_radius)[0][top_indices]

    return [
        {
            "lat": lats[i],
            "lon": lons[i],
            "heading": headings[i],
            "panoid": panoids[i],
            "score": scores[top_indices[j]]
        }
        for j, i in enumerate(original_indices)
    ]

# Usage
results = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_m=2000)
print(f"Top match: {results[0]}")
```

### LightGlue feature matching between query and candidate

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

def match_images(query_path, candidate_path, device):
    # Choose extractor based on device
    if device.type == "cuda":
        extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
        matcher = LightGlue(features="aliked").eval().to(device)
    else:
        extractor = DISK(max_num_keypoints=768).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)

    # Load images
    image0 = load_image(query_path).to(device)
    image1 = load_image(candidate_path).to(device)

    # Extract features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # Match
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
    kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]
    num_matches = len(matches01["matches"])

    return num_matches, kpts0, kpts1

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
n_matches, pts0, pts1 = match_images("query.jpg", "candidate_crop.jpg", device)
print(f"Matched keypoints: {n_matches}")
```

### RANSAC geometric verification

```python
import cv2
import numpy as np

def ransac_verify(kpts_query, kpts_candidate, threshold=3.0):
    """Filter matches to geometrically consistent inliers."""
    if len(kpts_query) < 4:
        return 0, None

    pts0 = kpts_query.cpu().numpy()
    pts1 = kpts_candidate.cpu().numpy()

    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, threshold)
    if mask is None:
        return 0, None

    inliers = int(mask.sum())
    return inliers, mask

# Usage
inliers, mask = ransac_verify(pts0, pts1)
print(f"RANSAC inliers: {inliers}")
# Candidates with 50+ inliers are strong matches
```

### Multi-FOV crop generation

```python
from PIL import Image
import numpy as np

def get_rectilinear_crop(panorama: Image.Image, heading_deg: float, fov_deg: float,
                          output_width=640, output_height=480) -> Image.Image:
    """
    Extract a rectilinear crop from an equirectangular panorama.
    heading_deg: 0=North, 90=East, 180=South, 270=West
    fov_deg: field of view (try 70, 90, 110 for multi-FOV)
    """
    pano_w, pano_h = panorama.size
    f = (output_width / 2) / np.tan(np.radians(fov_deg / 2))

    heading_rad = np.radians(heading_deg)

    # Build pixel grid
    xs = np.linspace(-output_width/2, output_width/2, output_width)
    ys = np.linspace(-output_height/2, output_height/2, output_height)
    xv, yv = np.meshgrid(xs, ys)

    # Ray directions
    ray_x = np.sin(heading_rad) + xv/f * np.cos(heading_rad)
    ray_y = np.cos(heading_rad) - xv/f * np.sin(heading_rad)
    ray_z = -yv / f

    # Spherical coordinates
    lon = np.arctan2(ray_x, ray_y)
    lat = np.arctan2(ray_z, np.sqrt(ray_x**2 + ray_y**2))

    # Map to panorama pixel coords
    px = ((lon / (2 * np.pi)) + 0.5) * pano_w
    py = (0.5 - lat / np.pi) * pano_h

    # Sample panorama
    px = np.clip(px.astype(np.int32), 0, pano_w - 1)
    py = np.clip(py.astype(np.int32), 0, pano_h - 1)

    pano_arr = np.array(panorama)
    crop_arr = pano_arr[py, px]
    return Image.fromarray(crop_arr.astype(np.uint8))

# Usage: generate 3 FOV crops for a candidate
for fov in [70, 90, 110]:
    crop = get_rectilinear_crop(panorama, heading_deg=90.0, fov_deg=fov)
    crop.save(f"crop_fov{fov}.jpg")
```

### Build large index with standalone builder

```bash
# For large areas (5km+), use the high-performance standalone builder
python build_index.py \
  --lat 48.8566 \
  --lon 2.3522 \
  --radius 5.0 \
  --resolution 300
```

---

## Ultra Mode

Enable via GUI checkbox for difficult images (night, blur, low texture). Adds:

1. **LoFTR** — detector-free dense matching (handles blur/low contrast)
2. **Descriptor hopping** — re-searches index using fingerprint of matched panorama (clean image)
3. **Neighborhood expansion** — searches all panoramas within 100m of best match

```python
# Ultra Mode requires kornia
import kornia.feature as KF
import torch

def loftr_match(img0_tensor, img1_tensor, device):
    """Dense matching without keypoint detection."""
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    input_dict = {
        "image0": img0_tensor.unsqueeze(0).to(device),
        "image1": img1_tensor.unsqueeze(0).to(device),
    }

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Filter by confidence
    mask = confidence > 0.7
    return mkpts0[mask], mkpts1[mask], confidence[mask]
```

---

## Multi-City Indexing

The index is unified — all cities share one index. Radius filtering handles isolation at search time:

```python
# Index Paris (runs independently, adds to shared index)
# GUI: Create mode, lat=48.8566, lon=2.3522, radius=5km

# Index London (adds to same index)
# GUI: Create mode, lat=51.5074, lon=-0.1278, radius=5km

# Search Paris only (radius excludes London)
results = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_m=5000)

# Search London only
results = search_index(descriptor, center_lat=51.5074, center_lon=-0.1278, radius_m=5000)
```

---

## Confidence Scoring

Netryx outputs a confidence score based on:
- **Inlier count** from RANSAC (50+ = strong, 100+ = very strong)
- **Geographic clustering** — multiple top candidates near same location boosts confidence
- **Uniqueness ratio** — best match inliers vs. runner-up at different location

```python
def compute_confidence(top_matches, cluster_radius_m=50):
    """Estimate confidence from match clustering."""
    if not top_matches:
        return 0.0

    best = top_matches[0]
    best_inliers = best["inliers"]

    # Find clustered matches
    clustered = [
        m for m in top_matches
        if haversine(best["lat"], best["lon"], m["lat"], m["lon"]) < cluster_radius_m
    ]

    cluster_size = len(clustered)
    runner_up = next(
        (m for m in top_matches if
         haversine(best["lat"], best["lon"], m["lat"], m["lon"]) > cluster_radius_m),
        None
    )

    uniqueness = best_inliers / runner_up["inliers"] if runner_up else 10.0

    # Heuristic confidence score 0–1
    inlier_score = min(best_inliers / 150, 1.0)
    cluster_score = min(cluster_size / 5, 1.0)
    unique_score = min(uniqueness / 5, 1.0)

    return (inlier_score * 0.5 + cluster_score * 0.3 + unique_score * 0.2)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| GUI appears blank | macOS tkinter bug | `brew install python-tk@3.11` |
| `ModuleNotFoundError: lightglue` | Not installed from GitHub | `pip install git+https://github.com/cvg/LightGlue.git` |
| CUDA out of memory | GPU VRAM too low | Reduce `max_num_keypoints` in extractor init |
| Very slow matching | Running on CPU | Ensure CUDA/MPS device is selected |
| No matches found | Query image outside indexed area | Increase radius or re-index correct area |
| Low confidence result | Blurry/dark/low-texture image | Enable Ultra Mode |
| Index build interrupted | Network or crash | Safe to restart — resumes from last checkpoint |
| LoFTR import error | kornia not installed | `pip install kornia` |
| AI Coarse mode fails | Missing API key | `export GEMINI_API_KEY="..."` |

### Check device detection

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")
# cuda → ALIKED (1024 keypoints)
# mps  → DISK (768 keypoints)
# cpu  → DISK (768 keypoints, slow)
```

### Verify index integrity

```python
import numpy as np

descriptors = np.load("index/cosplace_descriptors.npy")
meta = np.load("index/metadata.npz", allow_pickle=True)

print(f"Index size: {descriptors.shape[0]} panoramas")
print(f"Descriptor dim: {descriptors.shape[1]}")  # Should be 512
print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
```

---

## Model Reference

| Model | Role | Hardware |
|-------|------|----------|
| CosPlace (512-dim) | Global visual fingerprint | All |
| ALIKED (1024 kp) | Local keypoint extraction | CUDA only |
| DISK (768 kp) | Local keypoint extraction | MPS / CPU |
| LightGlue | Deep feature matching | All |
| LoFTR | Dense matching (Ultra Mode) | All (needs kornia) |

---

## Tips for Best Results

- **Index density**: Grid resolution 300 is the recommended default — don't lower it
- **Query photos**: Higher resolution, good lighting = better matches
- **Radius tuning**: Start small (1–2km) if you have a rough idea of location
- **Ultra Mode**: Always try for night shots, CCTV footage, or heavily compressed images
- **Multiple indexes**: You can maintain separate index folders per project and swap `index/` directory
- **Storage**: Budget ~70 MB per km² for dense urban areas
```
