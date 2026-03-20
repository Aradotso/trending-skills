```markdown
---
name: netryx-street-level-geolocation
description: Expert skill for using Netryx, the open-source local-first street-level geolocation engine that identifies GPS coordinates from street photos using CosPlace, ALIKED/DISK, and LightGlue.
triggers:
  - geolocate a street photo
  - find GPS coordinates from image
  - street level geolocation
  - netryx geolocation
  - index street view panoramas
  - locate where a photo was taken
  - osint geolocation tool
  - reverse geolocation from street image
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that finds precise GPS coordinates (sub-50m accuracy) from any street-level photograph. It crawls street-view panoramas, builds a local searchable index using CosPlace visual fingerprints, and verifies matches with ALIKED/DISK keypoint extraction + LightGlue deep feature matching — all on your own hardware, no cloud APIs required.

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

# Optional: LoFTR for Ultra Mode (difficult images)
pip install kornia
```

### GPU Support Matrix

| Hardware | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU (4GB+ VRAM) | CUDA | Uses ALIKED (1024 keypoints) — fastest |
| Apple Silicon (M1–M4) | MPS | Uses DISK (768 keypoints) |
| CPU only | CPU | Works but slow; DISK used |

### Optional: Gemini API for AI Coarse Mode

```bash
export GEMINI_API_KEY="your_key_here"   # From https://aistudio.google.com
```

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix:** `brew install python-tk@3.11` (match your Python version)

---

## Core Workflow

### Step 1 — Build an Index

The index is a database of CosPlace visual fingerprints for street-view panoramas in a geographic area. Build it once; reuse it for all searches.

**Via GUI:**
1. Select **Create** mode
2. Enter center lat/lon, radius (km), grid resolution (default: 300)
3. Click **Create Index**

**Index size estimates:**

| Radius | ~Panoramas | Build Time (M2 Max) | Disk |
|--------|-----------|---------------------|------|
| 0.5 km | ~500 | 30 min | ~60 MB |
| 1 km | ~2,000 | 1–2 hrs | ~250 MB |
| 5 km | ~30,000 | 8–12 hrs | ~3 GB |
| 10 km | ~100,000 | 24–48 hrs | ~7 GB |

Index saves incrementally — safe to interrupt and resume.

**High-performance standalone builder (for large areas):**

```bash
python build_index.py
```

### Step 2 — Search

**Via GUI:**
1. Select **Search** mode
2. Upload a street-level photo
3. Choose search method:
   - **Manual**: Enter approximate center lat/lon + radius
   - **AI Coarse**: Gemini analyzes visual clues to auto-guess region (requires `GEMINI_API_KEY`)
4. Click **Run Search** → **Start Full Search**
5. Result: GPS coordinates + confidence score on map

---

## Pipeline Architecture

```
Query Image
    │
    ├─ CosPlace descriptor (512-dim)
    ├─ Flipped descriptor (catches reversed perspectives)
    │
    ▼
Index Search
    ├─ Cosine similarity against all indexed descriptors
    ├─ Haversine radius filter
    └─ Top 500–1000 candidates
    │
    ▼
Geometric Verification (per candidate)
    ├─ Download panorama tiles → stitch → rectilinear crop
    ├─ Multi-FOV crops: 70°, 90°, 110°
    ├─ ALIKED (CUDA) / DISK (MPS/CPU) keypoint extraction
    ├─ LightGlue deep feature matching
    └─ RANSAC inlier filtering
    │
    ▼
Refinement
    ├─ Heading refinement: ±45° at 15° steps, top 15 candidates
    ├─ Spatial consensus clustering (50m cells)
    └─ Confidence scoring (clustering + uniqueness ratio)
    │
    ▼
📍 GPS Coordinates + Confidence Score
```

---

## Ultra Mode

Enable **Ultra Mode** checkbox in GUI for hard images (night, blur, low-texture):

| Feature | What it does |
|---------|-------------|
| LoFTR | Detector-free dense matching — works without clean keypoints |
| Descriptor hopping | Re-searches index using the matched panorama's clean descriptor |
| Neighborhood expansion | Searches all panoramas within 100m of best match |

Ultra Mode is significantly slower. Use for images that fail standard search.

---

## Project Structure

```
netryx/
├── test_super.py           # Main app: GUI + indexing + search pipeline
├── cosplace_utils.py       # CosPlace model loading + descriptor extraction
├── build_index.py          # Standalone index builder for large datasets
├── requirements.txt
├── cosplace_parts/         # Raw embedding chunks (.npz), created during indexing
└── index/
    ├── cosplace_descriptors.npy    # All 512-dim CosPlace descriptors
    └── metadata.npz                # Lat/lon, headings, panorama IDs
```

---

## Models Reference

| Model | Role | Source |
|-------|------|--------|
| CosPlace | Global visual place recognition (retrieval) | [github.com/gmberton/cosplace](https://github.com/gmberton/cosplace) |
| ALIKED | Local keypoints — CUDA devices | [github.com/naver/alike](https://github.com/naver/alike) |
| DISK | Local keypoints — MPS/CPU devices | [github.com/cvlab-epfl/disk](https://github.com/cvlab-epfl/disk) |
| LightGlue | Deep feature matching | [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue) |
| LoFTR | Dense detector-free matching (Ultra Mode) | [github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) |

---

## Code Examples

### Extract a CosPlace Descriptor Manually

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

# Load model (downloads weights on first run)
model = load_cosplace_model()
model.eval()

# Extract descriptor from an image file
img = Image.open("query_photo.jpg").convert("RGB")
descriptor = extract_descriptor(model, img)  # Returns torch.Tensor shape [512]

print(f"Descriptor shape: {descriptor.shape}")   # torch.Size([512])
print(f"Descriptor norm: {descriptor.norm():.4f}")  # Should be ~1.0 (L2 normalized)
```

### Compute Cosine Similarity Against Index

```python
import numpy as np
import torch

# Load pre-built index
descriptors = np.load("index/cosplace_descriptors.npy")   # shape: [N, 512]
meta = np.load("index/metadata.npz", allow_pickle=True)

lats = meta["lats"]      # shape: [N]
lons = meta["lons"]      # shape: [N]
panoids = meta["panoids"]  # shape: [N]

# Your query descriptor (from extract_descriptor above)
query = descriptor.numpy()  # shape: [512]

# Cosine similarity (descriptors are L2-normalized, so dot product = cosine sim)
similarities = descriptors @ query   # shape: [N]

# Get top 10 matches
top_indices = np.argsort(similarities)[::-1][:10]

for i, idx in enumerate(top_indices):
    print(f"Rank {i+1}: lat={lats[idx]:.6f}, lon={lons[idx]:.6f}, "
          f"panoid={panoids[idx]}, sim={similarities[idx]:.4f}")
```

### Haversine Radius Filter

```python
import numpy as np

def haversine_km(lat1, lon1, lats2, lons2):
    """Vectorized haversine distance from one point to many points."""
    R = 6371.0
    dlat = np.radians(lats2 - lat1)
    dlon = np.radians(lons2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lats2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def search_index(query_descriptor, center_lat, center_lon, radius_km=1.0, top_k=500):
    """Search the index within a geographic radius."""
    descriptors = np.load("index/cosplace_descriptors.npy")
    meta = np.load("index/metadata.npz", allow_pickle=True)
    lats, lons = meta["lats"], meta["lons"]

    # Step 1: radius filter
    distances = haversine_km(center_lat, center_lon, lats, lons)
    in_radius = np.where(distances <= radius_km)[0]

    if len(in_radius) == 0:
        print("No indexed panoramas found in this area. Index it first.")
        return []

    # Step 2: cosine similarity within radius
    query = query_descriptor.numpy() if hasattr(query_descriptor, 'numpy') else query_descriptor
    sims = descriptors[in_radius] @ query

    # Step 3: rank and return top_k
    ranked = in_radius[np.argsort(sims)[::-1][:top_k]]
    return [
        {"lat": lats[i], "lon": lons[i], "panoid": meta["panoids"][i], "sim": sims[np.where(in_radius == i)[0][0]]}
        for i in ranked
    ]

# Example usage
results = search_index(descriptor, center_lat=48.8566, center_lon=2.3522, radius_km=2.0)
print(f"Top match: {results[0]}")
```

### LightGlue Matching (Verification Stage)

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Choose extractor based on device
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def match_images(query_path, candidate_path):
    """Match two images and return inlier count."""
    img0 = load_image(query_path).to(device)
    img1 = load_image(candidate_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matched_kps0 = feats0["keypoints"][matches01["matches"][..., 0]]
    matched_kps1 = feats1["keypoints"][matches01["matches"][..., 1]]

    n_matches = len(matched_kps0)
    print(f"LightGlue matches: {n_matches}")
    return n_matches, matched_kps0, matched_kps1

# RANSAC verification (requires OpenCV)
import cv2

def ransac_inliers(kps0, kps1, threshold=3.0):
    """Filter LightGlue matches with RANSAC homography."""
    if len(kps0) < 4:
        return 0
    pts0 = kps0.cpu().numpy()
    pts1 = kps1.cpu().numpy()
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, threshold)
    if mask is None:
        return 0
    return int(mask.sum())

n_matches, kps0, kps1 = match_images("query.jpg", "candidate_crop.jpg")
inliers = ransac_inliers(kps0, kps1)
print(f"RANSAC inliers: {inliers}")  # >50 = likely correct match
```

### Spatial Consensus Clustering

```python
import numpy as np
from collections import defaultdict

def cluster_candidates(candidates, cell_size_m=50):
    """
    Group candidates into geographic cells.
    Prefer clusters over isolated high-inlier outliers.
    """
    cell_deg = cell_size_m / 111_000  # ~degrees per meter

    clusters = defaultdict(list)
    for c in candidates:
        cell_lat = round(c["lat"] / cell_deg) * cell_deg
        cell_lon = round(c["lon"] / cell_deg) * cell_deg
        clusters[(cell_lat, cell_lon)].append(c)

    # Score each cluster: size * max_inliers in cluster
    best_cluster = max(
        clusters.values(),
        key=lambda group: len(group) * max(c.get("inliers", 0) for c in group)
    )

    best_match = max(best_cluster, key=lambda c: c.get("inliers", 0))
    print(f"Cluster size: {len(best_cluster)}, best inliers: {best_match['inliers']}")
    return best_match

# Example
candidates = [
    {"lat": 48.8566, "lon": 2.3522, "inliers": 120},
    {"lat": 48.8567, "lon": 2.3523, "inliers": 95},
    {"lat": 48.8568, "lon": 2.3521, "inliers": 80},
    {"lat": 51.5074, "lon": -0.1278, "inliers": 200},  # outlier — different city
]
result = cluster_candidates(candidates)
# Returns Paris cluster winner, not the London outlier with 200 inliers
```

---

## Multi-FOV Crop Generation

Netryx tests 3 fields of view to handle zoom mismatches between query photos and indexed panoramas:

```python
import numpy as np
from PIL import Image

def rectilinear_crop(panorama: Image.Image, heading_deg: float, fov_deg: float,
                      pitch_deg: float = 0, output_size: int = 640) -> Image.Image:
    """
    Extract a rectilinear perspective crop from an equirectangular panorama.
    heading_deg: 0=North, 90=East, 180=South, 270=West
    fov_deg: horizontal field of view (70, 90, or 110 for multi-FOV)
    """
    pano_w, pano_h = panorama.size
    pano = np.array(panorama).astype(np.float32)

    fov_rad = np.radians(fov_deg)
    f = (output_size / 2) / np.tan(fov_rad / 2)

    # Build grid of output pixel coordinates
    u = np.linspace(-output_size / 2, output_size / 2, output_size)
    v = np.linspace(-output_size / 2, output_size / 2, output_size)
    uu, vv = np.meshgrid(u, v)

    # Direction vectors
    x = uu
    y = f
    z = -vv  # image y-axis goes down

    # Rotate by heading
    heading_rad = np.radians(heading_deg)
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    xr = cos_h * x - sin_h * y
    yr = sin_h * x + cos_h * y
    zr = z

    # Convert to spherical
    lon = np.arctan2(xr, yr)          # [-π, π]
    lat = np.arctan2(zr, np.sqrt(xr**2 + yr**2))  # pitch

    # Map to panorama pixel coords
    px = ((lon / (2 * np.pi)) + 0.5) * pano_w
    py = (0.5 - lat / np.pi) * pano_h
    px = np.clip(px.astype(np.int32), 0, pano_w - 1)
    py = np.clip(py.astype(np.int32), 0, pano_h - 1)

    crop = pano[py, px]
    return Image.fromarray(crop.astype(np.uint8))

# Generate multi-FOV crops for a candidate
for fov in [70, 90, 110]:
    crop = rectilinear_crop(panorama_img, heading_deg=135.0, fov_deg=fov)
    crop.save(f"candidate_fov{fov}.jpg")
```

---

## Index Management

The index is source-agnostic — works with Mapillary, KartaView, or any geo-tagged street imagery provider.

```python
import numpy as np
import os

def inspect_index(index_dir="index"):
    """Print index statistics."""
    desc_path = os.path.join(index_dir, "cosplace_descriptors.npy")
    meta_path = os.path.join(index_dir, "metadata.npz")

    if not os.path.exists(desc_path):
        print("Index not found. Run Create mode first.")
        return

    descriptors = np.load(desc_path)
    meta = np.load(meta_path, allow_pickle=True)

    print(f"Total panoramas indexed: {len(descriptors):,}")
    print(f"Descriptor shape: {descriptors.shape}")
    print(f"Lat range: {meta['lats'].min():.4f} – {meta['lats'].max():.4f}")
    print(f"Lon range: {meta['lons'].min():.4f} – {meta['lons'].max():.4f}")
    print(f"Index size: {os.path.getsize(desc_path) / 1e6:.1f} MB")

inspect_index()
```

```python
def merge_index_parts(parts_dir="cosplace_parts", output_dir="index"):
    """
    Manually merge cosplace_parts/*.npz into a unified searchable index.
    Netryx does this automatically, but useful if auto-build is interrupted.
    """
    import glob
    os.makedirs(output_dir, exist_ok=True)

    all_descs, all_lats, all_lons, all_panoids, all_headings = [], [], [], [], []

    for path in sorted(glob.glob(os.path.join(parts_dir, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        all_descs.append(data["descriptors"])
        all_lats.append(data["lats"])
        all_lons.append(data["lons"])
        all_panoids.append(data["panoids"])
        all_headings.append(data["headings"])
        print(f"  Loaded {path}: {len(data['lats'])} entries")

    descriptors = np.vstack(all_descs).astype(np.float32)
    # L2-normalize (required for cosine similarity via dot product)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors /= np.maximum(norms, 1e-8)

    np.save(os.path.join(output_dir, "cosplace_descriptors.npy"), descriptors)
    np.savez(os.path.join(output_dir, "metadata.npz"),
             lats=np.concatenate(all_lats),
             lons=np.concatenate(all_lons),
             panoids=np.concatenate(all_panoids),
             headings=np.concatenate(all_headings))

    print(f"\nIndex built: {len(descriptors):,} panoramas → {output_dir}/")

merge_index_parts()
```

---

## Common Patterns

### Pattern 1: Batch Geolocate Multiple Images

```python
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image

model = load_cosplace_model()
model.eval()

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
center_lat, center_lon = 48.8566, 2.3522  # Paris
radius_km = 3.0

for path in image_paths:
    img = Image.open(path).convert("RGB")
    desc = extract_descriptor(model, img)
    candidates = search_index(desc, center_lat, center_lon, radius_km, top_k=300)
    if candidates:
        print(f"{path}: best match at ({candidates[0]['lat']:.5f}, {candidates[0]['lon']:.5f})")
    else:
        print(f"{path}: no candidates found in radius")
```

### Pattern 2: Confidence Score Heuristic

```python
def confidence_score(candidates, top_n=10):
    """
    Estimate match confidence from candidate distribution.
    High confidence: top match clearly better than runners-up AND geographically clustered.
    """
    if not candidates:
        return 0.0

    sorted_c = sorted(candidates, key=lambda x: x.get("inliers", 0), reverse=True)
    best = sorted_c[0].get("inliers", 0)

    if best == 0:
        return 0.0

    # Uniqueness ratio: best vs second-best at a different location
    runner_up = next(
        (c for c in sorted_c[1:top_n]
         if haversine_km(sorted_c[0]["lat"], sorted_c[0]["lon"],
                         np.array([c["lat"]]), np.array([c["lon"]]))[0] > 0.1),
        None
    )
    uniqueness = best / (runner_up["inliers"] + 1) if runner_up else 5.0

    # Cluster score: how many top-N candidates are near the best match?
    nearby = sum(
        1 for c in sorted_c[1:top_n]
        if haversine_km(sorted_c[0]["lat"], sorted_c[0]["lon"],
                        np.array([c["lat"]]), np.array([c["lon"]]))[0] < 0.05
    )

    score = min(1.0, (uniqueness / 3.0) * 0.5 + (nearby / top_n) * 0.5)
    return round(score, 3)

# score > 0.7 → high confidence
# score 0.4–0.7 → medium, consider Ultra Mode
# score < 0.4 → low, try broader radius or Ultra Mode
```

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # Match your Python version (3.10, 3.12, etc.)
```

### LightGlue import error
```bash
# Must install from GitHub, not PyPI
pip uninstall lightglue -y
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory
```python
# Reduce keypoints in extractor
extractor = ALIKED(max_num_keypoints=512).eval().to(device)  # default 1024
```

### "No candidates found in radius"
- The area hasn't been indexed yet → run Create mode for those coordinates
- Radius too small → increase search radius
- Index not built from parts → check `cosplace_parts/` has `.npz` files, rebuild index

### Slow indexing
- Use `build_index.py` instead of the GUI for large areas (more efficient)
- Indexing requires internet (downloads street view tiles); broadband recommended
- Process is interruptible — resumes from last saved chunk

### Low match quality (few inliers)
- Enable **Ultra Mode** (LoFTR + descriptor hopping + neighborhood expansion)
- Increase `top_k` candidates in search
- Try a larger search radius
- Check image quality: motion blur, extreme night scenes, and heavily filtered photos are hardest

### MPS device errors (Apple Silicon)
```python
# If MPS causes issues, force CPU
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### Index descriptors not L2-normalized
```python
# Re-normalize if similarity scores seem off (all near 0 or >1)
import numpy as np
descs = np.load("index/cosplace_descriptors.npy")
norms = np.linalg.norm(descs, axis=1, keepdims=True)
descs = descs / np.maximum(norms, 1e-8)
np.save("index/cosplace_descriptors.npy", descs)
```

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Grid resolution | 300 | Higher = denser panorama coverage. Don't change unless necessary. |
| top_k candidates | 500–1000 | More = higher recall, slower verification |
| Keypoints (ALIKED) | 1024 | Reduce to 512 for low VRAM |
| Keypoints (DISK) | 768 | MPS/CPU default |
| RANSAC threshold | 3.0 px | Lower = stricter geometric verification |
| Heading refinement range | ±45° @ 15° steps | Applied to top 15 candidates |
| Cluster cell size | 50m | For spatial consensus grouping |
| Ultra neighborhood | 100m | Expansion radius around best match |
| Multi-FOV values | 70°, 90°, 110° | Handles zoom mismatch between query and indexed view |
```
