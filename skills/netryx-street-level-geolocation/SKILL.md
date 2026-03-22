```markdown
---
name: netryx-street-level-geolocation
description: Use Netryx to index street-view panoramas and geolocate any street-level photo to GPS coordinates using CosPlace, ALIKED/DISK, and LightGlue computer vision pipelines.
triggers:
  - geolocate a street photo
  - find GPS coordinates from an image
  - street level geolocation
  - index street view panoramas
  - run netryx search
  - identify location from photo
  - osint geolocation from image
  - netryx pipeline setup
---

# Netryx Street-Level Geolocation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Netryx is a locally-hosted geolocation engine that identifies the exact GPS coordinates of any street-level photograph. It crawls street-view panoramas, indexes them as 512-dimensional CosPlace fingerprints, then matches a query image through a three-stage pipeline: global retrieval → local geometric verification → spatial refinement. Sub-50m accuracy, no landmarks required, runs entirely on local hardware.

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

### Optional — Gemini AI Coarse Mode

```bash
export GEMINI_API_KEY="your_key_here"   # from https://aistudio.google.com
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB    | 8 GB+       |
| RAM       | 8 GB    | 16 GB+      |
| Storage   | 10 GB   | 50 GB+      |
| Python    | 3.9+    | 3.10+       |

GPU backends: **CUDA** (NVIDIA) → uses ALIKED · **MPS** (Apple Silicon) → uses DISK · **CPU** → uses DISK, slowest.

---

## Project Structure

```
netryx/
├── test_super.py          # Main GUI application — indexing + search
├── cosplace_utils.py      # CosPlace model loading and descriptor extraction
├── build_index.py         # High-performance standalone index builder
├── requirements.txt
├── cosplace_parts/        # Raw embedding chunks written during indexing
└── index/
    ├── cosplace_descriptors.npy   # All 512-dim descriptors (searchable)
    └── metadata.npz               # Lat/lng, headings, panorama IDs
```

---

## Launch the GUI

```bash
python test_super.py
```

> **macOS blank GUI fix:** `brew install python-tk@3.11` (match your Python version).

The GUI has two modes selectable at startup: **Create** (index an area) and **Search** (geolocate a photo).

---

## Stage 1 — Create an Index

Index a geographic area before searching. The index is built incrementally and resumes if interrupted.

### GUI workflow
1. Select **Create** mode.
2. Enter center coordinates (lat, lng).
3. Set radius in km (start with `0.5`–`1` for testing).
4. Set grid resolution — default `300`, do not change.
5. Click **Create Index**.

### Indexing time/storage estimates

| Radius | ~Panoramas | Time (M2 Max) | Index size |
|--------|-----------|---------------|------------|
| 0.5 km | ~500      | 30 min        | ~60 MB     |
| 1 km   | ~2 000    | 1–2 hrs       | ~250 MB    |
| 5 km   | ~30 000   | 8–12 hrs      | ~3 GB      |
| 10 km  | ~100 000  | 24–48 hrs     | ~7 GB      |

### Standalone index builder (large areas)

Use `build_index.py` directly for big datasets — it is more efficient than the GUI path:

```bash
python build_index.py \
  --lat 48.8566 \
  --lng 2.3522 \
  --radius 2.0 \
  --resolution 300
```

---

## Stage 2 — Search (Geolocate a Photo)

### GUI workflow
1. Select **Search** mode.
2. Upload the street-level photo.
3. Choose search method:
   - **Manual** — enter approximate center coordinates + radius if you know the region.
   - **AI Coarse** — Gemini analyzes signs/architecture/vegetation to estimate region (requires `GEMINI_API_KEY`).
4. Click **Run Search** → **Start Full Search**.
5. Watch the real-time candidate visualization; the result appears on the map with a confidence score.

### Enable Ultra Mode

Check **Ultra Mode** for degraded images (night, blur, low texture). Adds:
- **LoFTR** dense matching (no keypoint detection needed).
- **Descriptor hopping** — re-searches the index using the matched panorama's clean descriptor.
- **Neighborhood expansion** — checks all panoramas within 100 m of the top match.

Ultra Mode is 3–5× slower but significantly more robust.

---

## Pipeline Internals — Code Patterns

### Load CosPlace and extract a descriptor

```python
# cosplace_utils.py exposes these helpers
from cosplace_utils import load_cosplace_model, extract_descriptor
from PIL import Image
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = load_cosplace_model(device=device)   # downloads weights on first run

img = Image.open("query.jpg").convert("RGB")
descriptor = extract_descriptor(model, img, device=device)
# descriptor.shape → (512,)  float32 numpy array
```

### Index search — cosine similarity with radius filter

```python
import numpy as np

def search_index(query_desc, index_path="index", center_lat, center_lng, radius_km=2.0, top_k=500):
    descriptors = np.load(f"{index_path}/cosplace_descriptors.npy")   # (N, 512)
    meta        = np.load(f"{index_path}/metadata.npz", allow_pickle=True)
    lats        = meta["lats"]
    lngs        = meta["lngs"]
    headings    = meta["headings"]
    panoids     = meta["panoids"]

    # Haversine radius filter
    R = 6371.0
    dlat = np.radians(lats - center_lat)
    dlng = np.radians(lngs - center_lng)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(lats)) * np.sin(dlng/2)**2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))
    mask = dist_km <= radius_km

    filtered_desc = descriptors[mask]
    # Cosine similarity (descriptors are L2-normalised)
    sims = filtered_desc @ query_desc          # (M,)
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    filtered_indices = np.where(mask)[0]
    for rank, i in enumerate(top_idx):
        orig_i = filtered_indices[i]
        results.append({
            "rank":     rank,
            "score":    float(sims[i]),
            "lat":      float(lats[orig_i]),
            "lng":      float(lngs[orig_i]),
            "heading":  float(headings[orig_i]),
            "panoid":   str(panoids[orig_i]),
        })
    return results
```

### Feature extraction — ALIKED (CUDA) vs DISK (MPS/CPU)

```python
import torch
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")

# Pick extractor based on backend
if device.type == "cuda":
    extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
else:
    extractor = DISK(max_num_keypoints=768).eval().to(device)

matcher = LightGlue(features="aliked" if device.type == "cuda" else "disk").eval().to(device)

def match_images(query_path: str, candidate_path: str) -> int:
    """Returns number of RANSAC-verified inliers between two images."""
    img0 = load_image(query_path).to(device)
    img1 = load_image(candidate_path).to(device)

    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
    kpts0 = feats0["keypoints"][matches01["matches"][..., 0]]
    kpts1 = feats1["keypoints"][matches01["matches"][..., 1]]

    if len(kpts0) < 8:
        return 0

    import cv2
    pts0 = kpts0.cpu().numpy()
    pts1 = kpts1.cpu().numpy()
    _, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.999)
    return int(mask.sum()) if mask is not None else 0
```

### Ultra Mode — LoFTR dense matching

```python
import kornia.feature as KF
import torch, cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

def loftr_inliers(img0_gray: np.ndarray, img1_gray: np.ndarray) -> int:
    def to_tensor(arr):
        t = torch.from_numpy(arr).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)

    with torch.no_grad():
        out = loftr({"image0": to_tensor(img0_gray), "image1": to_tensor(img1_gray)})

    pts0 = out["keypoints0"].cpu().numpy()
    pts1 = out["keypoints1"].cpu().numpy()
    if len(pts0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.999)
    return int(mask.sum()) if mask is not None else 0
```

---

## Common Patterns

### Pattern 1 — Batch index multiple cities

```python
cities = [
    ("Paris",    48.8566,  2.3522,  5.0),
    ("London",   51.5074, -0.1278,  5.0),
    ("Tel Aviv", 32.0853, 34.7818,  3.0),
]
# All go into the same unified index; radius-filtering at search time separates them.
for name, lat, lng, radius in cities:
    print(f"Indexing {name}...")
    # Call build_index.py or the GUI Create flow for each city
```

### Pattern 2 — Heading refinement for top candidates

```python
import numpy as np

def refine_heading(base_heading: float, steps: int = 7, delta: float = 45.0):
    """Generate ±delta° heading offsets at `steps` intervals."""
    offsets = np.linspace(-delta, delta, steps)
    return [(base_heading + o) % 360 for o in offsets]

fovs = [70, 90, 110]   # degrees — three crops per heading

for candidate in top_15_candidates:
    best_inliers = 0
    for heading in refine_heading(candidate["heading"]):
        for fov in fovs:
            crop = fetch_street_view_crop(candidate["panoid"], heading, fov)
            inliers = match_images(query_path, crop)
            if inliers > best_inliers:
                best_inliers = inliers
                candidate["best_heading"] = heading
                candidate["best_fov"]     = fov
    candidate["refined_inliers"] = best_inliers
```

### Pattern 3 — Spatial consensus clustering

```python
from collections import defaultdict

def cluster_candidates(candidates, cell_size_m=50):
    """Group candidates into ~50m grid cells; prefer largest cluster."""
    R = 6_371_000
    clusters = defaultdict(list)
    for c in candidates:
        cell_lat = round(c["lat"] / (cell_size_m / R * (180 / np.pi)))
        cell_lng = round(c["lng"] / (cell_size_m / R * (180 / np.pi) / np.cos(np.radians(c["lat"]))))
        clusters[(cell_lat, cell_lng)].append(c)

    best_cluster = max(clusters.values(), key=lambda g: sum(x["refined_inliers"] for x in g))
    return max(best_cluster, key=lambda x: x["refined_inliers"])
```

### Pattern 4 — Descriptor hopping (Ultra Mode)

```python
def descriptor_hop(matched_panoid: str, index_path: str, center_lat, center_lng, radius_km=0.1):
    """
    Extract CosPlace descriptor from the matched (clean) panorama
    and re-search the index. Useful when query is degraded.
    """
    pano_img = download_panorama(matched_panoid)          # fetch clean GSV image
    hopped_desc = extract_descriptor(model, pano_img, device=device)
    return search_index(hopped_desc, index_path, center_lat, center_lng, radius_km)
```

---

## Configuration Reference

| Parameter | Where set | Default | Notes |
|-----------|-----------|---------|-------|
| Grid resolution | GUI / `build_index.py --resolution` | `300` | Do not change |
| Search radius | GUI / `search_index()` | varies | In km |
| Top-K candidates Stage 1 | `search_index(top_k=)` | 500–1000 | More = slower Stage 2 |
| Heading refinement steps | hardcoded | ±45° / 15° steps | Adjust in `refine_heading()` |
| RANSAC threshold | `cv2.findFundamentalMat` | `3.0 px` | Lower = stricter |
| ALIKED keypoints | `ALIKED(max_num_keypoints=)` | `1024` | CUDA only |
| DISK keypoints | `DISK(max_num_keypoints=)` | `768` | MPS/CPU |
| Ultra neighborhood | hardcoded | `100 m` | Panoramas near best match |
| Gemini API key | `GEMINI_API_KEY` env var | — | AI Coarse mode only |

---

## Troubleshooting

### GUI appears blank on macOS
```bash
brew install python-tk@3.11   # match your exact Python version
```

### `ImportError: No module named 'lightglue'`
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### `kornia` not found (Ultra Mode disabled)
```bash
pip install kornia
```

### CUDA out of memory
Reduce `max_num_keypoints`: `ALIKED(max_num_keypoints=512)` or switch to DISK.

### Index search returns no results
- Confirm center coordinates and radius actually overlap the indexed area.
- Run **Auto-build** after indexing (the GUI triggers this automatically; for manual runs check that `index/cosplace_descriptors.npy` exists).
- Check `cosplace_parts/` is non-empty — if empty, re-run Create Index.

### Poor match quality (low inlier counts)
1. Enable **Ultra Mode**.
2. Increase `top_k` candidates (try 1000).
3. Widen the search radius.
4. Check the query image is street-level (aerial/indoor photos are not supported).

### Indexing interrupted
Safe to restart — the builder reads existing `cosplace_parts/*.npz` files and skips already-processed grid points.

### Slow Stage 2 on CPU
Stage 2 (ALIKED/DISK + LightGlue) is GPU-bound. On CPU, reduce candidates:
```python
results = search_index(..., top_k=100)   # instead of 500
```
and use a narrower radius to reduce candidates.

---

## Full End-to-End Example

```python
"""
Minimal programmatic geolocation pipeline (no GUI).
Requires an existing index at ./index/
"""
import torch, numpy as np
from PIL import Image
from cosplace_utils import load_cosplace_model, extract_descriptor

# ── Config ──────────────────────────────────────────────────────────────────
QUERY_IMAGE  = "query.jpg"
CENTER_LAT   = 48.8566
CENTER_LNG   = 2.3522
RADIUS_KM    = 1.0
INDEX_PATH   = "index"
TOP_K        = 300

# ── Device ──────────────────────────────────────────────────────────────────
device = ("cuda" if torch.cuda.is_available() else
          "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Stage 1: Global Retrieval ────────────────────────────────────────────────
model      = load_cosplace_model(device=device)
img        = Image.open(QUERY_IMAGE).convert("RGB")
desc       = extract_descriptor(model, img, device=device)
desc_flip  = extract_descriptor(model, img.transpose(Image.FLIP_LEFT_RIGHT), device=device)
combined   = (desc + desc_flip) / 2                        # average both orientations
combined  /= np.linalg.norm(combined)

candidates = search_index(combined, INDEX_PATH, CENTER_LAT, CENTER_LNG, RADIUS_KM, TOP_K)
print(f"Stage 1: {len(candidates)} candidates retrieved")

# ── Stage 2: Geometric Verification ─────────────────────────────────────────
from lightglue import LightGlue, ALIKED, DISK
from lightglue.utils import load_image, rbd

extractor = (ALIKED(max_num_keypoints=1024) if device == "cuda"
             else DISK(max_num_keypoints=768)).eval().to(device)
feat_name = "aliked" if device == "cuda" else "disk"
matcher   = LightGlue(features=feat_name).eval().to(device)

best = {"inliers": 0, "candidate": None}
for cand in candidates:
    pano_crop = download_and_crop(cand["panoid"], cand["heading"])   # your fetch helper
    inliers   = match_images(QUERY_IMAGE, pano_crop)
    if inliers > best["inliers"]:
        best = {"inliers": inliers, "candidate": cand}

result = best["candidate"]
print(f"\n📍 Result: {result['lat']:.6f}, {result['lng']:.6f}")
print(f"   Inliers: {best['inliers']}  |  Score: {result['score']:.4f}")
print(f"   Maps: https://maps.google.com/?q={result['lat']},{result['lng']}")
```
```
