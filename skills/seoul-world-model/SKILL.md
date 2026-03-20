```markdown
---
name: seoul-world-model
description: Skill for using the Seoul World Model — a world simulation model grounded in a real-world metropolis (Seoul) by Naver AI
triggers:
  - seoul world model
  - world simulation model
  - grounding world model in real city
  - street view world model
  - naver seoul simulation
  - urban world model inference
  - seoul street view generation
  - metropolis world model
---

# Seoul World Model

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What Is Seoul World Model?

**Seoul World Model** (by Naver AI) is a research project that grounds world simulation models in real-world urban data from Seoul, South Korea. It enables:

- **World simulation**: Generate realistic video continuations of street-level scenes in Seoul
- **Street-view interpolation**: Synthesize smooth video transitions between street-view frames
- **Urban scene understanding**: Leverage a large-scale real-world metropolis dataset for training/evaluation

The project provides:
- Model checkpoints for world simulation inference
- Synthetic training data (Seoul street-view)
- Street-view interpolation model code and checkpoints

> ⚠️ **Note**: As of March 2026, the repository is undergoing internal review. Model checkpoints, inference code, and training data are planned for release. Monitor the [project page](https://seoul-world-model.github.io/#tldr) and repository for updates.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/naver-ai/seoul-world-model.git
cd seoul-world-model
```

### Python Environment (Recommended)

```bash
# Create and activate a conda environment
conda create -n seoul-world-model python=3.10 -y
conda activate seoul-world-model

# Install dependencies (once requirements.txt is released)
pip install -r requirements.txt
```

### Common Deep Learning Dependencies (Anticipated)

Based on the project type (video generation / world models), install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install einops timm imageio[ffmpeg] opencv-python
pip install numpy pillow tqdm
```

---

## Project Structure (Anticipated)

```
seoul-world-model/
├── README.md
├── checkpoints/          # Model weights (to be released)
├── data/                 # Synthetic training data (to be released)
├── inference/            # Inference scripts (to be released)
│   ├── world_model.py
│   └── interpolation.py
├── train/                # Training code (to be released)
├── configs/              # Model and training configs
└── utils/                # Utilities
```

---

## Key Concepts

| Component | Description |
|-----------|-------------|
| **World Simulation Model** | Generates future video frames conditioned on current observations and actions |
| **Street-View Interpolation** | Fills in smooth transitions between sparse street-view keyframes |
| **Seoul Dataset** | Large-scale real-world urban driving/walking data from Seoul |
| **Grounding** | Training on real-world data to improve simulation realism and physical plausibility |

---

## Inference (Anticipated API Pattern)

Once released, inference will likely follow this pattern:

### World Model Inference

```python
import torch
from PIL import Image

# Load model (path subject to change on release)
# from inference.world_model import SeoulWorldModel

# model = SeoulWorldModel.from_pretrained("checkpoints/world_model")
# model = model.to("cuda").eval()

# Prepare input frames
def load_frames(image_paths: list[str]) -> torch.Tensor:
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    frames = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(frames).unsqueeze(0)  # (1, T, C, H, W)

# context_frames = load_frames(["frame_000.jpg", "frame_001.jpg", "frame_002.jpg"])

# Run generation
# with torch.no_grad():
#     generated_frames = model.generate(
#         context=context_frames.cuda(),
#         num_frames=16,
#         guidance_scale=7.5,
#     )

# Save output
# save_video(generated_frames, "output.mp4", fps=10)
```

### Street-View Interpolation

```python
# Interpolate between two keyframe images
# from inference.interpolation import StreetViewInterpolator

# interpolator = StreetViewInterpolator.from_pretrained("checkpoints/interpolation")
# interpolator = interpolator.to("cuda").eval()

from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(path: str, size=(256, 512)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)

# frame_a = preprocess_image("frame_start.jpg").cuda()
# frame_b = preprocess_image("frame_end.jpg").cuda()

# with torch.no_grad():
#     interpolated = interpolator.interpolate(
#         frame_a, frame_b,
#         num_intermediate=8,
#     )
# save_video(interpolated, "interpolated.mp4", fps=8)
```

### Utility: Save Video

```python
import imageio
import numpy as np
import torch

def save_video(frames: torch.Tensor, output_path: str, fps: int = 10):
    """
    Save a tensor of frames as an MP4 video.
    
    Args:
        frames: Tensor of shape (T, C, H, W) in range [-1, 1] or [0, 1]
        output_path: Path to save .mp4
        fps: Frames per second
    """
    # Denormalize if in [-1, 1]
    if frames.min() < 0:
        frames = (frames + 1) / 2
    
    frames_np = (frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames_np:
            writer.append_data(frame)
    
    print(f"Saved video to {output_path}")
```

---

## Configuration (Anticipated)

World model configs will likely be YAML-based:

```yaml
# configs/world_model.yaml (example structure)
model:
  type: "SeoulWorldModel"
  checkpoint: "checkpoints/world_model/model.ckpt"
  image_size: [256, 512]
  num_frames: 16
  temporal_stride: 2

inference:
  guidance_scale: 7.5
  num_inference_steps: 50
  seed: 42
  device: "cuda"

data:
  context_frames: 3
  fps: 10
```

Load config in Python:

```python
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config("configs/world_model.yaml")
print(config["model"]["checkpoint"])
```

---

## Environment Variables

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Set checkpoint directory (if configurable via env)
export SEOUL_WM_CHECKPOINT_DIR=/path/to/checkpoints

# For HuggingFace model downloads (if applicable)
export HF_HOME=/path/to/hf_cache
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN  # do NOT hardcode tokens
```

---

## Common Patterns

### Batch Inference Over a Dataset

```python
import os
from pathlib import Path

def batch_infer(input_dir: str, output_dir: str, model, batch_size: int = 4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    for scene_dir in scene_dirs:
        frames = sorted(scene_dir.glob("*.jpg"))
        if len(frames) < 3:
            continue
        
        context = load_frames([str(f) for f in frames[:3]])
        
        with torch.no_grad():
            output = model.generate(context.cuda(), num_frames=16)
        
        out_path = output_dir / f"{scene_dir.name}_generated.mp4"
        save_video(output.squeeze(0), str(out_path))
        print(f"Processed: {scene_dir.name}")
```

### Evaluate Temporal Consistency (FVD-style)

```python
import torch
import torch.nn.functional as F

def compute_frame_similarity(generated: torch.Tensor) -> float:
    """
    Simple temporal consistency metric: average cosine similarity between adjacent frames.
    generated: (T, C, H, W)
    """
    T = generated.shape[0]
    similarities = []
    for t in range(T - 1):
        f1 = generated[t].flatten()
        f2 = generated[t + 1].flatten()
        sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        similarities.append(sim)
    return sum(similarities) / len(similarities)
```

---

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce resolution or use mixed precision
import torch

# Use bfloat16
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model.generate(context.cuda(), num_frames=8)  # reduce frames

# Or use CPU offloading (if supported by model)
# model.enable_model_cpu_offload()
```

### Repository Still Under Review

The codebase is not yet fully released (as of March 2026). Monitor:

```bash
# Check for updates
cd seoul-world-model
git fetch origin
git log --oneline origin/main

# Watch GitHub releases
# https://github.com/naver-ai/seoul-world-model/releases
```

### Dependency Conflicts

```bash
# If torch conflicts arise, install in order
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.0 transformers==4.39.0
```

---

## Resources

- **Project Page**: https://seoul-world-model.github.io/#tldr
- **GitHub**: https://github.com/naver-ai/seoul-world-model
- **Paper**: "Grounding World Simulation Models in a Real-World Metropolis"

---

## Release Checklist (Track Progress)

- [ ] Model checkpoints and inference code
- [ ] Synthetic training data
- [ ] Street-view interpolation model code and checkpoints
- [ ] Training scripts

Stay tuned to the repository for updates as these are released.
```
