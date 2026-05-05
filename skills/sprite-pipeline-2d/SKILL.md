```markdown
---
name: sprite-pipeline-2d
description: AI agent skill for the Sprite-Pipeline project — a reusable Python pipeline for turning video/frames into clean 256×256 horizontal sprite sheets with matting, review, and browser preview.
triggers:
  - "create a sprite sheet from video"
  - "extract frames for animation"
  - "build sprite strip from frames"
  - "matte background from sprite frames"
  - "set up sprite sheet pipeline"
  - "convert animation video to sprite sheet"
  - "review and promote sprite sheet output"
  - "view sprite sheets in browser"
---

# Sprite-Pipeline 2D

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A reusable Python pipeline that converts ordered animation frames (extracted from video) into clean horizontal `256×256` sprite strips, with optional background matting, JSON reports, contact-sheet previews, and a static browser viewer.

---

## What It Does

| Stage | Tool | Input → Output |
|---|---|---|
| Extract frames | `tools/extract_frames_ffmpeg.py` | `Videos/` → `work/extracted/` |
| Matte backgrounds | `tools/matte_frames.py` | `work/extracted/` → `work/matted/` |
| Build sprite strip | `tools/animation_pipeline.py` | frames dir → sprite sheet PNG + JSON report |
| Cleanup / repack | `tools/cleanup_repack.py` | loose frames → tidy cell folders |
| Contact sheet | `tools/contact_sheet.py` | frames dir → preview PNG |
| Resize | `tools/resize_sprites.py` | sheet PNG → scaled PNG |
| Gallery manifest | `tools/build_sprite_gallery_manifest.py` | `Final Sprite Sheets/` → `sprite_gallery_manifest.js` |
| Browser viewer | `sprite_viewer.html` | manifest → interactive preview |

---

## Installation & Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/LayrKits/Sprite-Pipeline.git
cd Sprite-Pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install FFmpeg (required for frame extraction)

```bash
# macOS
brew install ffmpeg

# Windows
winget install Gyan.FFmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 3. Verify setup

```bash
ffmpeg -version
python tools/animation_pipeline.py --help
```

---

## Folder Conventions

```
Sprite-Pipeline/
├── Videos/
│   └── To Be Processed/       # drop source .mp4/.mov here
├── work/
│   ├── extracted/<character>/<action>/   # raw frames from ffmpeg
│   └── matted/<character>/<action>/      # frames with alpha/white bg removed
├── Final Sprite Sheets/
│   └── <GameName>/<CharacterName>/<animation>/
├── Cleanup/
├── tools/
├── docs/
├── skills/
├── sprite_viewer.html
├── sprite_gallery_manifest.js
└── sprite_gallery_pins.json
```

---

## Key Commands

### Extract frames from video

```bash
python tools/extract_frames_ffmpeg.py \
  --input "Videos/hero_run.mp4" \
  --output "work/extracted/hero/run" \
  --fps 12
```

- `--fps` controls how many frames per second are extracted (match your target animation rate).
- Frames are written as zero-padded PNGs: `frame_0001.png`, `frame_0002.png`, …

### Matte light/white backgrounds

```bash
python tools/matte_frames.py \
  --input  "work/extracted/hero/run" \
  --output "work/matted/hero/run" \
  --threshold 240
```

- `--threshold` (0–255): pixels with all RGB channels above this value are made transparent.
- Outputs RGBA PNGs suitable for game engines.

### Build sprite sheet

```bash
python tools/animation_pipeline.py \
  --input  "work/matted/hero/run" \
  --output "work/previews/hero_run_sheet.png" \
  --size   256 \
  --report "work/previews/hero_run_report.json"
```

- `--size` sets the cell dimensions (default `256`; sheet will be `N×256` wide by `256` tall).
- `--report` writes a JSON file with frame count, dimensions, dropped frames, and warnings.
- Output is a **single horizontal strip**: all frames left-to-right in one row.

### Generate contact sheet (quick visual review)

```bash
python tools/contact_sheet.py \
  --input  "work/matted/hero/run" \
  --output "work/previews/hero_run_contact.png" \
  --cols   8
```

### Resize an existing sheet

```bash
python tools/resize_sprites.py \
  --input  "work/previews/hero_run_sheet.png" \
  --output "work/previews/hero_run_sheet_128.png" \
  --size   128
```

### Promote approved sheet

```bash
# Manually copy approved sheet + cells after review:
mkdir -p "Final Sprite Sheets/MyGame/Hero/run"
cp work/previews/hero_run_sheet.png "Final Sprite Sheets/MyGame/Hero/run/"
cp -r work/matted/hero/run/         "Final Sprite Sheets/MyGame/Hero/run/cells/"
```

### Rebuild gallery manifest and open viewer

```bash
python tools/build_sprite_gallery_manifest.py
# then open sprite_viewer.html in a browser (no server needed)
```

---

## End-to-End Workflow (copy-paste)

```bash
# 1. Extract
python tools/extract_frames_ffmpeg.py \
  --input  "Videos/To Be Processed/hero_run.mp4" \
  --output "work/extracted/hero/run" \
  --fps    12

# 2. Matte
python tools/matte_frames.py \
  --input     "work/extracted/hero/run" \
  --output    "work/matted/hero/run" \
  --threshold 240

# 3. Build sheet + report
python tools/animation_pipeline.py \
  --input  "work/matted/hero/run" \
  --output "work/previews/hero_run_sheet.png" \
  --size   256 \
  --report "work/previews/hero_run_report.json"

# 4. Review the JSON report
cat work/previews/hero_run_report.json

# 5. Promote if approved
mkdir -p "Final Sprite Sheets/MyGame/Hero/run"
cp work/previews/hero_run_sheet.png "Final Sprite Sheets/MyGame/Hero/run/"

# 6. Update viewer
python tools/build_sprite_gallery_manifest.py
open sprite_viewer.html    # macOS; or just double-click on Windows/Linux
```

---

## Python Usage (scripting the pipeline)

If you want to drive the pipeline from your own Python script:

```python
import subprocess, json, pathlib

def extract_frames(video_path: str, out_dir: str, fps: int = 12):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", "tools/extract_frames_ffmpeg.py",
        "--input",  video_path,
        "--output", out_dir,
        "--fps",    str(fps),
    ], check=True)

def matte_frames(in_dir: str, out_dir: str, threshold: int = 240):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", "tools/matte_frames.py",
        "--input",     in_dir,
        "--output",    out_dir,
        "--threshold", str(threshold),
    ], check=True)

def build_sheet(frames_dir: str, sheet_path: str, size: int = 256) -> dict:
    report_path = sheet_path.replace(".png", "_report.json")
    subprocess.run([
        "python", "tools/animation_pipeline.py",
        "--input",  frames_dir,
        "--output", sheet_path,
        "--size",   str(size),
        "--report", report_path,
    ], check=True)
    with open(report_path) as f:
        return json.load(f)

# Example usage
extract_frames("Videos/To Be Processed/hero_run.mp4", "work/extracted/hero/run", fps=12)
matte_frames("work/extracted/hero/run", "work/matted/hero/run", threshold=240)
report = build_sheet("work/matted/hero/run", "work/previews/hero_run_sheet.png", size=256)

print(f"Frames: {report['frame_count']}, Warnings: {report.get('warnings', [])}")
```

---

## JSON Report Schema

`animation_pipeline.py` writes a report like:

```json
{
  "source_dir":   "work/matted/hero/run",
  "output_sheet": "work/previews/hero_run_sheet.png",
  "cell_size":    256,
  "frame_count":  16,
  "sheet_width":  4096,
  "sheet_height": 256,
  "dropped_frames": [],
  "warnings": []
}
```

Check `warnings` and `dropped_frames` before promoting. Common warnings:
- Frame size mismatch (source frames not 256×256 — use `resize_sprites.py` first)
- Missing frames in sequence (gap in numbering)
- All-transparent frames

---

## AI-Assistant Skill Integration

The repo ships a self-contained skill at `skills/sprite-sheet-pipeline/SKILL.md`.

To give an AI assistant full context, point it at that file:

```
Use the sprite-sheet-pipeline skill at skills/sprite-sheet-pipeline/SKILL.md
```

Or install the folder in the assistant's skill directory. The skill routes tasks to:
- `docs/reference/PROMPTING_IMAGE_MODELS.md` — first poses, character refs
- `docs/reference/PROMPTING_VIDEO_MODELS.md` — Kling/image-to-video prompts
- `docs/QUICKSTART.md` — copy-paste processing commands
- Tools listed above

---

## Configuration Reference

All tools accept `--help`. Common shared flags:

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Source directory or file |
| `--output` | required | Destination directory or file |
| `--size` | `256` | Cell pixel dimension (square) |
| `--fps` | `12` | Frame extraction rate |
| `--threshold` | `240` | Matte brightness cutoff (0–255) |
| `--cols` | `8` | Contact sheet columns |
| `--report` | none | Path for JSON report output |

---

## Troubleshooting

### `ffmpeg: command not found`
FFmpeg is not on PATH. Install it per the setup section above; it is **not** included in `requirements.txt`.

### Frames look cropped or wrong size
Source frames must be square before building the sheet. Run `resize_sprites.py` on the frames directory first, or ensure the video source is square.

### All output pixels are transparent
The matte `--threshold` is too aggressive. Lower it (e.g. `200` instead of `240`), or skip matting if the source already has an alpha channel.

### Sheet has gaps / dropped frames
Check `dropped_frames` in the JSON report. Gaps usually mean FFmpeg dropped duplicate frames. Re-extract with a lower `--fps` or check the source video for corrupted segments.

### `sprite_viewer.html` shows no sheets
Run `python tools/build_sprite_gallery_manifest.py` after promotion. The viewer reads `sprite_gallery_manifest.js` from the same directory — both must be present.

### Import errors when running tools
Activate the virtual environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows), then retry.

---

## Asset Policy (what NOT to commit)

Keep out of the repo by default:
- Source videos (`Videos/`)
- Extracted / matted frames (`work/`)
- Preview PNGs and JSON reports
- Final sprite sheets and game art

Only commit tiny, intentional reference assets that are documented in code or tests.
```
