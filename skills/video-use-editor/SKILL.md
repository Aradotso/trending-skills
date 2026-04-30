```markdown
---
name: video-use-editor
description: Edit videos with AI coding agents using Claude Code, ffmpeg, and ElevenLabs transcription
triggers:
  - edit this video
  - cut out filler words
  - add subtitles to my video
  - color grade my footage
  - make a highlight reel
  - trim my talking head video
  - assemble these clips into a final video
  - add animations to my video
---

# video-use: AI Video Editing Skill

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

**video-use** lets AI coding agents edit video like a human editor — reading transcripts and timelines as structured text rather than processing raw frames. Drop raw footage in a folder, describe the edit, get `final.mp4` back.

---

## What video-use does

- **Cuts filler words** (`umm`, `uh`, false starts, dead air) using word-level timestamps
- **Color grades** every segment with ffmpeg filter chains (cinematic warm, neutral punch, or custom)
- **Burns subtitles** — 2-word UPPERCASE chunks by default, fully customizable
- **Generates animation overlays** via Manim, Remotion, or PIL in parallel sub-agents
- **Self-evaluates** rendered output at every cut boundary before showing you anything
- **Persists session memory** in `project.md` so future sessions pick up where you left off

The LLM never watches the video. It reads it through:
1. **Audio transcript** (ElevenLabs Scribe) — word-level timestamps, speaker diarization, audio events
2. **Visual composite on demand** — filmstrip + waveform + word labels PNG for ambiguous decisions only

---

## Installation

### Automated (paste into any coding agent)

```text
Set up https://github.com/browser-use/video-use for me.

Read install.md first to install this repo, wire up ffmpeg, register the skill with whichever agent you're running under, and set up the ElevenLabs API key — ask me to paste it when you need it. Then read SKILL.md for daily usage, and always read helpers/ because that's where the editing scripts live. After install, don't transcribe anything on your own — just tell me it's ready and wait for me to drop footage into a folder.
```

### Manual

```bash
# Clone and symlink into your agent's skills directory
git clone https://github.com/browser-use/video-use ~/Developer/video-use
ln -sfn ~/Developer/video-use ~/.claude/skills/video-use        # Claude Code
# ln -sfn ~/Developer/video-use ~/.codex/skills/video-use       # Codex

# Install Python dependencies
cd ~/Developer/video-use
uv sync                         # or: pip install -e .

# Install system dependencies
brew install ffmpeg              # required
brew install yt-dlp             # optional, for online sources

# Set up environment
cp .env.example .env
# Edit .env and add: ELEVENLABS_API_KEY=your_key_here
```

Get an ElevenLabs API key at [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys).

---

## Environment Configuration

```bash
# .env file
ELEVENLABS_API_KEY=your_key_here    # Required for transcription
```

---

## Starting an editing session

```bash
cd /path/to/your/raw/footage
claude    # or: codex, hermes, etc.
```

Then in the session, describe what you want:

```
edit these into a launch video
```

```
cut out all the umms and uhs, keep takes under 30 seconds
```

```
make a 60-second highlight reel from the best moments
```

The agent will:
1. Inventory source files
2. Propose an editing strategy
3. Wait for your approval
4. Produce `edit/final.mp4` next to your sources

All outputs live in `<videos_dir>/edit/` — the skill directory stays clean.

---

## Pipeline

```
Transcribe ──> Pack ──> LLM Reasons ──> EDL ──> Render ──> Self-Eval
                                                              │
                                                              └─ issue? fix + re-render (max 3)
```

### Step 1: Transcription

ElevenLabs Scribe produces word-level timestamps per source file, packed into `takes_packed.md`:

```markdown
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
  [007.10-009.80] S0 Uh — (pause) — let me show you what I mean.
```

~12KB of text replaces 45M tokens of frame analysis.

### Step 2: Edit Decision List (EDL)

The agent produces a structured EDL before touching any files:

```python
# Example EDL structure the agent reasons over
edl = [
    {
        "source": "C0103.mp4",
        "in":  2.52,
        "out": 5.36,
        "color_grade": "warm_cinematic",
        "audio_fade_ms": 30,
    },
    {
        "source": "C0103.mp4",
        "in":  6.08,
        "out": 6.74,
        "color_grade": "warm_cinematic",
        "audio_fade_ms": 30,
    },
]
```

### Step 3: Render via helpers/

The `helpers/` directory contains the ffmpeg scripts the agent calls. Always read this directory — it's where editing logic lives.

---

## Key helpers and scripts

### timeline_view — visual composite on demand

Called only at decision points (ambiguous pauses, retake comparisons, cut sanity checks):

```python
# helpers/timeline_view.py
# Produces: filmstrip + speaker track + waveform + word labels PNG
# Args: source file, start_time, end_time
python helpers/timeline_view.py C0103.mp4 2.0 10.0
# -> edit/timeline_C0103_2.0-10.0.png
```

### Cutting with ffmpeg (what the agent generates)

```bash
# Single segment cut with color grade and audio fade
ffmpeg -i C0103.mp4 \
  -ss 2.52 -to 5.36 \
  -vf "curves=vintage,fade=t=out:st=2.8:d=0.03:alpha=0" \
  -af "afade=t=in:st=0:d=0.03,afade=t=out:st=2.8:d=0.03" \
  -c:v libx264 -c:a aac \
  edit/seg_001.mp4
```

### Concatenating segments

```bash
# helpers/concat.py generates this automatically
ffmpeg -f concat -safe 0 -i edit/segments.txt -c copy edit/final_raw.mp4
```

### Burning subtitles

```bash
# 2-word UPPERCASE chunks, customizable via --style
python helpers/burn_subtitles.py \
  --input edit/final_raw.mp4 \
  --transcript edit/transcript.json \
  --style uppercase_2word \
  --output edit/final.mp4
```

### Color grade presets

```python
# helpers/color_grades.py
GRADES = {
    "warm_cinematic": "curves=vintage,colorbalance=rs=0.1:gs=0:bs=-0.1",
    "neutral_punch":  "eq=contrast=1.1:saturation=1.05:brightness=0.02",
    "cool_clean":     "colorbalance=rs=-0.05:gs=0:bs=0.1,curves=lighter",
    "raw":            None,   # pass-through
}
```

---

## Real code examples

### Transcribing a source file

```python
import os
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

with open("C0103.mp4", "rb") as f:
    transcript = client.speech_to_text.convert(
        file=f,
        model_id="scribe_v1",
        diarize=True,
        timestamps_granularity="word",
    )

# Word-level output
for word in transcript.words:
    print(f"[{word.start:.2f}-{word.end:.2f}] {word.text}")
```

### Finding filler words to cut

```python
FILLERS = {"umm", "uh", "um", "uhh", "hmm", "like", "you know"}

def find_filler_cuts(transcript_words):
    cuts = []
    for i, word in enumerate(transcript_words):
        if word.text.lower().strip(",.") in FILLERS:
            # Merge with surrounding silence if gap < 0.3s
            cut_start = word.start
            cut_end = word.end
            if i + 1 < len(transcript_words):
                gap = transcript_words[i + 1].start - word.end
                if gap < 0.3:
                    cut_end = transcript_words[i + 1].start
            cuts.append({"start": cut_start, "end": cut_end, "reason": word.text})
    return cuts
```

### Building a concat list from EDL

```python
import subprocess

def render_segment(source, t_in, t_out, grade, index, output_dir="edit"):
    vf = grade or "null"
    out_path = f"{output_dir}/seg_{index:03d}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", source,
        "-ss", str(t_in),
        "-to", str(t_out),
        "-vf", vf,
        "-af", f"afade=t=in:st=0:d=0.03,afade=t=out:st={t_out - t_in - 0.03:.3f}:d=0.03",
        "-c:v", "libx264", "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path

def build_final(edl, output_dir="edit"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    segment_paths = []
    for i, seg in enumerate(edl):
        path = render_segment(
            source=seg["source"],
            t_in=seg["in"],
            t_out=seg["out"],
            grade=seg.get("color_grade"),
            index=i,
            output_dir=output_dir,
        )
        segment_paths.append(path)

    # Write concat list
    concat_file = f"{output_dir}/segments.txt"
    with open(concat_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    # Concatenate
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        f"{output_dir}/final.mp4",
    ], check=True)
```

### Animation overlay with PIL (simple lower-third)

```python
from PIL import Image, ImageDraw, ImageFont
import subprocess

def make_lower_third(text, width=1920, height=1080, duration=3.0, fps=30, output="edit/lower_third.mp4"):
    frames_dir = "edit/lower_third_frames"
    os.makedirs(frames_dir, exist_ok=True)

    total_frames = int(duration * fps)
    for i in range(total_frames):
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Fade in first 15 frames, fade out last 15
        alpha = min(255, i * 17, (total_frames - i) * 17)
        draw.rectangle([0, height - 120, width, height], fill=(0, 0, 0, int(alpha * 0.7)))
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        draw.text((80, height - 90), text.upper(), font=font, fill=(255, 255, 255, alpha))
        img.save(f"{frames_dir}/frame_{i:05d}.png")

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuva420p",
        output,
    ], check=True)
```

---

## Session memory: project.md

The agent writes and reads `project.md` in your footage directory to persist state:

```markdown
# Project: Launch Video

## Sources
- C0101.mp4 — opening hook attempt 1 (weak, discard)
- C0102.mp4 — opening hook attempt 2 (strong, use [2.5-8.1])
- C0103.mp4 — main demo (use [2.52-5.36], [6.08-6.74])
- C0104.mp4 — closing CTA (use full take)

## Style decisions
- Color grade: warm_cinematic
- Subtitles: 2-word UPPERCASE, white, 80px Helvetica
- Target length: 90 seconds

## Completed
- [x] Transcription
- [x] EDL approved by user
- [x] Render pass 1
- [ ] Add lower-thirds for speaker names
```

At the start of each session, the agent reads this file before doing anything else.

---

## Self-evaluation loop

After every render, the agent runs `timeline_view` on the output at each cut boundary:

```python
def self_eval_cuts(final_path, edl, output_dir="edit"):
    issues = []
    for i, seg in enumerate(edl[:-1]):
        # Check the join between seg i and seg i+1
        join_time = sum(s["out"] - s["in"] for s in edl[:i+1])
        png = f"{output_dir}/eval_cut_{i:03d}.png"
        subprocess.run([
            "python", "helpers/timeline_view.py",
            final_path,
            str(join_time - 0.5),
            str(join_time + 0.5),
            "--output", png,
        ], check=True)
        # Agent inspects PNG for: visual jump, audio pop, hidden subtitle
        # If issue found, adjusts EDL and re-renders (max 3 attempts)
    return issues
```

---

## Common patterns

### Talking head / interview

```
cut out all filler words, color grade warm cinematic,
burn 2-word uppercase subtitles, target 3 minutes
```

### Product demo

```
keep only the segments where I'm actually showing the product,
cut all setup/mistakes, add lower-thirds with feature names
```

### Travel montage

```
pick the best 5 seconds from each clip,
assemble in sequence, no subtitles, color grade cool clean
```

### Tutorial with screen recording

```
sync the talking head and screen recording tracks,
cut to screen when I say "here" or "this", keep audio from talking head
```

---

## Troubleshooting

**`ELEVENLABS_API_KEY` not found**
```bash
# Verify .env is in the video-use directory (not your footage folder)
cat ~/Developer/video-use/.env
# Should show: ELEVENLABS_API_KEY=el_...
```

**ffmpeg not found**
```bash
brew install ffmpeg
# Verify: ffmpeg -version
```

**"No module named elevenlabs"**
```bash
cd ~/Developer/video-use
uv sync
# or: pip install -e .
```

**Skill not loading in Claude Code**
```bash
ls -la ~/.claude/skills/
# Should show: video-use -> /Users/you/Developer/video-use
# If missing:
ln -sfn ~/Developer/video-use ~/.claude/skills/video-use
```

**Audio pop at cut points**
The 30ms fade (`afade`) must be applied to every segment. Check that `render_segment()` is using the `-af` flag with both fade-in and fade-out. Minimum segment duration for clean fades is ~100ms.

**Subtitles cut off at frame edge**
Increase padding in `burn_subtitles.py` — the `MarginV` and `MarginH` ASS style parameters control placement.

**Render produces black frames**
Usually a seek precision issue. Use `-ss` before `-i` (input seek) rather than after for long seeks; use `-ss` after `-i` for sub-second precision near cut points.

---

## Design principles

1. **Text + on-demand visuals** — no frame-dumping; the transcript is the editing surface
2. **Audio is primary, visuals follow** — cuts come from speech boundaries and silence gaps
3. **Ask → confirm → execute → self-eval → persist** — never touch the cut without strategy approval
4. **Zero assumptions about content type** — look, ask, then edit
5. **30ms audio fades at every cut** — non-negotiable production correctness

---

## Key paths reference

| Path | Purpose |
|------|---------|
| `~/.claude/skills/video-use/` | Skill symlink (Claude Code) |
| `~/Developer/video-use/helpers/` | ffmpeg + editing scripts — always read this |
| `~/Developer/video-use/.env` | API keys |
| `<footage_dir>/edit/` | All outputs (segments, final, eval PNGs) |
| `<footage_dir>/project.md` | Session memory — read at start of every session |
| `<footage_dir>/edit/takes_packed.md` | Packed transcript — primary LLM reading surface |
```
