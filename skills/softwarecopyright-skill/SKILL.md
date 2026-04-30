```markdown
---
name: softwarecopyright-skill
description: Generate complete Chinese software copyright (软件著作权) application materials (DOCX/TXT) from real project source code using Codex AI agent skill
triggers:
  - generate software copyright materials
  - 生成软件著作权申请资料
  - create 软著 application documents
  - generate 软著材料 for my project
  - software copyright registration China
  - 生成软著申请材料
  - create Chinese software copyright docs
  - automate 软件著作权 filing materials
---

# Software Copyright Materials Skill (软件著作权申请材料生成器)

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Codex AI agent skill that reads your real project source code and automatically generates a complete set of Chinese software copyright (软件著作权) application materials — including the application form fields, operation manual (操作手册), and code excerpts (代码鉴别材料) — all as `.docx` and `.txt` files, locally, for free.

---

## What This Skill Does

- Reads your actual project source code (never fabricates code)
- Guides you through confirming key application fields interactively
- Generates the front-30-pages / back-30-pages code excerpt rule automatically
- Produces a business-aware operation manual (not a generic template)
- Outputs all files to `软件著作权申请资料/正式资料/` in your project directory
- Checks your environment before starting and stops for confirmation at every key step

**Output files:**
```
软件著作权申请资料/
├── 草稿/
├── 环境检查.md
├── 环境检查.json
└── 正式资料/
    ├── 申请表信息.txt              ← copy-paste into the official website form
    ├── <软件名称>_操作手册.docx   ← operation manual
    ├── <软件名称>-代码(前30页).docx
    └── <软件名称>-代码(后30页).docx
```

---

## Requirements

| Requirement | Status |
|---|---|
| **Codex** (AI agent runtime) | Required |
| **Python 3** | Required — drives all scripts |
| **Readable project source code** | Required — code must come from real project |
| **.NET SDK** | Optional — enables full OpenXML DOCX generation |
| **Chrome DevTools MCP** | Optional — for automated screenshot capture |
| **Codex Computer Use** | Optional — for desktop screenshot capture |

---

## Installation

### Global install (all projects)

```bash
# Clone the repository
git clone https://github.com/Fokkyp/SoftwareCopyright-Skill.git
cd SoftwareCopyright-Skill

# Install skill to Codex global skills directory
mkdir -p ~/.codex/skills
cp -R software-copyright-materials ~/.codex/skills/

# Verify installation
ls ~/.codex/skills/software-copyright-materials/SKILL.md
```

### Per-project install

```bash
# Replace <your-project-dir> with your actual project path
PROJECT_DIR="<your-project-dir>"

git clone https://github.com/Fokkyp/SoftwareCopyright-Skill.git
mkdir -p "$PROJECT_DIR/.codex/skills"
cp -R SoftwareCopyright-Skill/software-copyright-materials "$PROJECT_DIR/.codex/skills/"

# Verify
ls "$PROJECT_DIR/.codex/skills/software-copyright-materials/SKILL.md"
```

After installation, **restart Codex** or refresh your skill list.

---

## Basic Usage

Open your project in Codex, then say:

```
使用 software-copyright-materials 生成当前项目的软件著作权申请资料
```

Or in English:

```
Use the software-copyright-materials skill to generate Chinese software copyright application materials for this project
```

The skill will:
1. Run environment checks → output `环境检查.md` + `环境检查.json`
2. Analyze your project structure and business logic
3. **Stop and ask you to confirm** business description, key fields, and code file selection
4. Generate draft materials in `软件著作权申请资料/草稿/`
5. **Stop and ask you to confirm** the Markdown draft
6. Generate final `.docx` and `.txt` files in `软件著作权申请资料/正式资料/`

---

## Directory Structure of the Skill

```
software-copyright-materials/
├── SKILL.md          ← agent skill definition
├── agents/           ← agent prompt definitions and flow logic
├── references/       ← 软著申请规则参考文档
├── scripts/          ← Python scripts (run by Codex during generation)
│   ├── env_check.py
│   ├── analyze_project.py
│   ├── extract_code.py
│   ├── generate_draft.py
│   └── generate_docx.py
└── vendor/           ← vendored Python dependencies
```

---

## Scripts Reference

The skill's Python scripts can also be run standalone for debugging:

### Environment check

```bash
cd software-copyright-materials/scripts
python3 env_check.py --project-dir /path/to/your/project
```

Outputs:
- `软件著作权申请资料/环境检查.md` — human-readable status
- `软件著作权申请资料/环境检查.json` — machine-readable for agent use

Example `环境检查.json` output:
```json
{
  "python3": true,
  "dotnet_sdk": false,
  "basic_docx": true,
  "full_docx_openxml": false,
  "output_dir": "/your/project/软件著作权申请资料/正式资料/",
  "warnings": [".NET SDK not found — will use basic DOCX fallback"]
}
```

### Project analysis

```bash
python3 analyze_project.py --project-dir /path/to/your/project
```

Scans source files, detects language composition, estimates line count for `源程序量` field.

### Code extraction (前30页/后30页)

```bash
python3 extract_code.py \
  --project-dir /path/to/your/project \
  --output-dir ./软件著作权申请资料/正式资料/ \
  --software-name "我的软件" \
  --files "src/main.py,src/utils.py,src/models.py"
```

Rules applied automatically:
- **≥ 60 pages of code**: generates `代码(前30页).docx` + `代码(后30页).docx`
- **< 60 pages of code**: generates a single full code material file

### Generate DOCX files

```bash
python3 generate_docx.py \
  --draft-dir ./软件著作权申请资料/草稿/ \
  --output-dir ./软件著作权申请资料/正式资料/ \
  --software-name "我的软件" \
  --version "V1.0"
```

---

## Key Application Fields (申请表信息)

The skill collects and validates these fields, then writes them to `申请表信息.txt`:

| Field | Chinese | Example |
|---|---|---|
| Software name | 软件名称 | 项目管理系统 V1.0 |
| Version | 版本号 | V1.0 |
| Copyright holder | 著作权人 | 张三 / 某某科技有限公司 |
| Development completion date | 开发完成日期 | 2025年12月31日 |
| First publication date | 首次发表日期 | 2026年1月1日（未发表则填写同上） |
| Development environment | 开发环境 | Windows 11, Python 3.11, VS Code |
| Runtime environment | 运行环境 | Windows 10及以上 / Linux / macOS |
| Source lines | 源程序量 | 约8500行 |
| Development method | 开发方式 | 独立开发 |
| Function description | 功能说明 | ≤ 500 characters summary |

---

## Code Examples: Using Scripts Directly in Python

```python
import subprocess
import json
from pathlib import Path

PROJECT = Path("/path/to/your/project")
SKILL_SCRIPTS = Path.home() / ".codex/skills/software-copyright-materials/scripts"

# Step 1: Environment check
result = subprocess.run(
    ["python3", str(SKILL_SCRIPTS / "env_check.py"), "--project-dir", str(PROJECT)],
    capture_output=True, text=True
)
env_report = json.loads((PROJECT / "软件著作权申请资料/环境检查.json").read_text())
print("Basic DOCX available:", env_report["basic_docx"])
print("Full OpenXML available:", env_report["full_docx_openxml"])

# Step 2: Analyze project
subprocess.run(
    ["python3", str(SKILL_SCRIPTS / "analyze_project.py"), "--project-dir", str(PROJECT)],
)

# Step 3: Extract code (after confirming file selection with user)
selected_files = "src/main.py,src/api.py,src/models.py"
subprocess.run([
    "python3", str(SKILL_SCRIPTS / "extract_code.py"),
    "--project-dir", str(PROJECT),
    "--output-dir", str(PROJECT / "软件著作权申请资料/正式资料/"),
    "--software-name", "我的软件",
    "--files", selected_files,
])

# Step 4: Generate DOCX from confirmed draft
subprocess.run([
    "python3", str(SKILL_SCRIPTS / "generate_docx.py"),
    "--draft-dir", str(PROJECT / "软件著作权申请资料/草稿/"),
    "--output-dir", str(PROJECT / "软件著作权申请资料/正式资料/"),
    "--software-name", "我的软件",
    "--version", "V1.0",
])
```

---

## Screenshot Handling

During operation manual generation, the skill will ask you to choose:

1. **Chrome DevTools MCP** — automated web screenshot capture
2. **Codex Computer Use** — desktop GUI screenshot capture  
3. **User-provided screenshots** — manually place images in the designated folder
4. **Skip screenshots** — operation manual will include visible placeholder slots

If you skip, the DOCX will contain clearly marked `[截图预留位置]` placeholders you can fill in later.

---

## Submitting to the Official System

After generating materials:

1. Open `软件著作权申请资料/正式资料/申请表信息.txt`
2. Log in at https://register.ccopyright.com.cn/login.html
3. Start a new 计算机软件著作权登记申请
4. Copy-paste fields from `申请表信息.txt` into the online form
5. Export `.docx` files to PDF (Word → Save As PDF, or WPS PDF export)
6. Upload PDFs as required by the current official page

**Note:** `申请表信息.txt` is a fill-assist file — it is not uploaded directly. Only the PDF exports of the `.docx` files are uploaded.

Legal reference: [《计算机软件著作权登记办法》](https://www.gov.cn/zhengce/2002-02/20/content_5724627.htm)

---

## Troubleshooting

### "DOCX generation failed" / `.docx` file is corrupted

The full OpenXML DOCX path requires .NET SDK. Install it:

```bash
# macOS
brew install --cask dotnet-sdk

# Ubuntu/Debian
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0

# Verify
dotnet --version
```

Then re-run environment check and regenerate.

### "No source files found"

Make sure Codex has opened the correct project directory, not the skill repository itself. The skill analyzes the **current working project**, not `SoftwareCopyright-Skill/`.

### Generated code material is all from one file

When the skill stops to ask you to confirm code file selection, explicitly list the files you want included — the skill will not auto-select files without your confirmation.

### Operation manual reads as generic / doesn't match my product

At the business confirmation step, provide detailed descriptions of your software's actual screens, user flows, and features. The more specific your input, the more accurate the generated manual will be.

### Skill not found after installation

```bash
# Confirm correct path
ls ~/.codex/skills/software-copyright-materials/SKILL.md

# If missing, re-run install
cp -R /path/to/SoftwareCopyright-Skill/software-copyright-materials ~/.codex/skills/
```

Then restart Codex completely.

---

## What the Skill Will NOT Do

- **Generate fake source code** — all code material must come from your real project
- **Auto-submit to the official website** — you must review and submit manually
- **Guarantee approval** — always verify generated materials match your actual project before submission
- **Replace legal advice** — consult a lawyer for complex copyright ownership questions (joint development, employer-employee, commissioned work)
```
