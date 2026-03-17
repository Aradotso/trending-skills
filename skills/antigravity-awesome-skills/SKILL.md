```markdown
---
name: antigravity-awesome-skills
description: Install, browse, and use the 1,265+ agentic skills library for Claude Code, Gemini CLI, Cursor, Codex CLI, Kiro, and other AI coding assistants.
triggers:
  - install antigravity skills
  - add agentic skills to my AI assistant
  - use antigravity awesome skills
  - set up skills for Claude Code
  - browse the skills library
  - install skills for Cursor or Gemini CLI
  - how do I use SKILL.md files
  - add reusable skills to my coding agent
---

# Antigravity Awesome Skills

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

**Antigravity Awesome Skills** is a curated library of 1,265+ universal agentic `SKILL.md` files that teach AI coding assistants (Claude Code, Gemini CLI, Codex CLI, Cursor, Kiro, OpenCode, AdaL, GitHub Copilot) how to perform specific tasks — deployments, security reviews, brainstorming, testing, infrastructure, and much more — consistently and reliably.

Each skill is a structured Markdown file a developer installs once, then invokes by name in any supported agent chat.

---

## Installation

### Recommended: npx (auto-detects tool)

```bash
# Install to default Antigravity global path (~/.gemini/antigravity/skills)
npx antigravity-awesome-skills

# Install for a specific tool
npx antigravity-awesome-skills --claude      # ~/.claude/skills/
npx antigravity-awesome-skills --gemini      # ~/.gemini/skills/
npx antigravity-awesome-skills --codex       # ~/.codex/skills/
npx antigravity-awesome-skills --cursor      # .cursor/skills/  (workspace)
npx antigravity-awesome-skills --antigravity # ~/.gemini/antigravity/skills/
npx antigravity-awesome-skills --kiro        # ~/.kiro/skills/

# Install to a custom path
npx antigravity-awesome-skills --path ./my-skills
npx antigravity-awesome-skills --path .agents/skills   # OpenCode
npx antigravity-awesome-skills --path .adal/skills     # AdaL CLI
```

### Claude Code Plugin Marketplace

```text
/plugin marketplace add sickn33/antigravity-awesome-skills
/plugin install antigravity-awesome-skills
```

### Manual clone

```bash
git clone https://github.com/sickn33/antigravity-awesome-skills.git
# Then copy or symlink the skills/ directory to the correct path for your tool
cp -r antigravity-awesome-skills/skills/ ~/.claude/skills/
```

### Verify installation

```bash
# Antigravity / Gemini CLI default path
test -d ~/.gemini/antigravity/skills && echo "✅ Skills installed" || echo "❌ Not found"

# Claude Code
test -d ~/.claude/skills && echo "✅ Skills installed" || echo "❌ Not found"

# Count installed skills
ls ~/.gemini/antigravity/skills/*.md 2>/dev/null | wc -l
```

---

## Skill Installation Paths by Tool

| Tool | Global Path | Workspace Path |
|---|---|---|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| Gemini CLI | `~/.gemini/skills/` | `.gemini/skills/` |
| Codex CLI | `~/.codex/skills/` | `.codex/skills/` |
| Kiro CLI/IDE | `~/.kiro/skills/` | `.kiro/skills/` |
| Antigravity | `~/.gemini/antigravity/skills/` | `.agent/skills/` |
| Cursor | `.cursor/skills/` | `.cursor/skills/` |
| OpenCode | `.agents/skills/` | `.agents/skills/` |
| AdaL CLI | `.adal/skills/` | `.adal/skills/` |

---

## Invoking Skills

### Claude Code
```text
>> /brainstorming help me plan a SaaS MVP
>> /lint-and-validate run on src/api.py
>> /security-review check this authentication flow
```

### Gemini CLI
```text
Use brainstorming to plan a SaaS MVP.
Use security-review on this file.
```

### Cursor (in Chat panel)
```text
@brainstorming help me design the data model
@lint-and-validate check this component
```

### Codex CLI
```text
Use brainstorming to plan a SaaS MVP.
```

### OpenCode
```bash
opencode run @brainstorming help me plan a feature
opencode run @security-review src/auth.py
```

### Antigravity IDE
```text
Use @brainstorming to plan a SaaS MVP.
```

---

## SKILL.md File Format

Each skill follows a standard structure. You can create your own:

```markdown
---
name: my-custom-skill
description: One-line description of what this skill does.
triggers:
  - natural phrase 1
  - natural phrase 2
  - natural phrase 3
---

# My Custom Skill

## Purpose
What this skill helps the agent accomplish.

## Steps
1. First action
2. Second action
3. Third action

## Examples
...

## Notes
...
```

### Python helper to create a skill programmatically

```python
import os
from pathlib import Path

def create_skill(name: str, description: str, triggers: list[str], body: str, install_path: str = None) -> Path:
    """
    Create a SKILL.md file and optionally install it to a skills directory.
    
    Args:
        name: kebab-case skill name
        description: one-line description
        triggers: list of natural-language trigger phrases
        body: Markdown content (headings, steps, examples)
        install_path: directory to write the file; defaults to ~/.claude/skills/
    
    Returns:
        Path to the created skill file
    """
    if install_path is None:
        install_path = Path.home() / ".claude" / "skills"
    
    skills_dir = Path(install_path)
    skills_dir.mkdir(parents=True, exist_ok=True)
    
    trigger_lines = "\n".join(f"  - {t}" for t in triggers)
    frontmatter = f"---\nname: {name}\ndescription: {description}\ntriggers:\n{trigger_lines}\n---\n\n"
    
    skill_file = skills_dir / f"{name}.md"
    skill_file.write_text(frontmatter + f"# {name.replace('-', ' ').title()}\n\n" + body)
    
    print(f"✅ Skill created: {skill_file}")
    return skill_file


# Example usage
create_skill(
    name="docker-deploy",
    description="Build and deploy a Docker container to a remote host.",
    triggers=[
        "deploy with docker",
        "build and push docker image",
        "run docker deployment",
    ],
    body="""## Steps
1. Build the image: `docker build -t $IMAGE_NAME .`
2. Tag it: `docker tag $IMAGE_NAME $REGISTRY/$IMAGE_NAME:latest`
3. Push: `docker push $REGISTRY/$IMAGE_NAME:latest`
4. SSH to host and pull: `ssh $HOST 'docker pull $REGISTRY/$IMAGE_NAME:latest && docker compose up -d'`

## Environment Variables
- `IMAGE_NAME` — name of the Docker image
- `REGISTRY` — container registry URL (e.g. ghcr.io/myorg)
- `HOST` — deployment target hostname
""",
    install_path=Path.home() / ".claude" / "skills",
)
```

---

## Browsing & Searching Skills

### Web App (local)

```bash
cd apps/web-app
npm install
npm run dev
# Open http://localhost:3000
```

### CLI search with grep

```bash
# Search skill names
ls ~/.gemini/antigravity/skills/ | grep "security"

# Search skill descriptions (frontmatter)
grep -l "docker" ~/.gemini/antigravity/skills/*.md

# Full-text search across all skills
grep -r "CloudFormation" ~/.gemini/antigravity/skills/
```

### Python: parse and list all installed skills

```python
import re
from pathlib import Path

def list_skills(skills_dir: str = None) -> list[dict]:
    """
    Parse all SKILL.md files in a directory and return their metadata.
    """
    if skills_dir is None:
        skills_dir = Path.home() / ".gemini" / "antigravity" / "skills"
    
    skills_dir = Path(skills_dir)
    skills = []
    
    frontmatter_pattern = re.compile(
        r"^---\s*\n(.*?)\n---", re.DOTALL
    )
    
    for skill_file in sorted(skills_dir.glob("*.md")):
        content = skill_file.read_text(encoding="utf-8")
        match = frontmatter_pattern.match(content)
        
        if match:
            fm = match.group(1)
            name_match = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
            desc_match = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
            skills.append({
                "file": skill_file.name,
                "name": name_match.group(1).strip() if name_match else skill_file.stem,
                "description": desc_match.group(1).strip() if desc_match else "",
            })
    
    return skills


# Print all installed skills
for skill in list_skills():
    print(f"  {skill['name']:<40} {skill['description']}")
```

---

## Popular Skill Categories

| Category | Example Skills |
|---|---|
| Planning | `brainstorming`, `feature-planning`, `architecture-review` |
| Coding | `code-review`, `refactor`, `lint-and-validate` |
| Testing | `test-generation`, `coverage-report`, `e2e-testing` |
| Security | `security-review`, `dependency-audit`, `secrets-scan` |
| DevOps | `docker-deploy`, `terraform-plan`, `ci-pipeline` |
| Docs | `docstring-generation`, `readme-writer`, `changelog` |
| AWS | `cloudformation`, `lambda-deploy`, `s3-sync` |
| Data | `sql-optimize`, `data-migration`, `schema-review` |

---

## Starter Bundles

Bundles are documented in [`docs/users/bundles.md`](https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/users/bundles.md).

| Bundle | Skills Included | Best For |
|---|---|---|
| **Essentials** | brainstorming, code-review, lint-and-validate, security-review | Everyone |
| **Web Wizard** | react-components, api-design, css-audit, seo-check | Web developers |
| **Security Engineer** | security-review, secrets-scan, dependency-audit, threat-model | Security teams |
| **DevOps Pro** | docker-deploy, terraform-plan, ci-pipeline, k8s-deploy | Platform engineers |
| **Data Engineer** | sql-optimize, data-migration, schema-review, etl-pipeline | Data teams |

---

## Contributing a Skill

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/antigravity-awesome-skills.git
cd antigravity-awesome-skills

# 2. Create your skill file
cp skills/_template.md skills/my-new-skill.md
# Edit the file following the SKILL.md format

# 3. Validate locally (Python)
python tools/validate_skill.py skills/my-new-skill.md

# 4. Run the skill security scan
python tools/skill_security_scan.py skills/my-new-skill.md

# 5. Open a pull request
git checkout -b feat/my-new-skill
git add skills/my-new-skill.md
git commit -m "feat: add my-new-skill"
git push origin feat/my-new-skill
```

### Skill validation helper (Python)

```python
import re
import sys
from pathlib import Path

REQUIRED_FRONTMATTER_KEYS = {"name", "description", "triggers"}
HIGH_RISK_PATTERNS = [
    r"curl\s+.*\|\s*bash",
    r"wget\s+.*\|\s*sh",
    r"irm\s+.*\|\s*iex",
]

def validate_skill(filepath: str) -> bool:
    content = Path(filepath).read_text(encoding="utf-8")
    
    fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        print("❌ Missing YAML frontmatter")
        return False
    
    fm = fm_match.group(1)
    missing = []
    for key in REQUIRED_FRONTMATTER_KEYS:
        if not re.search(rf"^{key}:", fm, re.MULTILINE):
            missing.append(key)
    if missing:
        print(f"❌ Missing frontmatter keys: {missing}")
        return False
    
    warnings = []
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            warnings.append(f"⚠️  High-risk pattern found: {pattern}")
    
    for w in warnings:
        print(w)
    
    print(f"✅ Skill valid: {filepath}")
    return True


if __name__ == "__main__":
    ok = validate_skill(sys.argv[1])
    sys.exit(0 if ok else 1)
```

---

## Troubleshooting

### Skills not recognized by my agent

```bash
# Confirm files exist in the right path for your tool
ls ~/.claude/skills/          # Claude Code
ls ~/.gemini/skills/          # Gemini CLI
ls .cursor/skills/            # Cursor (workspace)
ls ~/.gemini/antigravity/skills/  # Antigravity

# Re-run the installer targeting your tool explicitly
npx antigravity-awesome-skills --claude
```

### npx installer fails or hangs

```bash
# Clear npx cache and retry
npx clear-npx-cache
npx antigravity-awesome-skills

# Alternatively, clone and copy manually
git clone https://github.com/sickn33/antigravity-awesome-skills.git /tmp/aas
cp -r /tmp/aas/skills/ ~/.claude/skills/
```

### Skill invocation does nothing in Claude Code

- Ensure you use the `>> /skill-name` prefix, not `@skill-name` (that's Cursor syntax).
- Check that the skill file name matches exactly: `brainstorming.md` → `/brainstorming`.
- Restart the Claude Code session after installing new skills.

### Wrong install path on Windows

```powershell
# Windows: use $env:USERPROFILE instead of ~
$skillsPath = "$env:USERPROFILE\.claude\skills"
New-Item -ItemType Directory -Force -Path $skillsPath
npx antigravity-awesome-skills --path $skillsPath
```

### Skill file has encoding issues

```python
# Re-save a skill file as UTF-8
from pathlib import Path

skill = Path("skills/my-skill.md")
content = skill.read_bytes().decode("utf-8", errors="replace")
skill.write_text(content, encoding="utf-8")
```

---

## Key Links

- **Repository**: https://github.com/sickn33/antigravity-awesome-skills
- **Usage Guide**: https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/users/usage.md
- **Bundles**: https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/users/bundles.md
- **Workflows**: https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/users/workflows.md
- **Full Catalog**: https://github.com/sickn33/antigravity-awesome-skills/blob/main/CATALOG.md
- **Web App**: `apps/web-app` (run locally with `npm run dev`)
- **npm package**: `antigravity-awesome-skills`
```
