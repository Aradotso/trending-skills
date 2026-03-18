```markdown
---
name: obsidian-skills-agent
description: Teach AI coding agents to work with Obsidian vaults using Markdown, Bases, JSON Canvas, and the Obsidian CLI
triggers:
  - work with obsidian vault
  - create obsidian notes
  - edit obsidian markdown
  - use obsidian bases
  - create json canvas
  - interact with obsidian cli
  - add obsidian agent skills
  - teach agent obsidian syntax
---

# Obsidian Skills for AI Agents

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

`obsidian-skills` is a collection of agent skills that teach AI coding agents (Claude Code, Codex CLI, OpenCode, Cursor, etc.) how to work with Obsidian vaults. It covers Obsidian Flavored Markdown, Bases, JSON Canvas, and the Obsidian CLI — following the [Agent Skills specification](https://agentskills.io/specification).

---

## Installation

### Via Marketplace (if supported)

```
/plugin marketplace add kepano/obsidian-skills
/plugin install obsidian@obsidian-skills
```

### Via npx

```bash
npx skills add git@github.com:kepano/obsidian-skills.git
```

### Claude Code (Manual)

Copy the repo contents into a `/.claude` folder at the root of your Obsidian vault:

```bash
git clone https://github.com/kepano/obsidian-skills.git /path/to/vault/.claude
```

### Codex CLI (Manual)

Copy the `skills/` directory into your Codex skills path:

```bash
git clone https://github.com/kepano/obsidian-skills.git /tmp/obsidian-skills
cp -r /tmp/obsidian-skills/skills/* ~/.codex/skills/
```

### OpenCode (Manual)

Clone the full repo into the OpenCode skills directory:

```bash
git clone https://github.com/kepano/obsidian-skills.git ~/.opencode/skills/obsidian-skills
```

> Do **not** copy only the inner `skills/` folder. Clone the full repo so the path resolves to:
> `~/.opencode/skills/obsidian-skills/skills/<skill-name>/SKILL.md`
>
> OpenCode auto-discovers all `SKILL.md` files under `~/.opencode/skills/`. Restart OpenCode after cloning.

---

## Available Skills

| Skill | File | Description |
|---|---|---|
| `obsidian-markdown` | `skills/obsidian-markdown/SKILL.md` | Obsidian Flavored Markdown with wikilinks, embeds, callouts, properties |
| `obsidian-bases` | `skills/obsidian-bases/SKILL.md` | Obsidian Bases (`.base`) with views, filters, formulas, summaries |
| `json-canvas` | `skills/json-canvas/SKILL.md` | JSON Canvas (`.canvas`) with nodes, edges, groups, connections |
| `obsidian-cli` | `skills/obsidian-cli/SKILL.md` | Obsidian CLI for vault, plugin, and theme interaction |
| `defuddle` | `skills/defuddle/SKILL.md` | Extract clean markdown from web pages using Defuddle |

---

## Obsidian Flavored Markdown

Obsidian uses a superset of standard Markdown. Key syntax additions:

### Properties (Frontmatter)

```markdown
---
title: My Note
tags:
  - research
  - ai
date: 2026-03-01
status: draft
aliases:
  - My Research Note
---
```

### Wikilinks

```markdown
[[Note Name]]
[[Note Name|Display Text]]
[[Folder/Note Name]]
[[Note Name#Heading]]
```

### Embeds

```markdown
![[Note Name]]
![[image.png]]
![[Note Name#Section]]
![[Note Name#^block-id]]
```

### Callouts

```markdown
> [!note]
> This is a note callout.

> [!warning] Custom Title
> This is a warning with a custom title.

> [!tip]- Collapsible Tip
> This callout is collapsible by default (closed).

> [!info]+ Expanded Collapsible
> This callout is collapsible and starts open.
```

Callout types: `note`, `abstract`, `info`, `tip`, `success`, `question`, `warning`, `failure`, `danger`, `bug`, `example`, `quote`

### Block IDs

```markdown
This paragraph has a block ID. ^my-block-id

- List item with block ID ^list-block
```

Reference with: `[[Note#^my-block-id]]`

### Tags

```markdown
#tag
#nested/tag
#multi-word-tag
```

---

## Obsidian Bases (`.base` files)

Bases are database-like views over your vault's markdown files, using `.base` file format.

### Basic Base File

```yaml
# my-projects.base
view: table
filter:
  type: eq
  field: type
  value: project
fields:
  - name
  - status
  - due
  - priority
sort:
  - field: due
    order: asc
```

### Views

```yaml
view: table    # Spreadsheet-style
view: list     # Simple list
view: board    # Kanban-style (requires groupBy)
view: gallery  # Card/image grid
view: calendar # Calendar (requires date field)
```

### Filters

```yaml
# Single filter
filter:
  type: eq
  field: status
  value: active

# Compound filter
filter:
  type: and
  conditions:
    - type: eq
      field: type
      value: project
    - type: neq
      field: status
      value: archived
```

Filter operators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `contains`, `not_contains`, `starts_with`, `ends_with`, `is_empty`, `is_not_empty`

### Formulas

```yaml
fields:
  - name: days_remaining
    formula: "dateDiff(now(), due, 'days')"
  - name: full_name
    formula: "concat(first_name, ' ', last_name)"
```

### Summaries

```yaml
summary:
  - field: budget
    type: sum
  - field: status
    type: count_by_value
```

---

## JSON Canvas (`.canvas` files)

JSON Canvas is an open format for infinite canvas data. Files use the `.canvas` extension.

### Minimal Canvas

```json
{
  "nodes": [
    {
      "id": "node1",
      "type": "text",
      "x": 0,
      "y": 0,
      "width": 250,
      "height": 100,
      "text": "Hello, Canvas!"
    }
  ],
  "edges": []
}
```

### Node Types

```json
{
  "nodes": [
    {
      "id": "text-node",
      "type": "text",
      "x": 0, "y": 0,
      "width": 250, "height": 100,
      "text": "## Markdown content\nSupports **bold**, *italic*, [[wikilinks]]"
    },
    {
      "id": "file-node",
      "type": "file",
      "x": 300, "y": 0,
      "width": 400, "height": 300,
      "file": "Notes/My Note.md"
    },
    {
      "id": "link-node",
      "type": "link",
      "x": 0, "y": 200,
      "width": 400, "height": 200,
      "url": "https://obsidian.md"
    },
    {
      "id": "group-node",
      "type": "group",
      "x": -50, "y": -50,
      "width": 600, "height": 400,
      "label": "My Group",
      "background": "#ff6b6b",
      "backgroundStyle": "solid"
    }
  ]
}
```

### Edges (Connections)

```json
{
  "edges": [
    {
      "id": "edge1",
      "fromNode": "node1",
      "fromSide": "right",
      "toNode": "node2",
      "toSide": "left",
      "label": "connects to",
      "color": "#ff0000"
    }
  ]
}
```

Edge sides: `top`, `right`, `bottom`, `left`

### Node Colors

```json
{
  "color": "1"
}
```

Preset colors: `"1"` (red), `"2"` (orange), `"3"` (yellow), `"4"` (green), `"5"` (cyan), `"6"` (purple)
Custom: `"color": "#ff6b6b"`

### Full Canvas Example

```json
{
  "nodes": [
    {
      "id": "idea-central",
      "type": "text",
      "x": 200,
      "y": 200,
      "width": 300,
      "height": 150,
      "color": "4",
      "text": "# Central Idea\nThe core concept here."
    },
    {
      "id": "note-ref",
      "type": "file",
      "x": 600,
      "y": 100,
      "width": 400,
      "height": 300,
      "file": "Research/Background.md"
    },
    {
      "id": "web-ref",
      "type": "link",
      "x": -200,
      "y": 150,
      "width": 350,
      "height": 200,
      "url": "https://example.com/source"
    }
  ],
  "edges": [
    {
      "id": "e1",
      "fromNode": "idea-central",
      "fromSide": "right",
      "toNode": "note-ref",
      "toSide": "left",
      "label": "supported by"
    },
    {
      "id": "e2",
      "fromNode": "web-ref",
      "fromSide": "right",
      "toNode": "idea-central",
      "toSide": "left",
      "label": "references"
    }
  ]
}
```

---

## Obsidian CLI

The Obsidian CLI lets you interact with vaults programmatically.

### Common Commands

```bash
# Open a vault
obsidian open /path/to/vault

# Open a specific note
obsidian open /path/to/vault --file "Notes/My Note.md"

# Create a new note
obsidian new /path/to/vault --file "Notes/New Note.md" --content "# New Note"

# List all notes in vault
obsidian list /path/to/vault

# Search vault
obsidian search /path/to/vault --query "search term"

# Install a plugin
obsidian plugin install /path/to/vault --id dataview

# Enable a plugin
obsidian plugin enable /path/to/vault --id dataview

# List installed plugins
obsidian plugin list /path/to/vault

# Install a theme
obsidian theme install /path/to/vault --name "Minimal"
```

---

## Defuddle — Clean Web Extraction

Defuddle extracts clean Markdown from web pages, removing navigation, ads, and clutter.

### Install

```bash
npm install -g defuddle-cli
```

### Usage

```bash
# Extract clean markdown from a URL
defuddle https://example.com/article

# Save output to a file
defuddle https://example.com/article -o output.md

# Extract and save directly into your vault
defuddle https://example.com/article -o /path/to/vault/Clippings/article.md

# Include metadata as frontmatter
defuddle https://example.com/article --frontmatter

# Specify output format
defuddle https://example.com/article --format markdown
defuddle https://example.com/article --format text
```

### Programmatic Usage

```javascript
import { Defuddle } from 'defuddle';

const result = await Defuddle.parse('https://example.com/article');
console.log(result.markdown);
console.log(result.title);
console.log(result.author);
console.log(result.date);
```

### Save Clippings to Vault

```bash
#!/bin/bash
# save-clip.sh — Save a URL as a clipping to Obsidian vault
URL="$1"
VAULT="/path/to/vault"
DATE=$(date +%Y-%m-%d)

defuddle "$URL" \
  --frontmatter \
  -o "$VAULT/Clippings/$DATE-clipping.md"
```

---

## Common Patterns

### Creating a Project Note

```markdown
---
title: My Project
type: project
status: active
due: 2026-06-01
tags:
  - work
  - development
---

# My Project

## Overview

Brief description here.

## Tasks

- [ ] Task one
- [ ] Task two
- [x] Completed task

## Related Notes

- [[Research/Background]]
- [[Meetings/Kickoff Meeting]]

## Resources

![[attachments/diagram.png]]
```

### Creating a Base for Active Projects

```yaml
# active-projects.base
view: board
groupBy: status
filter:
  type: and
  conditions:
    - type: eq
      field: type
      value: project
    - type: neq
      field: status
      value: archived
fields:
  - title
  - due
  - priority
sort:
  - field: due
    order: asc
summary:
  - field: title
    type: count
```

### Script: Bulk Import Web Clippings

```bash
#!/bin/bash
# bulk-clip.sh — Import a list of URLs into Obsidian vault
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/vault}"
CLIPPINGS_DIR="$VAULT/Clippings"
mkdir -p "$CLIPPINGS_DIR"

while IFS= read -r url; do
  filename=$(echo "$url" | sed 's|https\?://||;s|/|-|g' | cut -c1-60)
  echo "Clipping: $url"
  defuddle "$url" --frontmatter -o "$CLIPPINGS_DIR/${filename}.md"
done < urls.txt
```

---

## Troubleshooting

### Skills not loading in Claude Code

Ensure the repo is placed in the `/.claude` folder at the **root of your vault**, not in a subfolder:

```
vault/
  .claude/
    skills/
      obsidian-markdown/
        SKILL.md
      ...
```

### Skills not loading in OpenCode

Make sure you cloned the **full repo** (not just the `skills/` subfolder) into `~/.opencode/skills/`:

```
~/.opencode/skills/
  obsidian-skills/          ← full repo root here
    skills/
      obsidian-markdown/
        SKILL.md
```

Restart OpenCode after cloning — skills are auto-discovered on startup.

### Defuddle returns empty output

Some sites require JavaScript rendering. Use a headless browser flag if supported:

```bash
defuddle https://example.com --js
```

### Wikilinks not resolving

Wikilinks are case-sensitive and must match the note filename exactly. Check:
- Capitalization matches the file name
- Special characters are handled correctly
- The note exists in the vault (not just in a different vault)

### Base file not showing data

- Ensure the `.base` file is inside your Obsidian vault directory
- Check that referenced properties exist in your notes' frontmatter
- Validate YAML syntax in both the base file and note frontmatter

---

## Environment Variables

```bash
# Set your default vault path for CLI scripts
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"

# Defuddle API key (if using cloud features)
export DEFUDDLE_API_KEY="your-key-here"
```

---

## Resources

- [Obsidian Help: Obsidian Flavored Markdown](https://help.obsidian.md/obsidian-flavored-markdown)
- [Obsidian Help: Bases Syntax](https://help.obsidian.md/bases/syntax)
- [JSON Canvas Spec](https://jsoncanvas.org/)
- [Defuddle CLI](https://github.com/kepano/defuddle-cli)
- [Agent Skills Specification](https://agentskills.io/specification)
- [obsidian-skills GitHub](https://github.com/kepano/obsidian-skills)
```
