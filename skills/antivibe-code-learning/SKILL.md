```markdown
---
name: antivibe-code-learning
description: Transform AI-generated code into educational deep dives with AntiVibe, a Claude Code skill that explains what code does, why it was written that way, and how to learn from it.
triggers:
  - "deep dive into this code"
  - "explain what AI wrote"
  - "learn from this code"
  - "understand what AI wrote"
  - "generate a learning guide"
  - "antivibe this code"
  - "explain the design decisions here"
  - "turn this into a learning resource"
---

# AntiVibe Code Learning

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

AntiVibe is a Claude Code skill that fights vibe-coding by turning AI-generated code into structured, educational deep dives. Instead of copy-pasting code you don't understand, AntiVibe generates Markdown learning guides explaining **what** code does, **why** it was written that way, **when** to use these patterns, and **what alternatives** exist.

---

## Installation

```bash
# Clone and install as a global Claude Code skill
git clone https://github.com/mohi-devhub/antivibe.git
cp -r antivibe ~/.claude/skills/antivibe
```

For project-scoped installation:
```bash
cp -r antivibe .claude/skills/antivibe
```

To enable auto-trigger hooks (automatic deep dives after task completion):
```bash
cp antivibe/hooks/hooks.json .claude/hooks.json
```

---

## File Structure

```
antivibe/
├── SKILL.md                     # Main skill definition
├── hooks/
│   └── hooks.json              # SubagentStop / Stop hooks
├── scripts/
│   ├── capture-phase.sh        # Detect implementation phases
│   ├── analyze-code.sh         # Parse code structure
│   ├── find-resources.sh       # Find external resources
│   └── generate-deep-dive.sh  # Generate markdown output
├── agents/
│   └── explainer.md            # Subagent for detailed analysis
├── templates/
│   └── deep-dive.md            # Output template
└── reference/
    ├── language-patterns.md    # Framework-specific patterns
    └── resource-curation.md    # Curated learning resources
```

---

## Core Commands

AntiVibe responds to natural language triggers inside Claude Code sessions:

| Trigger | Action |
|--------|--------|
| `/antivibe` | Start an interactive deep dive |
| `"deep dive"` | Analyze recently written code |
| `"learn from this code"` | Generate a full learning guide |
| `"explain what AI wrote"` | Explain specific files |
| `"understand what AI wrote"` | Focus on design decisions |

Output is saved to:
```
deep-dive/<topic>-<date>.md
```

---

## Example Output

After triggering a deep dive on an auth system, AntiVibe generates:

```markdown
# Deep Dive: Authentication System

## Overview
This auth system uses JWT tokens with refresh token rotation...

## Code Walkthrough

### auth/service.ts
- **Purpose**: Token generation and validation
- **Key Components**:
  - `generateTokens()`: Creates access/refresh token pair
  - `verifyToken()`: Validates JWT signatures against secret

## Concepts Explained

### JWT (JSON Web Tokens)
- **What**: Stateless, signed tokens encoding user claims
- **Why**: Server avoids session storage; tokens are self-contained
- **When**: APIs, SPAs, microservices needing stateless auth
- **Alternatives**: Sessions + cookies, Paseto tokens, opaque tokens

## Learning Resources
- [JWT.io](https://jwt.io) — Interactive decoder and official docs
- [Auth0 Best Practices](https://auth0.com/blog) — Real-world patterns

## Next Steps
1. Study refresh token rotation to prevent reuse attacks
2. Read OWASP JWT Security Cheat Sheet
3. Explore token revocation strategies
```

---

## Configuration

### Change Output Directory

Edit `scripts/generate-deep-dive.sh`:

```bash
#!/usr/bin/env bash
OUTPUT_DIR="learning-notes"   # Default is "deep-dive"
DATE=$(date +%Y-%m-%d)
TOPIC="${1:-general}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/${TOPIC}-${DATE}.md"

# Render the deep-dive template with analyzed content
cat > "$OUTPUT_FILE" << EOF
# Deep Dive: $TOPIC
...
EOF

echo "Saved to: $OUTPUT_FILE"
```

### Auto-Trigger Hooks

`hooks/hooks.json` wires AntiVibe into Claude Code's event system:

```json
{
  "hooks": [
    {
      "event": "SubagentStop",
      "command": "bash ~/.claude/skills/antivibe/scripts/capture-phase.sh"
    },
    {
      "event": "Stop",
      "command": "bash ~/.claude/skills/antivibe/scripts/generate-deep-dive.sh session-summary"
    }
  ]
}
```

- **SubagentStop**: Fires when a sub-task finishes — captures phase-level explanations
- **Stop**: Fires when the session ends — generates a full session summary

---

## Scripts Reference

### `capture-phase.sh`
Detects which implementation phase just completed (e.g., "auth", "database", "API layer") and tags the context for the final deep dive.

```bash
#!/usr/bin/env bash
# Reads recent git diff or file changes to detect phase
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")
echo "Phase context: $CHANGED_FILES" >> /tmp/antivibe-phase.log
```

### `analyze-code.sh`
Parses code structure — functions, classes, imports — and feeds them to the explainer agent.

```bash
#!/usr/bin/env bash
TARGET="${1:-.}"
# List all source files modified recently
find "$TARGET" -name "*.ts" -o -name "*.py" -o -name "*.go" \
  | xargs grep -l "export\|def \|func " 2>/dev/null
```

### `find-resources.sh`
Maps detected concepts (JWT, React hooks, goroutines, etc.) to curated resources in `reference/resource-curation.md`.

```bash
#!/usr/bin/env bash
CONCEPT="$1"
grep -A 3 "### $CONCEPT" \
  ~/.claude/skills/antivibe/reference/resource-curation.md
```

---

## Extending AntiVibe

### Add Language Patterns

Edit `reference/language-patterns.md` to add framework-specific explanations:

```markdown
## Go

### Goroutines
- **Pattern**: `go func() { ... }()`
- **Why**: Lightweight concurrency without OS threads
- **Gotchas**: Always handle done channels to avoid leaks
- **Resources**: [Go Tour Concurrency](https://tour.golang.org/concurrency/1)

### Error Wrapping
- **Pattern**: `fmt.Errorf("context: %w", err)`
- **Why**: Preserves error chain for `errors.Is` / `errors.As`
```

### Add Curated Resources

Edit `reference/resource-curation.md`:

```markdown
## Authentication

### JWT
- [jwt.io](https://jwt.io) — Decoder + library list
- [OWASP JWT Cheat Sheet](https://cheatsheetseries.owasp.org) — Security patterns

### OAuth2
- [OAuth2 Simplified](https://aaronparecki.com/oauth-2-simplified/) — Plain-language guide
```

### Customize the Template

Edit `templates/deep-dive.md` to match your team's style:

```markdown
# Deep Dive: {{TOPIC}}
Generated: {{DATE}}

## TL;DR
{{SUMMARY}}

## Code Walkthrough
{{WALKTHROUGH}}

## Concepts
{{CONCEPTS}}

## Resources
{{RESOURCES}}

## What to Study Next
{{NEXT_STEPS}}
```

---

## Supported Languages & Frameworks

AntiVibe's pattern library covers:

| Language | Frameworks |
|----------|-----------|
| TypeScript/JavaScript | React, Node.js, Express, Next.js |
| Python | Django, FastAPI, Flask |
| Go | Standard library, Gin, Echo |
| Rust | Standard library, Actix-web |
| Java | Spring Boot |

Add more in `reference/language-patterns.md`.

---

## AntiVibe Principles

When generating deep dives, always follow these rules:

1. **Why over what** — Design decisions matter more than syntax
2. **Context matters** — Explain when and why to use each pattern
3. **Curated resources** — Link to authoritative docs, not random blogs
4. **Phase-aware** — Group explanations by implementation phase
5. **Learning path** — Always end with "What to Study Next"
6. **Concept mapping** — Connect implementation to underlying CS concepts

---

## Troubleshooting

**Deep dive not generating?**
- Confirm the skill is installed: `ls ~/.claude/skills/antivibe/SKILL.md`
- Check script permissions: `chmod +x ~/.claude/skills/antivibe/scripts/*.sh`

**Hooks not firing?**
- Verify hooks.json is in `.claude/hooks.json` (project root), not the skill folder
- Check hook event names match Claude Code's supported events (`SubagentStop`, `Stop`)

**Output directory missing?**
- The scripts create it automatically; if not, run: `mkdir -p deep-dive`

**Resources not found for a concept?**
- The concept name may not match a heading in `resource-curation.md`
- Add it: `echo "### YourConcept" >> reference/resource-curation.md`

---

## Quick Reference Card

```
Install:   cp -r antivibe ~/.claude/skills/antivibe
Trigger:   "deep dive" | "explain what AI wrote" | /antivibe
Output:    deep-dive/<topic>-<date>.md
Extend:    reference/language-patterns.md
           reference/resource-curation.md
Template:  templates/deep-dive.md
Hooks:     hooks/hooks.json → .claude/hooks.json
```
```
