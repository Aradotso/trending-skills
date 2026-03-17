```markdown
---
name: awesome-openclaw-skills
description: Curated collection of 5400+ community-built OpenClaw skills organized by category, with installation, discovery, and contribution guidance.
triggers:
  - install an openclaw skill
  - find openclaw skills for
  - browse clawhub skills
  - add a skill to openclaw
  - discover openclaw integrations
  - how to install a clawdbot skill
  - search awesome openclaw list
  - contribute a skill to openclaw registry
---

# Awesome OpenClaw Skills

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A curated, filtered list of 5,400+ community-built [OpenClaw](https://github.com/VoltAgent/voltagent) skills sourced from ClawHub (the official OpenClaw Skills Registry). Skills extend the OpenClaw AI assistant with integrations, automations, and specialized capabilities.

---

## What Is OpenClaw?

OpenClaw is a locally-running AI assistant. **Skills** are modular extensions that let it:

- Interact with external services (GitHub, Slack, Gmail, etc.)
- Automate workflows
- Perform specialized tasks (code generation, research, document processing, etc.)

Skills are distributed through **ClawHub** (`clawhub.ai`) and managed via the `clawhub` CLI.

---

## Installing a Skill

### Via ClawHub CLI (recommended)

```bash
# Install a skill by its slug
clawhub install <skill-slug>

# Examples
clawhub install steipete-slack
clawhub install sohamganatra-bitbucket-automation
clawhub install xukp20-arxiv-search-collector
```

### Manual Installation

Copy the skill folder to one of these paths:

| Scope     | Path                          |
|-----------|-------------------------------|
| Global    | `~/.openclaw/skills/`         |
| Workspace | `<your-project>/skills/`      |

**Priority order:** Workspace > Local/Global > Bundled

```bash
# Example: manually install a skill globally
cp -r ./my-skill ~/.openclaw/skills/my-skill

# Or workspace-scoped
cp -r ./my-skill ./skills/my-skill
```

### Via Chat (Zero-Config)

Paste a skill's GitHub repository URL directly into the OpenClaw chat and ask the assistant to use it. OpenClaw will handle setup automatically.

```
"Use this skill: https://github.com/openclaw/skills/tree/main/skills/steipete/slack"
```

---

## Discovering Skills

### Browse This Awesome List

Skills are organized into categories in the repository:

```
categories/
  git-and-github.md
  coding-agents-and-ides.md
  browser-and-automation.md
  web-and-frontend-development.md
  devops-and-cloud.md
  ...
```

Each category file lists skills with a short description and a ClawHub link:

```
https://clawskills.sh/skills/<author>-<skill-name>
```

### Key Categories & Counts

| Category | Skills |
|---|---|
| Coding Agents & IDEs | 1,222 |
| Web & Frontend Development | 938 |
| Browser & Automation | 335 |
| Search & Research | 352 |
| DevOps & Cloud | 409 |
| Productivity & Tasks | 206 |
| AI & LLMs | 197 |
| CLI Utilities | 186 |
| Communication | 149 |
| Git & GitHub | 170 |

---

## Skill Structure

A typical OpenClaw skill is a folder with:

```
my-skill/
  SKILL.md          # Skill manifest and instructions for the agent
  tools/            # Tool definitions (functions the agent can call)
    my_tool.py      # or .js, .ts, .sh depending on skill
  config.yaml       # Optional configuration schema
  README.md         # Human-readable documentation
```

### Minimal `SKILL.md` Example

```markdown
---
name: my-skill
description: Does something useful with an external API.
triggers:
  - use my service
  - connect to my service
  - fetch data from my service
---

# My Skill

Connects OpenClaw to MyService API.

## Configuration

Set `MY_SERVICE_API_KEY` in your environment.

## Tools

- `fetch_data` — Retrieves records from MyService.
- `post_update` — Sends an update to MyService.
```

### Python Tool Example

```python
# tools/fetch_data.py
import os
import requests

API_KEY = os.environ["MY_SERVICE_API_KEY"]
BASE_URL = "https://api.myservice.example.com/v1"

def fetch_data(query: str, limit: int = 10) -> dict:
    """Fetch records from MyService matching the query."""
    response = requests.get(
        f"{BASE_URL}/search",
        headers={"Authorization": f"Bearer {API_KEY}"},
        params={"q": query, "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
```

### TypeScript Tool Example

```typescript
// tools/fetchData.ts
import axios from "axios";

const API_KEY = process.env.MY_SERVICE_API_KEY!;
const BASE_URL = "https://api.myservice.example.com/v1";

export async function fetchData(query: string, limit = 10) {
  const { data } = await axios.get(`${BASE_URL}/search`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
    params: { q: query, limit },
  });
  return data;
}
```

---

## Configuration

Skills that require credentials use environment variables. Set them in your shell or a `.env` file at your project root:

```bash
# .env (never commit this file)
GITHUB_TOKEN=ghp_...
SLACK_BOT_TOKEN=xoxb-...
OPENAI_API_KEY=sk-...
MY_SERVICE_API_KEY=...
```

OpenClaw loads `.env` automatically when starting in a workspace directory.

---

## Contributing a Skill

1. **Publish to the official registry first:**
   ```
   https://github.com/openclaw/skills
   ```
   Skills must be in the official `openclaw/skills` repo — personal repos or gists are not accepted.

2. **Open a PR to this awesome list** with:
   - ClawHub link: `https://clawhub.ai/<author>/<skill-name>`
   - GitHub link: `https://github.com/openclaw/skills/tree/main/skills/<author>/<skill-name>`

3. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for formatting rules.

---

## Security

Skills are **curated, not audited**. Before installing:

- Review the source code on GitHub
- Check the VirusTotal report on the skill's ClawHub page (OpenClaw has a VirusTotal partnership)
- Use recommended scanners:
  - [Snyk Skill Security Scanner](https://github.com/snyk/agent-scan)
  - [Agent Trust Hub](https://ai.gendigital.com/agent-trust-hub)

**Risks to watch for:**
- Prompt injection in skill manifests
- Tool poisoning
- Hidden malware payloads
- Unsafe data handling

Report suspicious skills by [opening an issue](https://github.com/VoltAgent/awesome-clawdbot-skills/issues).

---

## What Was Filtered Out

This list curates 5,366 of the 13,729 skills on ClawHub:

| Filter | Excluded |
|---|---|
| Spam / bulk / bot / junk accounts | 4,065 |
| Duplicates or near-duplicate names | 1,040 |
| Low-quality or non-English descriptions | 851 |
| Crypto / blockchain / finance / trading | 731 |
| Confirmed malicious (security audit flagged) | 373 |
| **Total excluded** | **7,060** |

---

## Common Patterns

### Using a skill to automate a GitHub workflow

```bash
clawhub install trypto1019-arc-skill-gitops
```

Then in OpenClaw chat:
```
"Deploy the latest version of my skill and roll back if tests fail."
```

### Chaining skills (multi-agent orchestration)

```bash
clawhub install arminnaimi-agent-team-orchestration
```

```
"Orchestrate a team: one agent searches arXiv, another summarizes, another posts to Slack."
```

### Installing a browser automation skill

```bash
clawhub install <browser-skill-slug>
```

```
"Scrape the pricing table from example.com and save it as JSON."
```

---

## Resources

| Resource | URL |
|---|---|
| OpenClaw / VoltAgent GitHub | https://github.com/VoltAgent/voltagent |
| ClawHub Registry | https://clawhub.ai |
| Official Skills Repo | https://github.com/openclaw/skills |
| This Awesome List | https://github.com/VoltAgent/awesome-openclaw-skills |
| Discord Community | https://s.voltagent.dev/discord |
| Composio (managed OAuth) | https://composio.dev |

---

## Troubleshooting

**Skill not found after install:**
```bash
# Verify the skill exists in the correct path
ls ~/.openclaw/skills/
ls ./skills/

# Reinstall
clawhub install <skill-slug> --force
```

**Environment variable not picked up:**
```bash
# Confirm the variable is exported
export MY_SERVICE_API_KEY="..."

# Or ensure .env is in the project root
cat .env | grep MY_SERVICE_API_KEY
```

**Skill conflict (workspace vs global):**
Workspace-scoped skills take priority. Remove the global version if you want workspace-only:
```bash
rm -rf ~/.openclaw/skills/<skill-name>
```

**Flagged by VirusTotal:**
Do not install. Open an issue at:
```
https://github.com/VoltAgent/awesome-clawdbot-skills/issues
```
```
