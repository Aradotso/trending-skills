```markdown
---
name: awesome-codex-subagents
description: Collection of 136+ specialized Codex subagents in TOML format covering core development, language specialists, infrastructure, quality/security, and more categories.
triggers:
  - set up codex subagents
  - add a codex subagent
  - install codex agent
  - create custom codex subagent
  - configure codex agents directory
  - use specialized ai subagents with codex
  - add backend developer subagent
  - delegate tasks to codex subagents
---

# Awesome Codex Subagents

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A curated collection of 136+ ready-to-use Codex subagents in `.toml` format. Each subagent is a specialized AI assistant scoped to a specific development role — from backend developer to Kubernetes specialist to security auditor. Install them globally or per-project and delegate tasks to them explicitly in your Codex prompts.

---

## What This Project Does

- Provides pre-built `.toml` subagent definitions for Codex
- Each agent has a focused role, tuned model, sandbox permissions, and detailed instructions
- Covers 10 categories: Core Development, Language Specialists, Infrastructure, Quality & Security, AI/ML, Data Engineering, Documentation, Specialized Domains, Workflow, and Research
- Agents are composable — you can use multiple subagents in a single session

---

## Installation

### Prerequisites

- [OpenAI Codex CLI](https://github.com/openai/codex) installed
- A valid Codex session/API access

### Clone the Repository

```bash
git clone https://github.com/VoltAgent/awesome-codex-subagents.git
cd awesome-codex-subagents
```

### Install Global Agents (available in all projects)

```bash
mkdir -p ~/.codex/agents

# Install a single agent
cp categories/01-core-development/backend-developer.toml ~/.codex/agents/

# Install an entire category
cp categories/01-core-development/*.toml ~/.codex/agents/

# Install all agents
cp categories/**/*.toml ~/.codex/agents/
```

### Install Project-Specific Agents (higher precedence)

```bash
mkdir -p .codex/agents

cp categories/04-quality-security/reviewer.toml .codex/agents/
cp categories/04-quality-security/security-auditor.toml .codex/agents/
```

> **Note:** Project-level agents (`.codex/agents/`) override global agents (`~/.codex/agents/`) when names conflict.

---

## Directory Structure

```
awesome-codex-subagents/
├── categories/
│   ├── 01-core-development/
│   │   ├── backend-developer.toml
│   │   ├── frontend-developer.toml
│   │   ├── api-designer.toml
│   │   └── ...
│   ├── 02-language-specialists/
│   │   ├── python-pro.toml
│   │   ├── typescript-pro.toml
│   │   ├── rust-engineer.toml
│   │   └── ...
│   ├── 03-infrastructure/
│   │   ├── devops-engineer.toml
│   │   ├── kubernetes-specialist.toml
│   │   ├── terraform-engineer.toml
│   │   └── ...
│   ├── 04-quality-security/
│   │   ├── code-reviewer.toml
│   │   ├── security-auditor.toml
│   │   ├── qa-expert.toml
│   │   └── ...
│   └── ...
```

---

## Subagent TOML Format

Each subagent is a `.toml` file with this structure:

```toml
name = "backend-developer"
description = "When to invoke: building APIs, server logic, database models, authentication, or any server-side feature"
model = "gpt-5.3-codex-spark"
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"

[instructions]
text = """
You are a senior backend engineer specializing in scalable server-side architecture...

## Core Responsibilities
- Design and implement RESTful and GraphQL APIs
- Write performant database queries and ORM models
- Implement authentication and authorization patterns
- ...
"""
```

### Key Fields

| Field | Description |
|-------|-------------|
| `name` | Unique identifier used to delegate tasks |
| `description` | When Codex should use this agent |
| `model` | Which model powers this agent |
| `model_reasoning_effort` | `low`, `medium`, or `high` |
| `sandbox_mode` | `read-only` or `workspace-write` |
| `[instructions].text` | The full system prompt for the agent |

---

## Model Routing Reference

| Model | Best For | Example Agents |
|-------|----------|----------------|
| `gpt-5.4` | Deep reasoning, architecture, security audits | `security-auditor`, `architect-reviewer` |
| `gpt-5.3-codex-spark` | Fast scanning, synthesis, lighter tasks | `search-specialist`, `docs-researcher` |

---

## Sandbox Mode Reference

| Mode | File Access | Best For |
|------|-------------|----------|
| `read-only` | Can read, cannot write | Reviewers, auditors, analyzers |
| `workspace-write` | Full read/write | Developers, engineers, builders |

---

## Using Subagents in Codex

Codex does **not** auto-spawn custom subagents — you must delegate explicitly in your prompt.

### Basic Delegation

```
Ask the backend-developer subagent to add a POST /api/users endpoint with email validation and bcrypt password hashing.
```

```
Use the typescript-pro subagent to refactor src/utils/date.js to TypeScript with strict types.
```

```
Have the code-reviewer subagent review the changes in src/auth/ for security issues and best practices.
```

### Multi-Agent Workflow

```
1. Use the api-designer subagent to design the schema for a payments API
2. Then have the backend-developer subagent implement it
3. Finally, use the security-auditor subagent to review the implementation
```

### With Specific Files

```
Ask the python-pro subagent to optimize the database queries in app/models/user.py — focus on N+1 query elimination.
```

---

## Creating a Custom Subagent

You can write your own `.toml` agent and drop it in the agents directory:

```toml
# .codex/agents/stripe-integration-expert.toml

name = "stripe-integration-expert"
description = "When working with Stripe payments, webhooks, subscriptions, or billing logic"
model = "gpt-5.3-codex-spark"
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"

[instructions]
text = """
You are a Stripe integration specialist with deep knowledge of:
- Stripe Checkout and Payment Intents API
- Subscription and billing lifecycle management
- Webhook signature verification and event handling
- SCA/3DS compliance
- Stripe CLI for local webhook testing

## Key Patterns

### Always verify webhook signatures
```python
import stripe
from django.http import HttpResponse

def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ['STRIPE_WEBHOOK_SECRET']
        )
    except stripe.error.SignatureVerificationError:
        return HttpResponse(status=400)
```

### Use idempotency keys for payment creation
### Handle all relevant webhook event types
### Never log or expose raw card data
"""
```

---

## Category Quick Reference

### Core Development (`01-core-development`)

```bash
cp categories/01-core-development/backend-developer.toml ~/.codex/agents/
cp categories/01-core-development/frontend-developer.toml ~/.codex/agents/
cp categories/01-core-development/api-designer.toml ~/.codex/agents/
cp categories/01-core-development/fullstack-developer.toml ~/.codex/agents/
cp categories/01-core-development/ui-fixer.toml ~/.codex/agents/
```

### Language Specialists (`02-language-specialists`)

```bash
# Install just the languages you use
cp categories/02-language-specialists/python-pro.toml ~/.codex/agents/
cp categories/02-language-specialists/typescript-pro.toml ~/.codex/agents/
cp categories/02-language-specialists/golang-pro.toml ~/.codex/agents/
cp categories/02-language-specialists/rust-engineer.toml ~/.codex/agents/
cp categories/02-language-specialists/nextjs-developer.toml ~/.codex/agents/
```

### Infrastructure (`03-infrastructure`)

```bash
cp categories/03-infrastructure/devops-engineer.toml ~/.codex/agents/
cp categories/03-infrastructure/terraform-engineer.toml ~/.codex/agents/
cp categories/03-infrastructure/kubernetes-specialist.toml ~/.codex/agents/
cp categories/03-infrastructure/docker-expert.toml ~/.codex/agents/
```

### Quality & Security (`04-quality-security`)

```bash
cp categories/04-quality-security/code-reviewer.toml ~/.codex/agents/
cp categories/04-quality-security/security-auditor.toml ~/.codex/agents/
cp categories/04-quality-security/qa-expert.toml ~/.codex/agents/
cp categories/04-quality-security/debugger.toml ~/.codex/agents/
cp categories/04-quality-security/performance-engineer.toml ~/.codex/agents/
```

---

## Codex Config Integration

You can also reference agents in `.codex/config.toml`:

```toml
# .codex/config.toml

[agents]
default = "fullstack-developer"

[agents.overrides]
"**/*.test.ts" = "qa-expert"
"**/infra/**" = "terraform-engineer"
"**/*.sql" = "sql-pro"
```

---

## Common Patterns

### PR Review Workflow

```
Use the reviewer subagent to do a PR-style review of all files changed in the last commit. 
Check for correctness, security issues, and potential regressions.
```

### Debugging Workflow

```
The login endpoint returns 500 for users with special characters in their email.
Ask the debugger subagent to trace through src/auth/login.ts and identify the root cause.
```

### Infrastructure Review

```
Use the terraform-engineer subagent to review infra/aws/ for security misconfigurations 
and suggest improvements for least-privilege IAM policies.
```

### Tech Stack Specific

```
Ask the nextjs-developer subagent to migrate pages/dashboard.js to the App Router 
pattern using React Server Components and the new Next.js 14 data fetching conventions.
```

---

## Troubleshooting

### Agent Not Found

```
Error: No agent named 'backend-developer' found
```

**Fix:** Ensure the `.toml` file is in `~/.codex/agents/` or `.codex/agents/` and the `name` field in the TOML matches what you're referencing.

```bash
ls ~/.codex/agents/
# Verify the file exists and check the name field:
grep '^name' ~/.codex/agents/backend-developer.toml
```

### Project Agent Not Taking Precedence

Ensure your project-level `.codex/agents/` directory is at the repo root (same level as `.git/`):

```bash
ls -la .codex/agents/
```

### Agent Using Wrong Model

Check the `model` field in the `.toml` file. Override it for your needs:

```toml
# Change from spark to full model for complex tasks
model = "gpt-5.4"
model_reasoning_effort = "high"
```

### Sandbox Permission Errors

If an agent can't write files, check `sandbox_mode`:

```toml
# Change read-only to workspace-write if the agent needs to create files
sandbox_mode = "workspace-write"
```

---

## Bulk Install Script

```bash
#!/bin/bash
# install-codex-agents.sh
# Install a curated subset of agents for a typical web project

AGENTS_DIR="${1:-$HOME/.codex/agents}"
REPO_DIR="$(pwd)"

mkdir -p "$AGENTS_DIR"

AGENTS=(
  "categories/01-core-development/backend-developer.toml"
  "categories/01-core-development/frontend-developer.toml"
  "categories/01-core-development/api-designer.toml"
  "categories/02-language-specialists/typescript-pro.toml"
  "categories/02-language-specialists/python-pro.toml"
  "categories/03-infrastructure/devops-engineer.toml"
  "categories/03-infrastructure/docker-expert.toml"
  "categories/04-quality-security/code-reviewer.toml"
  "categories/04-quality-security/debugger.toml"
  "categories/04-quality-security/security-auditor.toml"
)

for agent in "${AGENTS[@]}"; do
  if [ -f "$REPO_DIR/$agent" ]; then
    cp "$REPO_DIR/$agent" "$AGENTS_DIR/"
    echo "✓ Installed $(basename $agent)"
  else
    echo "✗ Not found: $agent"
  fi
done

echo ""
echo "Installed agents to: $AGENTS_DIR"
ls "$AGENTS_DIR"
```

```bash
chmod +x install-codex-agents.sh
./install-codex-agents.sh                        # installs to ~/.codex/agents/
./install-codex-agents.sh .codex/agents/         # installs to project-level
```

---

## Resources

- [GitHub Repository](https://github.com/VoltAgent/awesome-codex-subagents)
- [OpenAI Codex Subagents Docs](https://developers.openai.com/codex/subagents)
- [VoltAgent Discord](https://s.voltagent.dev/discord)
```
