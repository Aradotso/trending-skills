```markdown
---
name: runbookhermes-aiops-agent
description: Hermes-native AIOps agent for evidence-driven incident response, approval-gated remediation, and runbook learning on payment and production systems.
triggers:
  - set up RunbookHermes for incident response
  - add approval-gated remediation to my AIOps agent
  - integrate Prometheus and Loki with runbook agent
  - create a runbook skill from an incident
  - configure evidence-driven root cause analysis
  - set up Alertmanager webhook for incident intake
  - add human-in-the-loop approval for production actions
  - build a Hermes AIOps agent with runbook learning
---

# RunbookHermes AIOps Agent

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

RunbookHermes is a Hermes-native AIOps agent for payment and production incident response. It extends the Hermes Agent runtime with evidence collection (Prometheus, Loki, Jaeger), an `EvidenceStack` context engine, `IncidentMemory`, approval-gated remediation, checkpoint/rollback safety gates, and runbook skill generation — turning one-off incident handling into accumulated operational knowledge.

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the Web Console static assets)
- Docker (optional, for the reference payment demo environment)

### Clone and Install

```bash
git clone https://github.com/Tommy-yw/RunbookHermes.git
cd RunbookHermes
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Install the Web Console dependencies (optional)

```bash
cd web
npm install
npm run build
cd ..
```

### Start the API + Web Console

```bash
uvicorn apps.runbook_api.main:app --host 0.0.0.0 --port 8080 --reload
```

Navigate to `http://localhost:8080` for the AIOps Web Console.

---

## Configuration

All production integrations are controlled by environment variables. Copy `.env.example` to `.env` and fill in your values.

```bash
cp .env.example .env
```

### Core environment variables

```dotenv
# ── Model provider (OpenAI-compatible endpoint) ──────────────────────────────
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# ── Observability backends ────────────────────────────────────────────────────
PROMETHEUS_URL=http://prometheus:9090
LOKI_URL=http://loki:3100
JAEGER_URL=http://jaeger:16686

# ── Deployment / execution backend ───────────────────────────────────────────
DEPLOY_BACKEND=custom_http          # or: kubernetes, argocd
DEPLOY_API_URL=http://deploy-api/v1
DEPLOY_API_TOKEN=$DEPLOY_API_TOKEN

# ── Notification integrations ────────────────────────────────────────────────
FEISHU_APP_ID=$FEISHU_APP_ID
FEISHU_APP_SECRET=$FEISHU_APP_SECRET
WECOM_CORP_ID=$WECOM_CORP_ID
WECOM_AGENT_ID=$WECOM_AGENT_ID
WECOM_SECRET=$WECOM_SECRET

# ── Approval gate ────────────────────────────────────────────────────────────
APPROVAL_REQUIRED=true              # set false for dry-run-only mode
APPROVAL_TIMEOUT_SECONDS=300

# ── Memory / persistence ─────────────────────────────────────────────────────
INCIDENT_MEMORY_BACKEND=sqlite      # or: postgres, redis
INCIDENT_MEMORY_DSN=sqlite:///data/incident_memory.db

# ── Mock / demo fallback ─────────────────────────────────────────────────────
USE_MOCK_OBSERVABILITY=true         # set false when real backends are reachable
```

Load the `.env` before running:

```bash
export $(cat .env | xargs)
uvicorn apps.runbook_api.main:app --reload
```

---

## Key Concepts

| Concept | Description |
|---|---|
| **EvidenceStack** | Structured context engine — organises alert, evidence, hypotheses, action plan, and final answer instead of dumping raw logs into the prompt. |
| **IncidentMemory** | Domain-specific memory provider — stores service profiles, team preferences, incident summaries, recurring root causes, and runbook skills. |
| **Approval Gate** | All write/destructive actions pass through: policy check → approval request → checkpoint → dry-run → controlled execution → recovery verification. |
| **Runbook Skill** | A reusable, machine-readable operational procedure generated from a successfully resolved incident. |
| **Profile** | `profiles/runbook-hermes/` — the Hermes agent profile that wires together tools, memory, context engine, and persona for AIOps. |

---

## Creating and Managing Incidents

### Via REST API

```python
import httpx

BASE = "http://localhost:8080"

# Create an incident manually
resp = httpx.post(f"{BASE}/api/incidents", json={
    "service": "payment-service",
    "severity": "critical",
    "title": "HTTP 503 spike on /v1/charge",
    "description": "Error rate exceeded 40% for 5 minutes",
    "source": "api",
})
incident = resp.json()
incident_id = incident["id"]
print(f"Created incident: {incident_id}")

# Fetch incident detail (evidence + root cause)
detail = httpx.get(f"{BASE}/api/incidents/{incident_id}").json()
print(detail["executive_summary"])
```

### Via Alertmanager Webhook

Configure Alertmanager to POST to the RunbookHermes gateway:

```yaml
# alertmanager.yml
receivers:
  - name: runbookhermes
    webhook_configs:
      - url: http://runbook-hermes:8080/gateway/alertmanager
        send_resolved: true
```

### Via Feishu / WeCom Event Callback

```python
# The gateway shells are registered automatically.
# Point your Feishu app "Event Subscription → Request URL" to:
#   http://runbook-hermes:8080/gateway/feishu/event
# Point WeCom "Receive Messages URL" to:
#   http://runbook-hermes:8080/gateway/wecom/event
```

---

## Evidence Collection

### Using Observability Adapters Directly

```python
from integrations.observability.prometheus import PrometheusAdapter
from integrations.observability.loki import LokiAdapter
from integrations.observability.trace import TraceAdapter

prom = PrometheusAdapter(base_url="http://prometheus:9090")
loki = LokiAdapter(base_url="http://loki:3100")
trace = TraceAdapter(base_url="http://jaeger:16686")

# Query error rate for the last 10 minutes
metrics = await prom.query_range(
    query='rate(http_requests_total{service="payment-service",status=~"5.."}[1m])',
    start="-10m",
    end="now",
    step="30s",
)

# Fetch recent error logs
logs = await loki.query_logs(
    query='{service="payment-service"} |= "ERROR"',
    limit=100,
    since="10m",
)

# Fetch a trace by ID
trace_data = await trace.get_trace(trace_id="abc123def456")
```

### Using the EvidenceStack Context Engine

```python
from plugins.context_engine.evidence_stack.stack import EvidenceStack

stack = EvidenceStack(incident_id="INC-20260501-001")

# Add alert trigger
stack.add_alert(
    title="HTTP 503 spike",
    service="payment-service",
    severity="critical",
    labels={"env": "prod", "region": "us-east-1"},
)

# Add collected evidence
stack.add_evidence(
    evidence_id="EV-001",
    kind="metric",
    summary="Error rate: 42% over last 5 min",
    source="prometheus",
    raw_ref="prometheus:query:rate(http_requests_total...)",  # not inlined
)

stack.add_evidence(
    evidence_id="EV-002",
    kind="log",
    summary="500+ 'upstream connect error' log lines in payment-service",
    source="loki",
)

# Add a hypothesis
stack.add_hypothesis(
    hypothesis="Upstream Redis connection pool exhausted, causing cascade 503s",
    confidence=0.85,
    supporting_evidence=["EV-001", "EV-002"],
)

# Render compressed context for model prompt
context = stack.render_for_prompt(max_tokens=2048)
print(context)
```

---

## Approval-Gated Remediation

### Submitting a Risky Action for Approval

```python
from runbook_hermes.approval import ApprovalGate, ActionRisk

gate = ApprovalGate()

# Register a high-risk action
approval_request = await gate.request_approval(
    incident_id="INC-20260501-001",
    action="rollback",
    risk=ActionRisk.HIGH,
    payload={
        "service": "payment-service",
        "target_version": "v2.3.1",
        "current_version": "v2.4.0",
    },
    dry_run_result={
        "would_affect": ["payment-service:prod:us-east-1"],
        "estimated_downtime_seconds": 30,
    },
    requester="runbook-hermes-agent",
)

print(f"Approval requested: {approval_request.id}")
print(f"Status: {approval_request.status}")  # pending
```

### Checking and Acting on Approval Status

```python
from runbook_hermes.approval import ApprovalGate

gate = ApprovalGate()

status = await gate.get_status(approval_id="APR-001")

if status.approved:
    result = await gate.execute_approved_action(approval_id="APR-001")
    print(f"Execution result: {result.outcome}")
elif status.rejected:
    print(f"Rejected by {status.decided_by}: {status.reason}")
else:
    print("Still pending operator review")
```

### Approving via API (operator / webhook integration)

```python
import httpx

httpx.post("http://localhost:8080/api/approvals/APR-001/decide", json={
    "decision": "approve",          # or "reject"
    "decided_by": "ops-team-lead",
    "reason": "Confirmed rollback target version is stable",
})
```

---

## IncidentMemory

```python
from plugins.memory.incident_memory.provider import IncidentMemoryProvider

memory = IncidentMemoryProvider(backend="sqlite", dsn="sqlite:///data/incident_memory.db")

# Store a service profile
await memory.store_service_profile(
    service="payment-service",
    profile={
        "tier": "P0",
        "owners": ["payments-team"],
        "runbook_url": "https://wiki.internal/runbooks/payment-service",
        "known_failure_modes": ["redis_pool_exhaustion", "downstream_timeout"],
    },
)

# Recall similar past incidents
similar = await memory.recall_similar_incidents(
    service="payment-service",
    symptoms=["HTTP 503", "redis timeout"],
    limit=3,
)
for inc in similar:
    print(inc.summary, inc.root_cause, inc.resolution)

# Index a newly generated runbook skill
await memory.index_skill(
    skill_id="SKILL-payment-503-redis",
    service="payment-service",
    trigger_symptoms=["HTTP 503", "redis connection refused"],
    skill_path="skills/runbooks/payment_503_redis.yaml",
)
```

---

## Runbook Skills

### Runbook Skill YAML Format

```yaml
# skills/runbooks/payment_503_redis.yaml
id: SKILL-payment-503-redis
version: "1.0"
title: Payment Service HTTP 503 — Redis Pool Exhaustion
service: payment-service
trigger_symptoms:
  - HTTP 503 spike
  - redis connection refused
  - upstream connect error

diagnosis_steps:
  - id: D1
    action: query_metric
    params:
      query: 'redis_connected_clients{service="payment-service"}'
      threshold: "> 450"
      interpretation: "Pool near exhaustion if above threshold"
  - id: D2
    action: query_logs
    params:
      query: '{service="payment-service"} |= "redis: connection refused"'
      since: "5m"

remediation_steps:
  - id: R1
    action: restart_service
    risk: medium
    requires_approval: true
    params:
      service: payment-service
      strategy: rolling
  - id: R2
    action: scale_redis_pool
    risk: low
    requires_approval: false
    params:
      service: payment-service
      pool_size: 600

verification_steps:
  - id: V1
    action: query_metric
    params:
      query: 'rate(http_requests_total{service="payment-service",status="503"}[1m])'
      expected: "< 0.01"
      timeout_seconds: 120

generated_from_incident: INC-20260501-001
generated_at: "2026-05-01T09:45:00Z"
```

### Loading and Executing a Runbook Skill

```python
from skills.runbooks.loader import RunbookSkillLoader
from runbook_hermes.executor import SkillExecutor

loader = RunbookSkillLoader(skills_dir="skills/runbooks/")
skill = loader.load("SKILL-payment-503-redis")

executor = SkillExecutor(
    incident_id="INC-20260501-002",
    approval_gate_enabled=True,
)

result = await executor.run(skill, context={
    "service": "payment-service",
    "environment": "prod",
})

print(result.outcome)          # resolved / partial / failed
print(result.actions_taken)
print(result.verification)
```

### Generating a Skill from a Resolved Incident

```python
from runbook_hermes.skill_generator import SkillGenerator

generator = SkillGenerator()

skill = await generator.generate_from_incident(
    incident_id="INC-20260501-001",
    evidence_stack=stack,         # EvidenceStack from the resolved incident
    approved_actions=[...],       # list of ApprovalRecord objects
    verification_result=result,
)

# Persist to skills directory and index in memory
await skill.save("skills/runbooks/")
await memory.index_skill(skill.id, skill.service, skill.trigger_symptoms, skill.path)
print(f"Skill generated: {skill.id}")
```

---

## Running the Hermes Agent Profile

```bash
# Start the RunbookHermes Hermes profile via CLI
hermes --profile profiles/runbook-hermes/ \
       --incident INC-20260501-001 \
       --mode triage

# Replay a past incident for skill generation
hermes --profile profiles/runbook-hermes/ \
       --replay INC-20260501-001 \
       --generate-skill
```

---

## Demo: Local Reference Payment Environment

```bash
# Start the reference payment microservices (payment, coupon, order)
cd demo/payment_system
docker compose up -d

# Inject a fault (Redis pool exhaustion simulation)
python scripts/inject_fault.py --service payment-service --fault redis_pool_exhaustion

# Trigger a mock alert to RunbookHermes
python scripts/smoke_alert.py --service payment-service --severity critical

# Watch the incident being processed
curl http://localhost:8080/api/incidents | python -m json.tool
```

---

## Common Patterns

### Pattern 1: Auto-triage with evidence, then await approval before action

```python
from runbook_hermes.pipeline import IncidentPipeline

pipeline = IncidentPipeline(
    evidence_backends=["prometheus", "loki", "trace", "deploy"],
    approval_required_for=["rollback", "restart", "config_mutation"],
    skill_generation=True,
)

async with pipeline.run(incident_id="INC-20260501-001") as ctx:
    await ctx.collect_evidence()
    await ctx.analyze()                # model-assisted, uses EvidenceStack
    await ctx.plan_actions()
    await ctx.request_approvals()      # blocks until approved or timeout
    await ctx.execute()
    await ctx.verify_recovery()
    await ctx.generate_skill()
```

### Pattern 2: Dry-run only (no production changes)

```python
pipeline = IncidentPipeline(
    evidence_backends=["prometheus", "loki"],
    approval_required_for=[],          # everything blocked at dry-run
    dry_run_only=True,
    skill_generation=False,
)
```

### Pattern 3: Adding a custom observability adapter

```python
from integrations.observability.base import ObservabilityAdapter

class DatadogAdapter(ObservabilityAdapter):
    async def query_metrics(self, query: str, **kwargs):
        # call Datadog API using DD_API_KEY / DD_APP_KEY env vars
        ...

    async def query_logs(self, query: str, **kwargs):
        ...

# Register it
from runbook_hermes.registry import AdapterRegistry
AdapterRegistry.register("datadog", DatadogAdapter(
    api_key=os.environ["DD_API_KEY"],
    app_key=os.environ["DD_APP_KEY"],
))
```

---

## Troubleshooting

### Observability backends not reachable

Set `USE_MOCK_OBSERVABILITY=true` to fall back to fixtures in `data/runbook_mock/`. Verify real backend URLs and network access before setting to `false`.

### Model summary not appearing

Ensure `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set. The model summary section is silently skipped when no provider is configured — check the settings page at `/settings` for interface status.

### Approval requests timing out

Increase `APPROVAL_TIMEOUT_SECONDS`. For automated testing, set `APPROVAL_REQUIRED=false` to skip the gate entirely.

### EvidenceStack prompt too large

Call `stack.render_for_prompt(max_tokens=N)` with a lower `N`. The stack will trim raw evidence and retain only summaries and evidence IDs.

### Runbook skill not indexed after incident resolution

Confirm `skill_generation=True` is set in the pipeline and that `INCIDENT_MEMORY_DSN` points to a writable location. Run `python scripts/smoke_alert.py` against the demo environment to validate the full pipeline end-to-end.

### Docker demo services not starting

```bash
# Check compose logs
docker compose -f demo/payment_system/docker-compose.yml logs -f

# Reset demo state
python scripts/reset_demo_state.py
```
```
