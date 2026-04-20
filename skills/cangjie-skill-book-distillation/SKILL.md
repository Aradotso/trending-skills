```markdown
---
name: cangjie-skill-book-distillation
description: Distill any book into a set of executable Agent Skills using the RIA-TV++ pipeline — structured extraction, triple verification, and Zettelkasten linking.
triggers:
  - distill a book into skills
  - convert book to agent skills
  - extract methodology from a book
  - create skill pack from book
  - book to skill pipeline
  - run cangjie skill on a book
  - generate RIA skills from reading material
  - build executable skills from book content
---

# Cangjie Skill — Book-to-Agent-Skill Distillation

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Cangjie Skill is a pipeline and template system that distills any high-value book into a structured collection of executable Agent Skills. Instead of summaries or notes, the output is a multi-file skill repository that AI coding agents (Claude Code, Cursor, Codex, etc.) can install and invoke in real workflows.

---

## What It Does

- Reads a book (text, PDF, markdown) and applies the **RIA-TV++ pipeline** in six phases
- Outputs a structured skill pack: `BOOK_OVERVIEW.md`, `INDEX.md`, per-skill `SKILL.md` files, and `test-prompts.json`
- Every skill must pass **Triple Verification** (evidence, predictive power, non-obviousness) — typical pass rate is 25–50%
- Skills include trigger conditions, executable steps, boundary/blind-spots, and Zettelkasten cross-links

---

## Repository Layout

```

cangjie-skill/
├── README.md
├── SKILL.md               ← Meta-skill: the book2skill execution spec
├── methodology/           ← RIA-TV++ phase docs
│   ├── phase1-adler.md
│   ├── phase2-extraction.md
│   ├── phase3-triple-verification.md
│   ├── phase4-ria-construction.md
│   ├── phase5-zettelkasten.md
│   └── phase6-stress-test.md
├── extractors/            ← 5 parallel extractor prompts
│   ├── framework-extractor.md
│   ├── principle-extractor.md
│   ├── case-extractor.md
│   ├── counter-case-extractor.md
│   └── term-extractor.md
└── templates/
    ├── SKILL.md.template
    ├── INDEX.md.template
    └── BOOK_OVERVIEW.md.template

```

---

## Installation / Setup

### Clone the repo

```bash
git clone https://github.com/kangarooking/cangjie-skill.git
cd cangjie-skill
```

### Install as an agent skill (Claude Code / Cursor / Codex)

Copy or symlink `SKILL.md` into your project's `.claude/skills/` (or equivalent) directory:

```bash
# Claude Code
mkdir -p .claude/skills
cp cangjie-skill/SKILL.md .claude/skills/cangjie-skill.md

# Or reference directly in your CLAUDE.md / system prompt
echo "$(cat cangjie-skill/SKILL.md)" >> .claude/CLAUDE.md
```

---

## The RIA-TV++ Pipeline (Six Phases)

### Phase 1 — Adler Analysis (`BOOK_OVERVIEW.md`)

Apply Mortimer Adler's four-step analytical reading:

| Step | Question |
|------|----------|
| Structural | What kind of book is this? What is it about as a whole? |
| Interpretive | What is being said in detail, and how? |
| Critical | Is it true? Is it complete? |
| Applied | What of it? Where can it be used? |

**Output file:** `BOOK_OVERVIEW.md`

```markdown
# BOOK_OVERVIEW — [Book Title]

## Structural Analysis
- Genre / Type:
- Central thesis in one sentence:
- Major parts and their relationship:

## Interpretive Analysis
- Key terms the author defines specially:
- Core propositions:
- Arguments and their logical structure:

## Critical Analysis
- Where the author succeeds:
- Where the author's argument is incomplete or contested:

## Applied Analysis
- Domains where this book's methodology transfers:
- Who benefits most from this book:
```

---

### Phase 2 — Parallel Extraction (5 Extractors)

Run five extractors **in parallel** against the book text. Each extractor targets a specific unit type:

```
extractors/
├── framework-extractor.md     → Repeatable multi-step frameworks
├── principle-extractor.md     → Named principles / heuristics
├── case-extractor.md          → Supporting examples from the book
├── counter-case-extractor.md  → Failure cases / anti-patterns
└── term-extractor.md          → Domain-specific vocabulary
```

**Extractor prompt pattern (framework-extractor):**

```
You are extracting FRAMEWORKS from the following book text.

A framework qualifies if:
1. It has at least 2 named steps or components
2. It is described as repeatable across different situations
3. The author presents it as a deliberate method, not a one-off observation

For each candidate framework, output:
- Name (as the author uses it, or infer a short label)
- Steps / Components (verbatim or near-verbatim from text)
- Page reference or chapter
- Raw quote (≤ 3 sentences)

Do NOT infer frameworks that are not in the text.
Output as JSON array.
```

**Example extractor output (JSON):**

```json
[
  {
    "type": "framework",
    "name": "Two-Track Decision Process",
    "steps": ["Fast intuition check", "Slow checklist verification"],
    "chapter": "Chapter 4",
    "quote": "Never make a major decision on intuition alone; run the checklist...",
    "page_ref": "p.87"
  }
]
```

---

### Phase 3 — Triple Verification

Every candidate from Phase 2 must pass **all three checks**:

```
TV-1: EVIDENCE
  ✓ At least 2 independent supporting instances in the book
  ✓ Instances must span different chapters or contexts (cross-domain)
  ✗ Fail: single anecdote, even if vivid

TV-2: PREDICTIVE POWER
  ✓ The skill can answer a NEW question not explicitly stated in the book
  ✓ Ask: "If I applied this skill to [novel scenario], does it give non-obvious guidance?"
  ✗ Fail: only restates what the book says, no generative power

TV-3: NON-OBVIOUSNESS
  ✓ The content would NOT appear in a generic business/self-help summary
  ✓ It requires reading THIS book to derive
  ✗ Fail: "set clear goals", "communicate openly", etc.
```

**Verification record template:**

```markdown
## Verification: [Candidate Name]

**TV-1 Evidence**
- Instance A: [chapter/page + summary]
- Instance B: [chapter/page + summary]
- Cross-domain? YES / NO

**TV-2 Predictive Power**
- Novel scenario tested: [describe]
- Non-obvious guidance produced: [yes/no + what]

**TV-3 Non-Obviousness**
- Would appear in generic summary? NO
- Unique to this book? YES

**VERDICT:** PASS / FAIL
**Reason if FAIL:**
```

Typical pass rate: **25–50%** of candidates.

---

### Phase 4 — RIA++ Construction

For each verified candidate, build a full skill file using the six-dimension RIA++ structure:

| Dimension | Meaning |
|-----------|---------|
| **R** | Raw reference — verbatim quote(s) from the book |
| **I** | Interpretation — rewritten in your own words |
| **A1** | Application 1 — case from the book |
| **A2** | Application 2 — future trigger scenario (not in the book) |
| **E** | Execution — concrete, ordered steps an agent can follow |
| **B** | Boundary — where this skill breaks down, blind spots, prerequisites |

**SKILL.md template (filled example):**

```markdown
---
skill_id: SK-007
name: inversion-before-commitment
source_book: Poor Charlie's Almanack
chapter: "The Art of Stock Picking"
verified: true
tags: [decision-making, risk, mental-models]
depends_on: [SK-003-checklist-thinking]
contrasts_with: [SK-012-optimism-bias]
---

# Inversion Before Commitment

## R — Raw Reference
> "Invert, always invert. Turn a situation or problem upside down.
>  Look at it backwards." — Charlie Munger, p.211

## I — Interpretation
Before committing to any plan or investment, explicitly ask:
"What would make this fail?" rather than "Why will this succeed?"
The human brain defaults to confirming a thesis; inversion forces
it to search for disconfirming evidence first.

## A1 — Book Case
Munger describes how he evaluates every Berkshire investment by
listing all the ways the business could deteriorate, not by
building DCF models of upside. He found this caught 3 major
near-misses that conventional analysis missed.

## A2 — Future Trigger Scenario
**Trigger:** A developer is about to merge a large refactor
and asks "should I ship this?"
**Inversion applied:** Ask instead: "What are all the ways
this merge could break production?" — list them, then decide.

## E — Execution Steps
1. State the proposal/plan in one sentence.
2. Write the heading: "Ways this could fail / be wrong."
3. List ≥5 failure modes without filtering.
4. For each failure mode, rate likelihood (H/M/L) and impact (H/M/L).
5. If any HH cell exists, address it before proceeding.
6. Only after step 5: evaluate the upside case.

## B — Boundary & Blind Spots
- **Prerequisite:** You must have enough domain knowledge to
  generate realistic failure modes; shallow inversion produces
  generic fears, not useful signals.
- **Does NOT apply:** Time-sensitive decisions under seconds
  (emergency response, real-time trading triggers).
- **Risk of misuse:** Paralysis — inversion without a stopping
  rule can delay indefinitely. Cap the inversion session at 20 min.
- **Blind spot:** Inversion finds known unknowns; it does not
  surface unknown unknowns (use pre-mortem with a diverse team
  for that).
```

---

### Phase 5 — Zettelkasten Linking (`INDEX.md`)

Map relationships between all skills:

```markdown
# INDEX — [Book Title] Skill Pack

## Skill Map

| ID | Name | Tags | Depends On | Contrasts With |
|----|------|------|------------|----------------|
| SK-001 | circle-of-competence | decision-making | — | SK-008 |
| SK-007 | inversion-before-commitment | decision-making, risk | SK-003 | SK-012 |
| SK-003 | checklist-thinking | execution | — | — |

## Dependency Graph (text)

SK-007 → SK-003 (inversion requires checklist to record findings)
SK-001 → SK-007 (define competence boundary before inverting)

## Combo Patterns

**Due-Diligence Stack:** SK-001 → SK-007 → SK-003
Use circle-of-competence to scope the domain, inversion to find
failure modes, checklist to ensure nothing is missed.

**Anti-pattern:** Using SK-012 (optimism bias) without SK-007
leads to confirmation-only analysis.
```

---

### Phase 6 — Stress Testing (`test-prompts.json`)

For each skill, write at least 3 test prompts including ≥1 **decoy** (a scenario where the skill should NOT trigger):

```json
{
  "skill_id": "SK-007",
  "skill_name": "inversion-before-commitment",
  "tests": [
    {
      "id": "T-007-01",
      "type": "positive",
      "prompt": "I'm about to deploy a database migration to production. How should I think about this decision?",
      "expected_trigger": true,
      "expected_behavior": "Agent applies inversion: lists failure modes before evaluating go/no-go."
    },
    {
      "id": "T-007-02",
      "type": "positive",
      "prompt": "We're considering acquiring a small startup. Walk me through how to evaluate it.",
      "expected_trigger": true,
      "expected_behavior": "Agent leads with failure modes of the acquisition before upside modeling."
    },
    {
      "id": "T-007-03",
      "type": "decoy",
      "prompt": "What's the capital of France?",
      "expected_trigger": false,
      "expected_behavior": "Agent answers directly. No inversion applied — factual lookup, no commitment decision."
    },
    {
      "id": "T-007-04",
      "type": "boundary",
      "prompt": "There's a fire alarm going off right now, should I evacuate?",
      "expected_trigger": false,
      "expected_behavior": "Emergency response — inversion does not apply. Agent should say: evacuate first, analyze later."
    }
  ]
}
```

**Run stress tests against an agent:**

```bash
# Pseudo-script: pipe test prompts to your agent and check responses
cat test-prompts.json | jq -r '.tests[].prompt' | while read prompt; do
  echo "--- PROMPT ---"
  echo "$prompt"
  echo "--- RESPONSE ---"
  # Replace with your agent CLI invocation
  your-agent-cli --skill SK-007 --prompt "$prompt"
  echo ""
done
```

Skills that fail stress tests go back to Phase 4 for revision.

---

## Creating a New Skill Pack (End-to-End Example)

```bash
# 1. Create a new skill pack directory
mkdir my-book-skill && cd my-book-skill
cp -r ../cangjie-skill/templates .

# 2. Place your book text
cp /path/to/book.txt ./source.txt

# 3. Run Phase 1 — Adler analysis (example using Claude CLI)
cat templates/BOOK_OVERVIEW.md.template | \
  claude --context source.txt > BOOK_OVERVIEW.md

# 4. Run 5 parallel extractors
for extractor in framework principle case counter-case term; do
  cat ../cangjie-skill/extractors/${extractor}-extractor.md | \
    claude --context source.txt > extracted-${extractor}.json &
done
wait

# 5. Merge and triple-verify (manual review + AI assist)
cat extracted-*.json | jq -s 'add' > candidates.json

# 6. For each passing candidate, generate a SKILL.md
# (use templates/SKILL.md.template as base)

# 7. Build INDEX.md
# (use templates/INDEX.md.template)

# 8. Generate test-prompts.json and run stress tests
```

---

## Existing Skill Packs (Reference Implementations)

| Pack | Source Book | Skills | Repo |
|------|-------------|--------|------|
| buffett-letters-skill | Buffett Shareholder Letters 1957–2023 | 20 | [link](https://github.com/kangarooking/buffett-letters-skill) |
| poor-charlies-almanack-skill | Poor Charlie's Almanack | 12 | [link](https://github.com/kangarooking/poor-charlies-almanack-skill) |
| no-rules-rules-skill | No Rules Rules (Netflix) | 10 | [link](https://github.com/kangarooking/no-rules-rules-skill) |
| cognitive-dividend-skill | 《认知红利》 | 15 | [link](https://github.com/kangarooking/cognitive-dividend-skill) |
| duan-yongping-skill | 段永平投资问答录 | 15 | [link](https://github.com/kangarooking/duan-yongping-skill) |
| huangdi-neijing-skill | 《黄帝内经》素问+灵枢 | 22 | [link](https://github.com/kangarooking/huangdi-neijing-skill) |

Clone any of these to see a complete, production-ready skill pack as a reference.

---

## Common Patterns

### Pattern 1: Invoke a single skill in an agent prompt

```
You have access to the following skill:
[paste contents of SK-007-inversion-before-commitment/SKILL.md]

User question: "Should we rewrite our auth service from scratch?"

Apply the skill's E (Execution) steps explicitly.
```

### Pattern 2: Combo stack (chained skills)

```
Apply the Due-Diligence Stack in order:
1. SK-001 (circle-of-competence): Is this within our domain?
2. SK-007 (inversion): What are all the ways this could fail?
3. SK-003 (checklist): Have we checked every required item?
```

### Pattern 3: Skill selection from INDEX

```
I have the [Book Name] skill pack installed.
User request: [describe scenario]

Step 1: Consult INDEX.md to identify which skills are relevant.
Step 2: Check each skill's B (Boundary) section to confirm applicability.
Step 3: Apply the E (Execution) steps of the selected skill(s).
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Extractor produces too many candidates (>20) | Book is dense OR extractor prompt too permissive | Add stricter qualification criteria to extractor prompt; require ≥3 explicit steps for frameworks |
| All candidates fail TV-3 (non-obviousness) | Book is not methodology-dense enough | Consider whether book is suitable for distillation; may only yield 2–3 skills instead of 10+ |
| Stress test decoy triggers skill incorrectly | A2 trigger scenario is too broadly written | Narrow the trigger condition in the A2 section; add negative examples to the trigger definition |
| Skills feel redundant / overlapping | Phase 5 linking not done | Run Zettelkasten phase; merge overlapping skills or add `contrasts_with` / `depends_on` explicitly |
| Skill too abstract to execute | E steps are vague | Each E step must start with an imperative verb and produce a concrete artifact or decision; revise |

---

## Relationship to the Skill Ecosystem

```
nuwa-skill   → distills PEOPLE (thinking style, expression DNA)
cangjie-skill → distills BOOKS (methodology, frameworks, principles)
darwin-skill  → evolves ANY skill over time
```

Skills produced by cangjie-skill are compatible with nuwa-skill and darwin-skill — they share the same SKILL.md format and can be combined in the same agent context.

---

## License

MIT — see [LICENSE](https://github.com/kangarooking/cangjie-skill/blob/main/LICENSE).
```
