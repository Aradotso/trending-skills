```markdown
---
name: goal-driven-multi-agent
description: A framework for running multi-agent systems that sustain 100+ hours of continuous effort to solve complex coding or math problems with verifiable criteria
triggers:
  - set up a goal-driven multi-agent system
  - run agents for hours on a complex problem
  - create a master agent with subagents
  - solve a hard coding problem with multiple agents
  - long-running multi-agent task
  - use goal-driven to build a compiler
  - automate a complex task with agents until criteria are met
  - keep agents running until a goal is achieved
---

# Goal-Driven Multi-Agent Framework

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Goal-Driven is a prompt-based framework for running multi-agent systems (Claude Code, Codex, OpenClaw, etc.) that sustain **300+ hours of continuous effort** on extremely complex, verifiable problems. A **master agent** supervises one or more **subagents**, restarting them if they go idle, and halts only when strict success criteria are met.

Ideal use cases: compiler design, theorem proving, database engines, EDA simulation, computational math, and large-scale system architecture.

---

## How It Works

```
while (criteria not met) {
    let subagent work on solving the problem toward the Goal
}
```

| Role | Responsibility |
|------|---------------|
| **Master Agent** | Creates subagents, monitors every 5 min, evaluates criteria, restarts idle agents |
| **Subagent** | Breaks down the goal, works continuously, reports status |

The master agent is the **only** entity that decides if the goal is complete. It does **not** stop unless criteria are fully satisfied or the user intervenes manually.

---

## Installation / Setup

Goal-Driven requires no package installation. It is a **prompt template** you paste into your AI agent tool.

### Step 1 — Copy the prompt template

Grab the full prompt from the repository:

```
https://github.com/lidangzzz/goal-driven
```

### Step 2 — Fill in Goal and Criteria

Open the template and replace the two placeholder blocks:

```
Goal: [[[[[YOUR GOAL HERE]]]]]

Criteria for success: [[[[[YOUR CRITERIA HERE]]]]]
```

### Step 3 — Run in your agent tool

Paste the completed prompt into:
- **Claude Code** — paste into the system prompt or a new conversation
- **Codex CLI** — use as the task description
- **OpenClaw** — configure as the master agent's initial instruction

---

## The Full Prompt Template

```
# Goal-Driven (1 master agent + 1 subagent) System

Here we define a goal-driven multi-agent system for solving any problem.

Goal: [[[[[DEFINE YOUR GOAL HERE]]]]]

Criteria for success: [[[[[DEFINE YOUR CRITERIA FOR SUCCESS HERE]]]]]

Here is the System: The system contains a master agent and a subagent.
You are the master agent, and you need to create 1 subagent to help you
complete the task.

## Subagent's description:
The subagent's goal is to complete the task assigned by the master agent.
The goal defined above is the final and the only goal for the subagent.
The subagent should break down the task into smaller sub-tasks and continue
to work on the task until the criteria for success are met.

## Master agent's description:
1. Create subagents to complete the task.
2. If the subagent finishes or fails, evaluate the result against the
   criteria. If met, stop all subagents. If not, ask the subagent to
   continue.
3. Check subagent activity every 5 minutes. If inactive, verify whether
   the goal is reached. If not, restart a new subagent with the same name.
4. DO NOT STOP THE AGENTS UNTIL THE USER STOPS THEM MANUALLY FROM OUTSIDE.

## Pseudocode:
create a subagent to complete the goal

while (criteria are not met) {
  check the activity of the subagent every 5 minutes
  if (subagent is inactive or declares goal reached) {
    check if goal is reached and verify status
    if (criteria are not met) {
      restart a new subagent with the same name
    } else {
      stop all subagents and end the process
    }
  }
}
```

---

## Real Examples

### Example 1 — TypeScript Compiler in C++

```
Goal: [[[[[Write a TypeScript compiler in C++ that correctly transpiles
TypeScript into JavaScript, including complete documentation and unit tests.]]]]]

Criteria for success: [[[[[Ensure that the TypeScript compiler successfully
compiles and generates 2,000 comprehensive TypeScript test case files covering
as many TypeScript syntax features as possible. Confirm that the C++ TypeScript
compiler correctly transpiles the code into JavaScript. Then, run both the
outputs from this compiler and the official tsc transpiler on Node.js, and
verify that the two resulting JavaScript files produce identical outputs.]]]]]
```

Output: [TypeScript-C-Implementation-by-OnlySpecs](https://github.com/lidangzzz/TypeScript-C-Implementation-by-OnlySpecs) (~100 hours)

---

### Example 2 — SQLite in Rust

```
Goal: [[[[[Implement a SQLite-compatible database engine in Rust that supports
SQL parsing, B-tree storage, and CRUD operations with full test coverage.]]]]]

Criteria for success: [[[[[The Rust implementation must pass the official SQLite
compatibility test suite. Run 500 SQL queries against both the Rust engine and
the reference SQLite binary; all results must be byte-for-byte identical.
Include documentation and benchmarks.]]]]]
```

Output: [sqlite-rust-by-OnlySpecs](https://github.com/lidangzzz/sqlite-rust-by-OnlySpecs) (~30 hours)

---

### Example 3 — Math Theorem Proving

```
Goal: [[[[[Prove the infinitude of twin primes up to a bound of 10^6 using
a verified Lean4 proof script.]]]]]

Criteria for success: [[[[[The Lean4 proof must compile without errors using
`lake build`. All lemmas must be machine-checked. Output a final `#check`
statement confirming the main theorem.]]]]]
```

---

### Example 4 — Database Architecture

```
Goal: [[[[[Design and implement a distributed key-value store in Go with
consistent hashing, replication factor 3, and a REST API.]]]]]

Criteria for success: [[[[[The system must pass a chaos test suite: 1,000
concurrent PUT/GET operations with random node failures. Consistency is
verified by comparing all replica values after each operation. Zero data
loss is acceptable only with acknowledged writes.]]]]]
```

---

## Writing Effective Goals and Criteria

### Goal — Best Practices

```
# Too vague (avoid)
Goal: [[[[[Write a compiler.]]]]]

# Good — specific, scoped, includes deliverables
Goal: [[[[[Write a Lua 5.4 interpreter in Zig that correctly executes Lua
scripts, including a standard library stub, error handling, and a test suite
of at least 500 scripts covering all major language features.]]]]]
```

### Criteria — Best Practices

Criteria must be **mechanically verifiable**. Use one of these patterns:

| Pattern | Example |
|---------|---------|
| Test suite pass rate | "All 2,000 test cases pass" |
| Output diff | "Output of X and reference Y are identical" |
| Compilation success | "`lake build` exits with code 0" |
| Benchmark threshold | "P99 latency < 10ms under 1,000 RPS" |
| Line/file count | "At least 500 documented test files" |

```
# Criteria template
Criteria for success: [[[[[
1. [Quantitative check — e.g., N test files generated]
2. [Correctness check — e.g., outputs match reference implementation]
3. [Build/compilation check — e.g., compiles without warnings]
4. [Optional: performance check]
]]]]]
```

---

## Choosing Problem Types

| ✅ Great fit | ❌ Poor fit |
|-------------|------------|
| Compiler / interpreter | Open-ended creative writing |
| Database engine | Subjective design reviews |
| Theorem proving (Lean4, Coq) | Tasks with no verifiable output |
| Algorithm implementation | Conversational tasks |
| EDA / simulation | Exploratory research |
| Constraint satisfaction | Ambiguous success conditions |

---

## Multi-Subagent Variant

For parallelism, spawn multiple subagents with distinct sub-goals:

```
# Goal-Driven (1 master + 3 subagents)

Goal: [[[[[Build a full-stack web framework in TypeScript...]]]]]

Criteria: [[[[[All integration tests pass, coverage > 90%...]]]]]

Subagents:
- subagent-parser: implements the HTML/template parser
- subagent-router: implements the request routing layer
- subagent-runtime: implements the server runtime and middleware

Master agent monitors all three every 5 minutes. A subagent that goes
inactive is restarted. The master only halts when ALL subagents' combined
output satisfies the criteria.
```

---

## Token and Cost Planning

| Problem complexity | Estimated hours | Estimated tokens |
|-------------------|-----------------|-----------------|
| Small (SQLite-scale) | 20–40 hrs | 5M–15M |
| Medium (compiler) | 60–120 hrs | 20M–60M |
| Large (theorem prover) | 100–300 hrs | 50M–200M+ |

**Before starting:**
- Verify your API plan supports long-running sessions
- Set spending limits/alerts in your provider dashboard
- Use `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc. — never hardcode keys

```bash
# Example: set API key via environment variable
export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
export OPENAI_API_KEY=$OPENAI_API_KEY
```

---

## Common Patterns and Tips

### Pattern: Incremental Criteria

Break criteria into phases so the subagent has checkpoints:

```
Criteria for success: [[[[[
Phase 1 (minimum): The compiler handles all primitive types and functions.
Phase 2 (target):  The compiler passes 1,000 generated test cases.
Phase 3 (stretch): Output is identical to tsc for all 2,000 test cases.
The system halts only when Phase 3 is complete.
]]]]]
```

### Pattern: Self-Verification Script

Include a verification script in the criteria so the agent can self-check:

```
Criteria for success: [[[[[
Run `./verify.sh` from the repo root. This script:
1. Generates 500 test TypeScript files
2. Compiles each with the C++ compiler and with tsc
3. Runs both outputs in Node.js
4. Diffs the results
The criteria are met when `./verify.sh` exits with code 0 and prints
"ALL TESTS PASSED".
]]]]]
```

### Pattern: Artifact Pinning

Tell the master agent where to find the output:

```
Goal: [[[[[...all output must be committed to ./output/ in the repo root.]]]]]

Criteria: [[[[[The master agent verifies by reading ./output/results.json
and confirming the "passed" field equals 2000.]]]]]
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Subagent declares done too early | Tighten criteria — add quantitative thresholds |
| Master agent stops unexpectedly | Add "DO NOT STOP" emphasis to master prompt |
| Subagent loops without progress | Add "report blockers to master agent every 30 min" |
| Context window exhausted | Break goal into phases; use file-based state |
| Inconsistent subagent output | Pin a specific verification script in criteria |
| Too expensive | Reduce scope, use a smaller model for subagents |

### Preventing premature termination

Add this line explicitly to your master agent prompt:

```
IMPORTANT: DO NOT declare success and stop unless the verification script
exits with code 0. Do not trust the subagent's self-reported status alone.
Always re-run verification independently.
```

---

## Key Principles (Summary)

1. **Goal** — one clear, unambiguous objective
2. **Criteria** — machine-verifiable, not subjective
3. **Master agent** — only controller; evaluates criteria independently
4. **Subagent** — works continuously; never trusted to self-terminate
5. **Loop** — `while (criteria not met) { work }` — no exceptions
6. **Do not add to skills/plugins** — this prompt contaminates context windows if reused as a persistent skill in the same session
```
