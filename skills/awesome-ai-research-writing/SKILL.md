```markdown
---
name: awesome-ai-research-writing
description: A curated collection of battle-tested prompts and agent skills for AI-assisted academic research writing, translation, and polishing
triggers:
  - help me polish my research paper
  - translate my paper from Chinese to English
  - improve my academic writing
  - remove AI tone from my paper
  - help me write a paper abstract
  - review my paper like a reviewer
  - shrink or expand my paper section
  - help me with LaTeX academic writing
---

# Awesome AI Research Writing

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A community-maintained prompt library and agent skill collection for AI-assisted academic research writing. Sourced from researchers at MSRA, ByteDance Seed, Shanghai AI Lab, Peking University, USTC, and SJTU. Covers translation, polishing, logic checking, de-AI-ification, figure/table captions, reviewer simulation, and more.

---

## What This Project Does

This repository provides:
- **Prompt templates** for common writing tasks (Chinese↔English translation, condensing, expanding, logic checking, LaTeX polishing)
- **Agent skills** that extend AI coding assistants to handle academic paper workflows
- **Best practices** from top research institutions, ready to copy-paste

The prompts are designed for LaTeX (English papers) and Word (Chinese papers) environments.

---

## Installation / Setup

This is a prompt + skill library — no package to install. Use it in two ways:

### Option A: Copy-Paste Prompts Directly
Open any LLM chat (Claude, GPT-4, Gemini, etc.) and paste the relevant prompt from the sections below.

### Option B: Install as an Agent Skill
Save this `SKILL.md` to your project root or agent skills directory:

```bash
# For Claude Code
cp SKILL.md .claude/skills/awesome-ai-research-writing.md

# For Cursor
cp SKILL.md .cursor/skills/awesome-ai-research-writing.md

# For general agent use
mkdir -p .agent/skills && cp SKILL.md .agent/skills/
```

---

## Core Prompt Templates

### 1. Chinese → English (LaTeX)

Use this when drafting a section in Chinese and need publication-ready English LaTeX.

```
# Role
You are an assistant with dual identity: a top academic writing expert and a senior conference reviewer (ICML/ICLR). You have zero tolerance for logical gaps and language flaws.

# Task
Translate and polish the provided [Chinese Draft] into an [English academic paper fragment].

# Constraints
1. Visual & Layout:
   - Avoid bold, italic, or unnecessary quotes.
   - Keep LaTeX source clean.

2. Style & Logic:
   - Rigorous logic, precise wording, concise and coherent expression.
   - Prefer common words over obscure vocabulary.
   - Avoid em-dashes (—); use clauses or appositives instead.
   - No \item lists — use coherent paragraphs.
   - Remove "AI flavor" — write naturally, avoid mechanical connectives.

3. Tense:
   - Present tense for methods, architectures, and experimental conclusions.
   - Past tense only for specific historical events.

4. Output Format:
   - Part 1 [LaTeX]: English LaTeX only. Escape special characters (95\%, model\_v1, R\&D). Keep math formulas with $ intact.
   - Part 2 [Translation]: Chinese back-translation for logic verification.
   - No extra commentary.

# Input
[Paste your Chinese draft here]
```

---

### 2. English → Chinese (LaTeX input)

Use when quickly reading and understanding a LaTeX paper section.

```
# Role
You are a senior academic translator in computer science, helping researchers quickly understand complex English paper paragraphs.

# Task
Translate the provided [English LaTeX code fragment] into fluent, readable [Chinese text].

# Constraints
1. LaTeX Cleaning:
   - Delete all \cite{...}, \ref{...}, \label{...} — do not translate them.
   - For \textbf{text}, \emph{text}: translate only the inner text.
   - Convert math to readable natural language (e.g., $\alpha$ → alpha, \frac{a}{b} → a/b).

2. Translation Principles:
   - Strict literal translation — no polishing, rewriting, or logic optimization.
   - Preserve sentence structure to allow easy back-reference.
   - Reflect source errors faithfully — do not auto-correct.

3. Output:
   - Pure Chinese text only. No LaTeX syntax.

# Input
[Paste your English LaTeX code here]
```

---

### 3. Chinese → Chinese (Word / Chinese Paper)

For Chinese academic papers written in Word.

```
# Role
You are a senior editor of Chinese academic journals (e.g., Journal of Computer Science, Journal of Software) and a top conference reviewer. You excel at reconstructing fragmented, colloquial text into rigorous, well-crafted academic prose.

# Task
Rewrite the provided [Chinese Draft] into a logically coherent, academically standard [paper body paragraph].

# Constraints
1. Format (Word-compatible):
   - Pure text output: NO Markdown bold/italic/heading symbols.
   - Punctuation: Chinese full-width punctuation (，。；：""). Add spaces around English terms.

2. Logic & Structure:
   - Identify the logical thread; reconnect loose sentences.
   - Convert lists into coherent paragraphs.
   - One paragraph = one core idea.

3. Style:
   - Highly formal. Convert colloquial → written (e.g., "不管A还是B" → "无论A抑或B").
   - Objective and neutral tone.
   - Keep technical terms in English (Transformer, CNN, Few-shot).

4. Output:
   - Part 1 [Refined Text]: Rewritten paragraph.
   - Part 2 [Logic flow]: Brief explanation of restructuring decisions.

# Input
[Paste your Chinese draft, scattered ideas, or bullet points here]
```

---

### 4. Condense (Shorten by ~5–15 words)

```
# Role
You are a top academic editor specializing in conciseness — reducing word count without losing any information.

# Task
Slightly condense the provided [English LaTeX code fragment].

# Constraints
- Target: reduce ~5–15 words.
- Do NOT remove core info, technical details, or experimental parameters.
- Techniques: convert clauses to phrases, eliminate filler ("in order to" → "to").
- No bold/italic/quotes. No em-dashes. No itemization. Keep math formulas intact.

# Output:
- Part 1 [LaTeX]: Condensed English LaTeX (escape special chars: \%, \_, \&).
- Part 2 [Translation]: Chinese back-translation to verify information integrity.
- Part 3 [Modification Log]: Chinese summary of changes made.

# Input
[Paste your English LaTeX code here]
```

---

### 5. Expand (Lengthen by ~5–15 words)

```
# Role
You are a top academic editor specializing in logical fluency — deepening content and strengthening logical connections.

# Task
Slightly expand the provided [English LaTeX code fragment].

# Constraints
- Target: add ~5–15 words.
- No padding: do NOT add meaningless adjectives or repetitive filler.
- Techniques: surface implicit conclusions/premises/causality, add connectives (Furthermore, Notably), upgrade simple descriptions to precise academic expressions.
- No bold/italic/quotes. No em-dashes. No itemization.

# Output:
- Part 1 [LaTeX]: Expanded English LaTeX (escape special chars).
- Part 2 [Translation]: Chinese back-translation to verify new logic matches original intent.
- Part 3 [Modification Log]: Chinese summary of additions.

# Input
[Paste your English LaTeX code here]
```

---

### 6. Polish English Paper Expression

```
# Role
You are a native English speaker with a PhD in Computer Science and 10+ years of experience reviewing for NeurIPS, ICML, and ICLR.

# Task
Polish the provided [English LaTeX code] for publication quality.

# Constraints
- Fix grammar, word choice, and sentence flow.
- Maintain all original technical content and meaning.
- Present tense for methods and results. No em-dashes. No Markdown formatting.
- Escape LaTeX special characters in output.

# Output:
- Part 1 [LaTeX]: Polished LaTeX (English only).
- Part 2 [Changes]: Chinese bullet list of key changes made.

# Input
[Paste your English LaTeX code here]
```

---

### 7. Remove "AI Flavor" — LaTeX English

Use when text feels robotic, over-structured, or obviously LLM-generated.

```
# Role
You are a seasoned academic author who writes papers that read like they were written by a thoughtful human researcher.

# Task
Rewrite the provided [English LaTeX code] to remove "AI flavor" while preserving all technical content.

# What "AI flavor" means:
- Overused transitions: "Furthermore", "Moreover", "It is worth noting that", "In summary"
- Excessive parallel structure and repetitive sentence patterns
- Hollow intensifiers: "significant", "crucial", "novel", "robust" (without justification)
- Over-enumeration using \item lists
- Unnatural hedging: "It can be observed that", "As can be seen"

# Constraints:
- Vary sentence length and structure naturally.
- Keep all technical details, numbers, and citations intact.
- Output clean LaTeX — no Markdown, no added formatting.
- Escape special characters (\%, \_, \&).

# Output:
- Part 1 [LaTeX]: De-AI-ified English LaTeX.
- Part 2 [Translation]: Chinese back-translation for verification.
- Part 3 [Modifications]: Chinese list of specific changes.

# Input
[Paste your English LaTeX code here]
```

---

### 8. Remove "AI Flavor" — Word Chinese

```
# Role
You are a senior editor at a top Chinese CS journal with a sharp eye for AI-generated text patterns in Chinese academic writing.

# Task
Rewrite the provided [Chinese academic text] to remove AI flavor while maintaining academic standards.

# What "AI flavor" means in Chinese:
- 程式化过渡词：首先、其次、此外、综上所述、值得注意的是
- 过度使用并列结构，句式单调重复
- 空洞的强调词：显著、关键、创新性、鲁棒（无数据支撑）
- 机械的"本文提出……本文验证……本文证明"句式

# Constraints:
- Pure text output — NO Markdown symbols (no **, no ##).
- Chinese full-width punctuation only.
- Preserve all technical content and data.

# Output:
- Part 1 [Refined Text]: De-AI-ified Chinese paragraph.
- Part 2 [Modifications]: Brief list of what was changed and why.

# Input
[Paste your Chinese academic text here]
```

---

### 9. Reviewer Simulation

Use before submission to anticipate reviewer critiques.

```
# Role
You are a senior Area Chair at NeurIPS/ICML/ICLR who has reviewed 500+ papers. You are thorough, critical, and fair.

# Task
Review the provided paper section as if writing an official review.

# Review Dimensions:
1. Clarity & Writing: Is the contribution clearly stated? Are claims well-supported?
2. Technical Soundness: Are there logical gaps? Unjustified assumptions?
3. Experimental Rigor: Are baselines appropriate? Are ablations sufficient?
4. Novelty: Is the contribution incremental or significant?
5. Weaknesses: List concrete, actionable weaknesses.
6. Questions for Authors: List 3–5 clarifying questions.

# Output Format:
- Summary (2–3 sentences)
- Strengths (bullet list)
- Weaknesses (bullet list, be specific)
- Questions (numbered list)
- Preliminary Score: [1–10] with justification

# Input
[Paste your paper section or full paper here]
```

---

### 10. Generate Figure Captions

```
# Role
You are an expert in scientific figure design and caption writing for top-tier CS venues.

# Task
Generate a publication-quality figure caption for the described figure.

# Constraints:
- Start with a bold short title summarizing the figure (LaTeX: \textbf{Title.})
- Follow with 2–4 sentences explaining: what is shown, key takeaway, and how to read it.
- Use present tense. Be specific about what the figure demonstrates.
- Reference subfigures as (a), (b), (c) if applicable.
- Keep total length under 80 words.

# Input
[Describe your figure: what it shows, axes, key results, subfigures if any]
```

---

### 11. Generate Table Captions

```
# Role
You are an expert academic writer specializing in results presentation.

# Task
Generate a publication-quality table caption.

# Constraints:
- Start with \textbf{Short descriptive title.}
- Explain what the table compares, the metric(s) used, and the key finding.
- Note any special symbols (↑/↓ for better/worse, bold for best, underline for second-best).
- Present tense. Under 60 words.

# Input
[Describe your table: what it compares, metrics, datasets, key results]
```

---

### 12. Experimental Results Analysis

```
# Role
You are a research scientist who excels at interpreting experimental results and connecting numbers to scientific insights.

# Task
Analyze the provided experimental results and generate an academic-style analysis paragraph.

# Constraints:
- Go beyond restating numbers — explain WHY results occur.
- Connect results to the paper's core claims.
- Acknowledge limitations or surprising findings honestly.
- Write in present tense, coherent paragraph form (no bullet lists).
- Output LaTeX-compatible text.

# Input
[Paste your experimental results table or data here, with brief context about your method]
```

---

## Common Workflows

### Full Paper Section Workflow (Chinese → English Publication)

```
1. Draft in Chinese (free-form, don't worry about polish)
2. Use Prompt #1 (Chinese→English) to get initial LaTeX
3. Use Prompt #7 (Remove AI Flavor) to naturalize the output
4. Use Prompt #6 (Polish Expression) for final grammar pass
5. Use Prompt #4 or #5 (Condense/Expand) to hit page limits
6. Use Prompt #9 (Reviewer Simulation) before submission
```

### Quick Read Workflow (Reading Others' Papers)

```
1. Copy LaTeX section from paper
2. Use Prompt #2 (English→Chinese) for fast comprehension
3. Use Prompt #9 (Reviewer) to quickly identify the paper's weaknesses
```

### Figure & Table Caption Workflow

```
1. Describe your figure/table in plain language
2. Use Prompt #10 (Figure Caption) or #11 (Table Caption)
3. Adjust numbers/specifics manually
4. Run through Prompt #6 (Polish) if needed
```

---

## Model Recommendations

| Task | Recommended Model | Notes |
|------|-------------------|-------|
| Chinese→English translation | Claude 3.5 Sonnet / GPT-4o | Best bilingual quality |
| Logic checking / Review simulation | Claude 3.5 Sonnet | Strong reasoning |
| Remove AI flavor | Claude 3 Opus / GPT-4o | More creative rewriting |
| Quick translation/condensing | GPT-4o-mini / Claude Haiku | Fast and cheap |
| Math-heavy sections | GPT-4o | Better LaTeX handling |

---

## Tips for Best Results

### Do
- ✅ Paste complete, self-contained paragraphs (not single sentences)
- ✅ Include surrounding context when asking for logic checks
- ✅ Specify your target venue (NeurIPS, CVPR, ACL) for reviewer simulation
- ✅ Always verify the Chinese back-translation (Part 2) to catch meaning drift
- ✅ Use the Modification Log (Part 3) to understand what changed

### Don't
- ❌ Don't paste the entire paper at once — work section by section
- ❌ Don't skip the back-translation check — LLMs sometimes subtly alter meaning
- ❌ Don't use condense/expand prompts for >500 word sections (results degrade)
- ❌ Don't rely solely on AI reviewer simulation — get real human feedback too

---

## Troubleshooting

**Output contains Chinese in LaTeX part:**
→ Add to your prompt: "CRITICAL: Part 1 must be 100% English. Re-check before outputting."

**LaTeX special characters not escaped:**
→ Remind the model: "Remember to escape all %, _, &, # characters in LaTeX output."

**Output too long / too short after condense/expand:**
→ Adjust the word count target: "reduce by ~25 words" or "add ~30 words"

**AI flavor still detectable after de-AI pass:**
→ Run the de-AI prompt twice, or manually identify specific phrases and ask: "Rewrite this sentence to not start with 'Furthermore' and avoid parallel structure."

**Reviewer simulation is too generic:**
→ Specify: "Focus your review on Section 4 (Experiments). Be specific about whether the ablation study in Table 3 is sufficient."

**Chinese output has Markdown formatting in Word:**
→ Add: "CRITICAL: Output plain text only. Absolutely no **, ##, or any Markdown symbols."

---

## Contributing

To add your own prompts to the community collection:
1. Fork the repository at `github.com/Leey21/awesome-ai-research-writing`
2. Add your prompt following the existing template structure
3. Include: Role, Task, Constraints, Output Format, and a sample Input
4. Submit a PR with a brief description of the use case

---

## Quick Reference Card

```
中→英 (LaTeX)     : Prompt #1  — Full translation + back-check
英→中 (LaTeX)     : Prompt #2  — Quick comprehension
中→中 (Word)      : Prompt #3  — Chinese paper rewrite
缩写              : Prompt #4  — Shorten ~5-15 words
扩写              : Prompt #5  — Expand ~5-15 words
英文润色          : Prompt #6  — Grammar + flow polish
去AI味 (英/LaTeX) : Prompt #7  — Remove robotic patterns
去AI味 (中/Word)  : Prompt #8  — Remove Chinese AI patterns
Reviewer视角      : Prompt #9  — Pre-submission review
图标题            : Prompt #10 — Figure captions
表标题            : Prompt #11 — Table captions
实验分析          : Prompt #12 — Results analysis paragraph
```
```
