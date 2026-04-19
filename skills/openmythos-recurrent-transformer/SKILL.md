```markdown
---
name: openmythos-recurrent-transformer
description: Build and experiment with Recurrent-Depth Transformer (RDT) models using OpenMythos, a theoretical reconstruction of the Claude Mythos architecture with looped transformers, MLA/GQA attention, and sparse MoE.
triggers:
  - implement a looped transformer model
  - build a recurrent depth transformer
  - use OpenMythos for inference time scaling
  - configure MLA or GQA attention in OpenMythos
  - set up mixture of experts with recurrent blocks
  - train a model with adaptive loop iterations
  - generate text with variable reasoning depth
  - explore compute-adaptive transformer architectures
---

# OpenMythos Recurrent-Depth Transformer

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

OpenMythos is an open-source theoretical implementation of a Recurrent-Depth Transformer (RDT) inspired by the Claude Mythos architecture. It divides computation into three stages: **Prelude** (standard transformer layers run once), a **Recurrent Block** (looped up to `max_loop_iters` times with stable injection), and a **Coda** (standard transformer layers run once). Attention is switchable between MLA (Multi-head Latent Attention) and GQA (Grouped Query Attention), and feed-forward layers use sparse MoE with routed and shared experts.

## Installation

```bash
git clone https://github.com/The-Swarm-Corporation/OpenMythos.git
cd OpenMythos
pip install -r requirements.txt
```

## Core Concepts

### Architecture Flow
```
Input → Embedding
  ↓
[Prelude]          — N standard transformer layers, run once
  ↓
[Recurrent Block]  — looped T times per forward pass
  ↑_________↓        h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
  ↓
[Coda]             — N standard transformer layers, run once
  ↓
Output Logits
```

- `e` = encoded input from Prelude (injected every loop to prevent drift)
- `A`, `B` = learned stable injection parameters (spectral radius ρ(A) < 1 enforced)
- More loops at inference = deeper implicit reasoning (no extra parameters)

### Attention Types
- **MLA** (`"mla"`): Multi-head Latent Attention — uses KV LoRA compression, separate RoPE and NoPE head dimensions
- **GQA** (`"gqa"`): Grouped Query Attention — fewer KV heads than Q heads, simpler config

## Configuration Reference

### `MythosConfig` Parameters

| Parameter | Type | Description |
|---|---|---|
| `vocab_size` | int | Vocabulary size |
| `dim` | int | Model hidden dimension |
| `n_heads` | int | Number of attention query heads |
| `n_kv_heads` | int | Number of KV heads (GQA ratio = n_heads/n_kv_heads) |
| `max_seq_len` | int | Maximum sequence length |
| `max_loop_iters` | int | Maximum recurrent loop iterations |
| `prelude_layers` | int | Number of Prelude transformer layers |
| `coda_layers` | int | Number of Coda transformer layers |
| `n_experts` | int | Total number of MoE routed experts |
| `n_shared_experts` | int | Always-active shared experts |
| `n_experts_per_tok` | int | Top-K experts selected per token |
| `expert_dim` | int | Hidden dim inside each expert FFN |
| `lora_rank` | int | LoRA rank for injection parameters |
| `attn_type` | str | `"mla"` or `"gqa"` |
| `kv_lora_rank` | int | MLA only: KV compression rank |
| `q_lora_rank` | int | MLA only: Q compression rank |
| `qk_rope_head_dim` | int | MLA only: RoPE head dimension |
| `qk_nope_head_dim` | int | MLA only: NoPE head dimension |
| `v_head_dim` | int | MLA only: Value head dimension |

## Usage Patterns

### Pattern 1: GQA Model (Simpler Config)

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_kv_heads=2,          # GQA: 4x fewer KV heads
    max_seq_len=2048,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=2,
    n_experts_per_tok=2,
    expert_dim=256,
    lora_rank=16,
    attn_type="gqa",
)

model = OpenMythos(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass with 4 reasoning loops
ids = torch.randint(0, cfg.vocab_size, (2, 64))
logits = model(ids, n_loops=4)
print(f"Logits shape: {logits.shape}")  # (2, 64, 32000)
```

### Pattern 2: MLA Model (More Expressive)

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_kv_heads=8,           # MLA uses full heads
    max_seq_len=2048,
    max_loop_iters=16,
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=2,
    n_experts_per_tok=2,
    expert_dim=256,
    lora_rank=16,
    attn_type="mla",
    # MLA-specific compression dims
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_rope_head_dim=32,
    qk_nope_head_dim=32,
    v_head_dim=32,
)

model = OpenMythos(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 128))

# More loops = deeper implicit reasoning
logits_shallow = model(ids, n_loops=2)
logits_deep    = model(ids, n_loops=16)
print(f"Shallow logits: {logits_shallow.shape}")
print(f"Deep logits:    {logits_deep.shape}")
```

### Pattern 3: Text Generation with Variable Depth

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=50257,
    dim=768,
    n_heads=12,
    n_kv_heads=4,
    max_seq_len=1024,
    max_loop_iters=12,
    prelude_layers=2,
    coda_layers=2,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=512,
    lora_rank=32,
    attn_type="gqa",
)

model = OpenMythos(cfg)
model.eval()

prompt_ids = torch.randint(0, cfg.vocab_size, (1, 16))

# Simple task: fewer loops
with torch.no_grad():
    easy_output = model.generate(
        prompt_ids,
        max_new_tokens=32,
        n_loops=2,          # shallow reasoning
    )

# Complex task: more loops (same model, more compute)
with torch.no_grad():
    hard_output = model.generate(
        prompt_ids,
        max_new_tokens=32,
        n_loops=12,         # deep reasoning
    )

print(f"Easy output shape: {easy_output.shape}")
print(f"Hard output shape: {hard_output.shape}")
```

### Pattern 4: Stability Verification

Always verify the spectral radius constraint after initialization and training:

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=8,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="gqa",
)

model = OpenMythos(cfg)

# Check injection matrix spectral radius — must be < 1
A = model.recurrent.injection.get_A()
spectral_radius = A.max().item()
print(f"Spectral radius ρ(A): {spectral_radius:.6f}")
assert spectral_radius < 1.0, "UNSTABLE: spectral radius >= 1!"
print("✓ Stability constraint satisfied")
```

### Pattern 5: Scaling Experiment — Loops vs Quality

```python
import torch
import torch.nn.functional as F
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=64,
    max_loop_iters=16,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="gqa",
)

model = OpenMythos(cfg)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (1, 32))
targets = torch.randint(0, cfg.vocab_size, (1, 32))

results = {}
with torch.no_grad():
    for n_loops in [1, 2, 4, 8, 16]:
        logits = model(ids, n_loops=n_loops)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            targets.view(-1)
        )
        results[n_loops] = loss.item()
        print(f"n_loops={n_loops:2d} → loss={loss.item():.4f}")

# Expect diminishing returns (saturating exponential decay)
print("\nDelta losses (should decrease):")
loop_counts = sorted(results.keys())
for i in range(1, len(loop_counts)):
    delta = results[loop_counts[i-1]] - results[loop_counts[i]]
    print(f"  {loop_counts[i-1]}→{loop_counts[i]} loops: Δloss={delta:.4f}")
```

### Pattern 6: Training Loop with Stability Monitoring

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=10000,
    dim=512,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=256,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=256,
    lora_rank=16,
    attn_type="gqa",
)

model = OpenMythos(cfg)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

def train_step(model, optimizer, input_ids, target_ids, n_loops=4):
    model.train()
    optimizer.zero_grad()
    
    logits = model(input_ids, n_loops=n_loops)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        target_ids.view(-1),
    )
    loss.backward()
    
    # Gradient clipping recommended for looped models
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def check_stability(model):
    A = model.recurrent.injection.get_A()
    rho = A.max().item()
    return rho

# Example training iterations
for step in range(10):
    batch_size, seq_len = 4, 64
    input_ids  = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    # Curriculum: start with fewer loops, increase over training
    n_loops = min(2 + step // 3, cfg.max_loop_iters)
    
    loss = train_step(model, optimizer, input_ids, target_ids, n_loops=n_loops)
    rho  = check_stability(model)
    
    print(f"Step {step:3d} | loss={loss:.4f} | ρ(A)={rho:.4f} | loops={n_loops}")
    
    if rho >= 1.0:
        print("WARNING: Spectral radius constraint violated!")
```

### Pattern 7: MLA vs GQA Comparison

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

def build_model(attn_type: str) -> OpenMythos:
    base = dict(
        vocab_size=8000,
        dim=256,
        n_heads=8,
        max_seq_len=128,
        max_loop_iters=6,
        prelude_layers=1,
        coda_layers=1,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=8,
        attn_type=attn_type,
    )
    if attn_type == "gqa":
        cfg = MythosConfig(**base, n_kv_heads=2)
    else:  # mla
        cfg = MythosConfig(
            **base,
            n_kv_heads=8,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
        )
    return OpenMythos(cfg)

for attn_type in ["gqa", "mla"]:
    model = build_model(attn_type)
    param_count = sum(p.numel() for p in model.parameters())
    
    ids = torch.randint(0, 8000, (2, 32))
    logits = model(ids, n_loops=4)
    
    rho = model.recurrent.injection.get_A().max().item()
    
    print(f"{attn_type.upper()}: params={param_count:,} | "
          f"logits={logits.shape} | ρ(A)={rho:.4f}")
```

## Key API Reference

### `OpenMythos(config)`
Main model class.

**`forward(input_ids, n_loops=None)`**
- `input_ids`: `LongTensor` of shape `(batch, seq_len)`
- `n_loops`: int, number of recurrent iterations (default: `config.max_loop_iters`)
- Returns: `FloatTensor` of shape `(batch, seq_len, vocab_size)`

**`generate(input_ids, max_new_tokens, n_loops=None)`**
- `input_ids`: `LongTensor` of shape `(batch, seq_len)`
- `max_new_tokens`: int
- `n_loops`: int, loops per forward step during generation
- Returns: `LongTensor` of shape `(batch, seq_len + max_new_tokens)`

### `model.recurrent.injection.get_A()`
Returns the learned injection matrix `A`. Check `A.max().item() < 1.0` for stability.

## Common Patterns & Best Practices

### Loop Curriculum During Training
Start with fewer loops and increase gradually — helps with initial stability:
```python
n_loops = min(max_loops, 2 + global_step // 1000)
```

### Inference-Time Scaling
Use fewer loops for simple/fast tasks, more for complex reasoning — same weights, adaptive compute:
```python
# Classification / simple completion
logits = model(ids, n_loops=2)

# Multi-step reasoning / hard math
logits = model(ids, n_loops=model.cfg.max_loop_iters)
```

### Gradient Clipping
Always clip gradients when training looped models — prevents the rare instability when gradients backpropagate through many loop unrolls:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### MoE Load Balancing
The router uses dynamic bias adjustment — no special loss term needed, but monitor expert utilization during training to detect collapse.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ρ(A) >= 1.0` after training | Injection constraint violated | Lower learning rate; the constraint should be enforced by construction — check model version |
| Loss NaN / explosion | Residual stream diverging | Enable grad clipping (`max_norm=1.0`); reduce `n_loops` early in training |
| `RuntimeError` on MLA config | Missing MLA-specific params | Ensure `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` are all set |
| OOM with high `n_loops` | Memory scales with loop unrolls during training | Use `torch.no_grad()` for eval; reduce batch size during high-loop training |
| GQA `n_kv_heads` error | Must divide `n_heads` evenly | Ensure `n_heads % n_kv_heads == 0` |
| Slow generation | Generating with max loops | Reduce `n_loops` in `generate()` for faster inference |

## Project Structure

```
OpenMythos/
├── open_mythos/
│   └── main.py          # OpenMythos, MythosConfig, all sub-modules
├── docs/
│   └── open_mythos.md   # Full API reference
├── requirements.txt
└── README.md
```
```
