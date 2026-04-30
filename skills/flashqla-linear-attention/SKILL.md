```markdown
---
name: flashqla-linear-attention
description: High-performance linear attention kernel library for GDN Chunked Prefill built on TileLang, achieving 2-3x speedup over FLA Triton kernels on NVIDIA Hopper GPUs
triggers:
  - use FlashQLA for linear attention
  - implement gated delta rule attention
  - chunk gated delta rule forward backward
  - linear attention kernel optimization
  - FlashQLA chunked prefill
  - fast linear attention on Hopper GPU
  - GDN attention kernel with TileLang
  - QwenLM flash linear attention
---

# FlashQLA Linear Attention Kernel Library

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

FlashQLA is a high-performance linear attention kernel library built on [TileLang](https://github.com/tile-ai/tilelang), providing optimized forward and backward passes for GDN (Gated Delta-rule Network) Chunked Prefill. It achieves 2-3× forward speedup and 2× backward speedup over FLA Triton kernels on NVIDIA Hopper (SM90+) GPUs.

## Requirements

- GPU: SM90 or above (NVIDIA Hopper or newer)
- CUDA: 12.8 or above
- PyTorch: 2.8 or above

## Installation

```bash
git clone https://github.com/QwenLM/FlashQLA.git
cd FlashQLA
pip install -v .
```

For benchmarking and testing, install comparison baselines:

```bash
pip install flash_linear_attention==0.5.0
pip install flashinfer-python==0.6.9
```

## Core API

### High-Level API: `chunk_gated_delta_rule`

The primary entry point for GDN chunked prefill attention:

```python
import torch
from flash_qla import chunk_gated_delta_rule

# Tensor shapes:
# q, k:     [B, T, H_q, K]   - query and key
# v:        [B, T, H_v, V]   - value
# g:        [B, T, H_v]      - gate (exponential decay)
# beta:     [B, T, H_v]      - delta rule beta coefficient
# initial_state: [B, H_v, K, V] - optional initial recurrent state

B, T, H_q, K = 2, 4096, 32, 128
H_v, V = 32, 128

q = torch.randn(B, T, H_q, K, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, H_q, K, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, H_v, V, dtype=torch.bfloat16, device='cuda')
g = torch.randn(B, T, H_v, dtype=torch.bfloat16, device='cuda')
beta = torch.randn(B, T, H_v, dtype=torch.bfloat16, device='cuda')
scale = K ** -0.5

o, final_state = chunk_gated_delta_rule(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=scale,
    initial_state=None,        # optional: [B, H_v, K, V]
    output_final_state=True,   # whether to return final recurrent state
    cu_seqlens=None,           # optional: for variable-length sequences
)
# o: [B, T, H_v, V]
# final_state: [B, H_v, K, V]
```

### High-Level API with Variable-Length Sequences

For batches with variable sequence lengths (packed/ragged batches):

```python
import torch
from flash_qla import chunk_gated_delta_rule

# cu_seqlens: cumulative sequence lengths, shape [B+1], dtype int32
# Example: batch of 3 sequences with lengths [512, 1024, 768]
seq_lens = [512, 1024, 768]
total_tokens = sum(seq_lens)
cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).numpy()),
                           dtype=torch.int32, device='cuda')

H_q, K, H_v, V = 32, 128, 32, 128

# Packed tensors: [1, total_tokens, H, D]
q = torch.randn(1, total_tokens, H_q, K, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, total_tokens, H_q, K, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, total_tokens, H_v, V, dtype=torch.bfloat16, device='cuda')
g = torch.randn(1, total_tokens, H_v, dtype=torch.bfloat16, device='cuda')
beta = torch.randn(1, total_tokens, H_v, dtype=torch.bfloat16, device='cuda')

o, final_state = chunk_gated_delta_rule(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=K ** -0.5,
    output_final_state=True,
    cu_seqlens=cu_seqlens,
)
```

### Low-Level API: Separate Forward and Backward

For custom training loops or gradient checkpointing:

```python
from flash_qla import chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd

# Forward pass — returns intermediate tensors needed for backward
g_out, A, o, h, final_state = chunk_gated_delta_rule_fwd(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=scale,
    initial_state=h0,       # optional initial state [B, H_v, K, V]
    cu_seqlens=cu_seqlens,  # optional
)
# g_out: processed gate tensor
# A: intra-chunk attention matrix (saved for backward)
# o: output [B, T, H_v, V]
# h: intermediate hidden states
# final_state: [B, H_v, K, V]

# Backward pass
dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    A=A,           # from forward pass
    do=do,         # gradient of output, [B, T, H_v, V]
    dht=dht,       # gradient of final state, optional [B, H_v, K, V]
    scale=scale,
    initial_state=h0,
    cu_seqlens=cu_seqlens,
)
# Returns: dq, dk, dv, dbeta, dg, dinitial_state
```

## Integration with PyTorch Autograd

```python
import torch
import torch.nn as nn
from flash_qla import chunk_gated_delta_rule

class GDNAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.g_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
    
    def forward(self, x, initial_state=None, cu_seqlens=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)
        g = torch.sigmoid(self.g_proj(x))   # [B, T, H] — gate in (0,1)
        beta = torch.sigmoid(self.beta_proj(x))  # [B, T, H]
        
        # Convert to bfloat16 for kernel
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        g, beta = g.to(torch.bfloat16), beta.to(torch.bfloat16)
        
        o, final_state = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        
        o = o.reshape(B, T, H * D).to(x.dtype)
        return self.out_proj(o), final_state
```

## Head Size Configurations (TP Settings)

FlashQLA is optimized for the head configurations used by Qwen3.5/Qwen3.6 family:

| Head Dim (h_k,v) | TP Setting |
|-----------------|------------|
| 64              | TP1        |
| 48              | TP2        |
| 32              | TP3        |
| 24              | TP4        |
| 16              | TP6        |
| 8               | TP8        |

```python
# Example: TP2 configuration (H_q=H_v=48 head dim)
q = torch.randn(B, T, num_heads, 48, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, num_heads, 48, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, num_heads, 48, dtype=torch.bfloat16, device='cuda')
```

## Running Tests

```bash
cd tests

# Development tests (quick sanity check)
python test_gdr.py --set develop

# Variable-length sequence tests with 32 heads
python test_gdr.py --set varlen --num-heads 32

# Profiling tests
python test_gdr.py --set profile --num-heads 32

# Production accuracy tests (compare against float32 reference)
python test_gdr.py --set product --ref-dtype float32 --num-heads 32
```

## Running Benchmarks

```bash
cd benchmark

# Benchmark against FLA Triton and FlashInfer baselines
python bench_gated_delta_rule.py
```

Benchmark results on H200 are in `benchmark/benchmark_results_H200.txt`.

## Common Patterns

### Autoregressive Inference with State Caching

```python
from flash_qla import chunk_gated_delta_rule

def autoregressive_step(q, k, v, g, beta, cached_state, scale):
    """Process a single chunk and update recurrent state."""
    o, new_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=cached_state,
        output_final_state=True,
    )
    return o, new_state

# Initialize state
B, H, K, V = 1, 32, 64, 64
state = torch.zeros(B, H, K, V, dtype=torch.bfloat16, device='cuda')

# Process chunks sequentially
chunk_size = 512
for chunk_tokens in token_chunks:
    o_chunk, state = autoregressive_step(
        q=chunk_tokens['q'],
        k=chunk_tokens['k'],
        v=chunk_tokens['v'],
        g=chunk_tokens['g'],
        beta=chunk_tokens['beta'],
        cached_state=state,
        scale=K ** -0.5,
    )
```

### Gradient Checkpointing with Low-Level API

```python
import torch
from flash_qla import chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd

class GDNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, h0):
        g_out, A, o, h, final_state = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=scale, initial_state=h0
        )
        ctx.save_for_backward(q, k, v, g, beta, A, h0)
        ctx.scale = scale
        return o, final_state
    
    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, A, h0 = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q, k, v, g, beta, A, do,
            dht=dht, scale=ctx.scale, initial_state=h0
        )
        return dq, dk, dv, dg, db, None, dh0
```

## Troubleshooting

### GPU Compatibility Error
```
# FlashQLA requires SM90+; verify your GPU:
python -c "import torch; print(torch.cuda.get_device_capability())"
# Must return (9, 0) or higher
```

### CUDA Version Issues
```bash
# Check CUDA version (requires 12.8+)
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Import Error After Installation
```bash
# Reinstall with verbose output to see compilation errors
pip uninstall flash_qla -y
cd FlashQLA
pip install -v .
```

### dtype Issues
```python
# FlashQLA kernels expect bfloat16 — always cast inputs:
q = q.to(torch.bfloat16)
k = k.to(torch.bfloat16)
v = v.to(torch.bfloat16)
g = g.to(torch.bfloat16)
beta = beta.to(torch.bfloat16)
```

### Shape Mismatch for Variable-Length
```python
# cu_seqlens must be int32 and on CUDA with shape [batch+1]
cu_seqlens = cu_seqlens.to(dtype=torch.int32, device='cuda')
assert cu_seqlens[0] == 0
assert cu_seqlens[-1] == total_tokens
```

### Numerical Precision
```python
# For production accuracy testing, compare against float32:
# python test_gdr.py --set product --ref-dtype float32 --num-heads 32
# The kernel is designed to match float32 reference within bfloat16 tolerance
```

## Key Performance Notes

1. **Gate-driven CP**: FlashQLA automatically exploits exponential decay in GDN gates for intra-card context parallelism — works best with long sequences and small head counts (high TP settings).

2. **Optimal chunk size**: Performance is tuned for the chunk sizes used in pretraining; the kernel selects optimal tile sizes internally via TileLang.

3. **Warp specialization**: The kernels use warpgroup specialization to overlap data movement, Tensor Core computation, and CUDA Core computation — no user configuration needed.

4. **Best throughput scenarios**: Pretraining (long sequences, large batches) and agentic inference (repeated state updates) show the largest gains over FLA Triton baseline.
```
