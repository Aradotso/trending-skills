```markdown
---
name: how-to-train-your-gpt
description: A 12-chapter, 3671-line annotated textbook for building a modern LLaMA-style GPT from scratch in PyTorch — tokenizer through inference.
triggers:
  - "build a GPT from scratch"
  - "train a language model"
  - "implement transformer attention"
  - "how does tokenization work"
  - "implement RoPE positional encoding"
  - "write a training loop for LLM"
  - "implement KV cache inference"
  - "understand transformer architecture"
---

# How to Train Your GPT

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A 12-chapter interactive textbook that teaches you to build a modern LLaMA 3-style decoder-only Transformer from absolute scratch. Every line of code is annotated with WHAT it does and WHY it's there. Covers BPE tokenization → embeddings → RoPE → multi-head attention → training → inference.

## Installation

```bash
git clone https://github.com/raiyanyahya/how-to-train-your-gpt.git
cd how-to-train-your-gpt

python -m venv gpt_env
source gpt_env/bin/activate        # Mac/Linux
# gpt_env\Scripts\activate         # Windows

pip install torch tiktoken datasets numpy matplotlib
```

Verify GPU availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Project Structure

```
chapters/
├── 00_overview.md          # Big picture: what is a GPT?
├── 01_setup.md             # Environment, PyTorch basics
├── 02_tokenization.md      # BPE walkthrough
├── 03_embeddings.md        # Token embeddings, semantic space
├── 04_positional_encoding.md  # RoPE math + numeric example
├── 05_attention.md         # ⭐ Core: Q,K,V, scaling, causal mask (713 lines)
├── 06_transformer_block.md # RMSNorm, SwiGLU, residuals
├── 07_gpt_model.md         # Full 124M parameter model
├── 08_training.md          # AdamW, cosine warmup, mixed precision
├── 09_inference.md         # KV cache, sampling strategies
├── 10_full_script.md       # Runnable main.py (copy this to run)
└── 11_glossary.md          # Architecture provenance table
```

## Architecture Overview

This project implements a **LLaMA 3-style** decoder-only Transformer with:

| Component | Technique | Why |
|-----------|-----------|-----|
| Positional encoding | RoPE | Relative position, no learned params |
| Normalization | RMSNorm | 15% faster than LayerNorm |
| Activation | SwiGLU | Gated: learns what to pass/block |
| Norm placement | Pre-norm | Stable gradients at 100+ layers |
| Optimizer | AdamW | Better generalization than Adam |
| Tokenizer | BPE | Handles any text including emoji |
| Param sharing | Weight tying | Saves 30% params |
| Training | Mixed precision (fp16) | 2× speed, half memory |

## Key Code Patterns

### 1. BPE Tokenizer

```python
import tiktoken

# Load GPT-4 BPE tokenizer (same family as used by OpenAI)
enc = tiktoken.get_encoding("cl100k_base")

# Encode text to token IDs
tokens = enc.encode("Hello, how are you?")
# → [9906, 11, 1268, 527, 499, 30]

# Decode back to text
text = enc.decode(tokens)
# → "Hello, how are you?"

vocab_size = enc.n_vocab  # 100,277 for cl100k_base
```

### 2. Token Embeddings

```python
import torch
import torch.nn as nn

# Embedding table: maps token ID → dense vector
# vocab_size tokens, each represented as d_model floats
embedding = nn.Embedding(vocab_size, d_model)

# token_ids shape: (batch_size, seq_len)
token_ids = torch.tensor([[9906, 11, 1268]])  # (1, 3)

# x shape: (batch_size, seq_len, d_model)
x = embedding(token_ids)  # (1, 3, 768)
```

### 3. RoPE (Rotary Position Encoding)

```python
def precompute_rope_freqs(d_model: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute rotation frequencies for RoPE.
    LLaMA rotates Q and K vectors by position-dependent angles
    so attention scores encode relative distance automatically.
    """
    # Frequency for each pair of dimensions: theta^(-2i/d)
    # Shape: (d_model // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    
    # Position indices: 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len)
    
    # Outer product: each position × each frequency
    # Shape: (max_seq_len, d_model // 2)
    freqs = torch.outer(positions, freqs)
    
    # Convert to complex numbers for efficient rotation
    # Shape: (max_seq_len, d_model // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Rotate query or key vectors by their position's angle.
    x shape: (batch, seq_len, num_heads, head_dim)
    """
    # Reshape to pairs of floats, treat as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Multiply = rotate in complex plane
    x_rotated = x_complex * freqs_cis
    
    # Back to real numbers
    return torch.view_as_real(x_rotated).flatten(3).type_as(x)
```

### 4. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # e.g. 768 // 12 = 64
        self.scale = self.head_dim ** -0.5    # 1/√d_k: prevents softmax saturation
        
        # Single projection for Q, K, V (more efficient than 3 separate)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, 
                mask: torch.Tensor = None):
        B, T, C = x.shape  # batch, seq_len, d_model
        
        # Step 1: Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)
        
        # Step 2: Reshape into heads
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Apply RoPE to Q and K (not V)
        q = apply_rope(q.transpose(1, 2), freqs_cis).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2), freqs_cis).transpose(1, 2)
        
        # Step 4: Scaled dot-product attention scores
        # (B, heads, T, head_dim) @ (B, heads, head_dim, T) → (B, heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Step 5: Causal mask — token i cannot attend to token j > i
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 6: Softmax over last dim → attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Step 7: Weighted sum of values
        # (B, heads, T, T) @ (B, heads, T, head_dim) → (B, heads, T, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # Step 8: Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```

### 5. RMSNorm

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Faster than LayerNorm: no mean subtraction, just scale by RMS.
    Used by LLaMA, Mistral, Gemma instead of LayerNorm.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learned scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS across last dimension
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize then scale
        return (x / rms) * self.weight
```

### 6. SwiGLU Feed-Forward

```python
class SwiGLU(nn.Module):
    """
    Gated feed-forward network used by LLaMA/PaLM.
    Two projections: one goes through Swish activation (gate),
    the other is linear. Their element-wise product lets the network
    learn WHICH information to pass through.
    
    hidden_dim is typically 8/3 * d_model (vs 4× for standard FFN).
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(gate) ⊙ up: gating mechanism
        gate = torch.nn.functional.silu(self.gate_proj(x))  # SiLU = Swish
        up   = self.up_proj(x)
        return self.down_proj(gate * up)
```

### 7. Full Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    One decoder layer: Pre-norm → Attention → residual,
    then Pre-norm → FFN → residual.
    
    Pre-norm (normalize BEFORE sublayer) is critical for stable
    training at depth — gradients flow cleanly through residuals.
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, ffn_hidden)
    
    def forward(self, x, freqs_cis, mask):
        # Residual + pre-norm attention
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        # Residual + pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x
```

### 8. Complete GPT Model

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, 
                 num_heads=12, max_seq_len=1024, dropout=0.1):
        super().__init__()
        ffn_hidden = int(8/3 * d_model)  # SwiGLU uses 8/3 × d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_hidden)
            for _ in range(num_layers)
        ])
        self.norm     = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: token_emb and lm_head share the same matrix.
        # Saves ~30% parameters; both learn token representations.
        self.lm_head.weight = self.token_emb.weight
        
        # Precompute RoPE frequencies once
        self.freqs_cis = precompute_rope_freqs(d_model // num_heads, max_seq_len)
        
        # Causal mask: lower triangular — token i only sees tokens 0..i
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask)
    
    def forward(self, token_ids: torch.Tensor):
        B, T = token_ids.shape
        
        x = self.dropout(self.token_emb(token_ids))  # (B, T, d_model)
        
        freqs_cis = self.freqs_cis[:T].to(token_ids.device)
        mask = self.causal_mask[:T, :T]
        
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 9. Training Loop

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train(model, dataloader, num_steps=50_000, lr=3e-4, warmup_steps=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95),   # LLaMA defaults
        weight_decay=0.1      # L2 regularization
    )
    scaler = GradScaler()     # Mixed precision gradient scaler
    
    def get_lr(step):
        # Linear warmup then cosine decay
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    model.train()
    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break
        
        # Update learning rate
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        input_ids = batch['input_ids'].to(device)   # (B, T)
        labels    = batch['labels'].to(device)       # (B, T) — shifted by 1
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass (fp16 on GPU)
        with autocast():
            logits = model(input_ids)               # (B, T, vocab_size)
            # Flatten for cross-entropy
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                labels.view(-1)                      # (B*T,)
            )
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping — prevents exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
        
        step += 1
```

### 10. Inference with Temperature & Top-K Sampling

```python
@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
             temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9):
    """
    Autoregressive generation: feed current sequence, predict next token,
    append it, repeat.
    
    temperature: >1 = more random, <1 = more focused, 0 = greedy
    top_k: only sample from k most likely tokens
    top_p: nucleus sampling — sample from smallest set summing to p
    """
    model.eval()
    device = next(model.parameters()).device
    x = prompt_ids.to(device)  # (1, T)
    
    for _ in range(max_new_tokens):
        # Get logits for last position only (next token prediction)
        logits = model(x)[:, -1, :]  # (1, vocab_size)
        
        # Apply temperature scaling
        logits = logits / max(temperature, 1e-8)
        
        # Top-K filtering: zero out all but top-k logits
        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < top_k_vals, float('-inf'))
        
        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens beyond cumulative probability p
            remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float('-inf')
            # Scatter back to original ordering
            logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        
        # Append to sequence
        x = torch.cat([x, next_token], dim=1)
        
        # Stop at EOS token (50256 for GPT-2 tokenizer)
        if next_token.item() == 50256:
            break
    
    return x  # (1, original_T + new_tokens)
```

### 11. Running the Full Script

```bash
# Copy the complete main.py from Chapter 10
cp chapters/10_full_script.md main.py   # then remove markdown fencing

# Or copy the code blocks manually into main.py
python main.py
```

Expected output:
```
GPT initialized with 124,439,808 parameters
Training starting!
Step    100/50,000 | Loss: 6.2345 | LR: 1.50e-05 | Toks/sec: 45,000
Step    200/50,000 | Loss: 5.1234 | LR: 3.00e-05 | Toks/sec: 45,200
...
Step 50,000/50,000 | Loss: 2.8901 | LR: 1.00e-05 | Toks/sec: 44,800
✅ Training complete! 112.3 min | Best loss: 2.8901
```

## Common Configurations

### Tiny Model (CPU / Testing)
```python
model = GPT(
    vocab_size=50_257,
    d_model=128,       # was 768
    num_layers=2,      # was 12
    num_heads=4,       # was 12
    max_seq_len=256,   # was 1024
)
# ~3M parameters — trains in minutes on CPU
```

### GPT-2 Scale (124M params)
```python
model = GPT(
    vocab_size=50_257,
    d_model=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=1024,
)
# ~124M parameters — needs GPU (RTX 3090 ≈ 2 hours)
```

### GPT-2 Medium Scale (350M params)
```python
model = GPT(
    vocab_size=50_257,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    max_seq_len=1024,
)
# ~350M parameters — needs A100 or multiple GPUs
```

## Troubleshooting

### Loss is NaN from step 1
```python
# Check for bad data (empty sequences, -1 labels leaking into loss)
print(batch['labels'].min(), batch['labels'].max())  # should be 0 to vocab_size-1

# Check for zero learning rate
print(optimizer.param_groups[0]['lr'])  # should not be 0 at step > 0

# Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # try 3e-4 → 1e-4
```

### Loss plateaus early (e.g. stuck at ~6.0)
```python
# Check that labels are properly shifted (predict NEXT token, not current)
# input:  [tok_0, tok_1, tok_2]
# labels: [tok_1, tok_2, tok_3]  ← must be shifted by 1

# Check gradient norms — if very small, increase LR or check norm
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

### CUDA out of memory
```python
# Reduce batch size
batch_size = 4  # try 8 → 4 → 2

# Use gradient accumulation to simulate larger batches
accum_steps = 8  # effective batch = batch_size × accum_steps
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch) / accum_steps  # normalize
    scaler.scale(loss).backward()
    if (i + 1) % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# Reduce sequence length
max_seq_len = 512  # was 1024
```

### Generation repeats itself
```python
# Add repetition penalty or increase temperature
temperature = 1.2   # more diverse
top_p = 0.85        # tighter nucleus

# Or use top_k to avoid degenerate loops
top_k = 40
```

### Causal mask not working (model sees future tokens)
```python
# Verify mask is lower-triangular
mask = torch.tril(torch.ones(T, T))
print(mask)
# Should be:
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
# -inf fills go in the UPPER triangle (where mask == 0)
```

## Key Concepts Reference

| Term | One-Line Explanation |
|------|---------------------|
| **Token** | A chunk of text (not always a word). "unbelievably" → ["un", "believ", "ably"] |
| **Embedding** | A learned vector that represents a token's meaning |
| **Q, K, V** | Query (what I'm looking for), Key (what I advertise), Value (what I share) |
| **Attention score** | How much token A should attend to token B: `softmax(QK^T / √d_k)V` |
| **Causal mask** | Prevents token at position i from attending to position j > i |
| **RoPE** | Encodes position by rotating Q/K vectors — relative distance baked into dot product |
| **RMSNorm** | Normalize by root-mean-square only (no mean shift) — faster than LayerNorm |
| **SwiGLU** | Gated FFN: `down(silu(gate(x)) * up(x))` — learns what info to propagate |
| **Pre-norm** | Normalize input BEFORE attention/FFN sublayer (vs post-norm after) |
| **Residual** | `x = x + sublayer(x)` — skip connection; gradient highway through depth |
| **Weight tying** | `lm_head.weight = token_emb.weight` — shared matrix saves ~30% params |
| **KV cache** | Cache K,V for past tokens; only compute new token each step — 500× faster |
| **Temperature** | Divide logits before softmax: <1 sharper, >1 flatter distribution |
| **Top-K** | Sample only from K most probable next tokens |
| **Top-P (nucleus)** | Sample from smallest set of tokens whose probabilities sum to P |
| **AdamW** | Adam + decoupled weight decay — standard LLM optimizer |
| **Cosine warmup** | LR: linear ramp-up then cosine decay — prevents early instability |

## Reading Order

Read chapters sequentially — each builds on the previous:

```
00 → 01 → 02 → 03 → 04 → 05 ⭐ → 06 → 07 → 08 → 09 → 10 → 11
```

Chapter 05 (Attention) is the longest at 713 lines and the most important. Re-read it until the 8-step walkthrough is clear.
```
