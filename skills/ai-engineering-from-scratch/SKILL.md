```markdown
---
name: ai-engineering-from-scratch
description: Comprehensive AI engineering curriculum with 230+ hands-on lessons covering math, ML, DL, NLP, vision, transformers, LLMs, agents, and swarms across Python, TypeScript, Rust, and Julia.
triggers:
  - "help me follow the ai engineering from scratch course"
  - "set up the ai engineering curriculum project"
  - "work through a lesson in rohitg00 ai engineering"
  - "build a neural network from scratch following this course"
  - "implement an agent using the ai engineering course structure"
  - "run the jupyter notebooks for this ai course"
  - "add a new lesson to the ai engineering from scratch repo"
  - "explain the phase structure of this ai course"
---

# AI Engineering from Scratch

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A structured, hands-on AI engineering curriculum with 230+ lessons across 20 phases. Covers everything from linear algebra to autonomous agent swarms. Every lesson produces reusable artifacts: prompts, skills, agents, MCP servers. Supports Python, TypeScript, Rust, and Julia.

---

## What This Project Does

- **20 phases** progressing from setup/tooling → math → classical ML → deep learning → NLP → vision → speech → transformers → LLMs → agents → swarms
- **230+ runnable lessons** — each in its own directory with code, notebooks, and docs
- **Multi-language** — Python (primary), TypeScript, Rust, Julia
- Each lesson outputs something reusable (a tool, prompt, agent, MCP server, or notebook)
- Designed for AI coding agents and humans to learn, build, and ship together

---

## Repository Structure

```
ai-engineering-from-scratch/
├── phases/
│   ├── 00-setup-and-tooling/
│   │   ├── 01-dev-environment/
│   │   ├── 02-git-and-collaboration/
│   │   └── ...
│   ├── 01-math-foundations/
│   ├── 02-ml-fundamentals/
│   ├── 03-deep-learning-core/
│   ├── 04-computer-vision/
│   ├── 05-nlp-foundations-to-advanced/
│   ├── 06-speech-and-audio/
│   └── ... (phases 07–19)
├── assets/
├── glossary/
│   └── terms.md
├── ROADMAP.md
├── CONTRIBUTING.md
└── README.md
```

Each lesson directory typically contains:
```
phases/NN-phase-name/NN-lesson-name/
├── README.md          # lesson explanation and goals
├── solution.py        # reference implementation
├── notebook.ipynb     # interactive walkthrough
├── requirements.txt   # lesson-specific dependencies
└── tests/             # validation tests
```

---

## Installation & Environment Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Node.js 18+ (for TypeScript lessons)
node --version

# Rust (for Rust lessons)
rustup --version

# Julia (for math lessons)
julia --version
```

### Clone and Bootstrap

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
```

### Python Environment (recommended: per-phase venv)

```bash
# Create a base environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Or use conda
conda create -n ai-eng python=3.11
conda activate ai-eng

# Install base dependencies
pip install -r requirements.txt  # if present at root

# Install lesson-specific deps
pip install -r phases/01-math-foundations/01-linear-algebra-intuition/requirements.txt
```

### Common Python Dependencies Across Lessons

```bash
pip install numpy scipy matplotlib pandas scikit-learn
pip install torch torchvision torchaudio          # deep learning
pip install transformers datasets tokenizers       # HuggingFace
pip install openai anthropic                       # LLM APIs
pip install jupyter notebook ipykernel             # notebooks
pip install pytest black ruff                      # dev tools
```

### API Keys (set as environment variables — never hardcode)

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export HUGGINGFACE_TOKEN="your-token-here"
export GOOGLE_API_KEY="your-key-here"
```

Or use a `.env` file with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Running Lessons

### Jupyter Notebooks

```bash
# Start Jupyter for a specific phase
cd phases/01-math-foundations/01-linear-algebra-intuition/
jupyter notebook notebook.ipynb

# Or JupyterLab
jupyter lab

# Run all notebooks in a phase non-interactively
jupyter nbconvert --to notebook --execute phases/01-math-foundations/*/notebook.ipynb
```

### Python Scripts

```bash
# Run a specific lesson's solution
python phases/02-ml-fundamentals/02-linear-regression/solution.py

# Run with arguments (common pattern)
python phases/03-deep-learning-core/01-the-perceptron/solution.py --epochs 100 --lr 0.01
```

### Tests

```bash
# Test a specific lesson
pytest phases/02-ml-fundamentals/02-linear-regression/tests/

# Test an entire phase
pytest phases/02-ml-fundamentals/

# All tests
pytest phases/
```

---

## Real Code Examples

### Phase 1 — Linear Algebra (Math Foundations)

```python
# phases/01-math-foundations/02-vectors-matrices-operations/solution.py
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)          # 32
cosine_sim = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # shape: (3, 5)

# Eigendecomposition
M = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"Eigenvalues: {eigenvalues}")
```

### Phase 2 — Linear Regression from Scratch

```python
# phases/02-ml-fundamentals/02-linear-regression/solution.py
import numpy as np

class LinearRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # Gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

# Usage
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=5, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegressionScratch(lr=0.001, n_iters=2000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Phase 3 — Backpropagation from Scratch

```python
# phases/03-deep-learning-core/03-backpropagation-from-scratch/solution.py
import numpy as np

class Value:
    """Scalar autograd engine (micrograd-style)."""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Forward + backward pass
x = Value(2.0)
w = Value(-3.0)
b = Value(6.0)
out = (x * w + b).relu()
out.backward()
print(f"x.grad={x.grad}, w.grad={w.grad}")  # x.grad=-3.0, w.grad=2.0
```

### Phase 3 — Neural Network with PyTorch

```python
# phases/03-deep-learning-core/11-introduction-to-pytorch/solution.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=10, hidden_dim=64, output_dim=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy data
X = torch.randn(500, 10)
y = torch.randint(0, 2, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(20):
    loss = train(model, loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: loss={loss:.4f}")
```

### Phase 5 — Attention Mechanism from Scratch

```python
# phases/05-nlp-foundations-to-advanced/10-attention-mechanism/solution.py
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_k, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # (batch, heads, seq_q, seq_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    weights = softmax(scores, axis=-1)
    return weights @ V, weights

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# Multi-Head Attention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq=10, d_model=512)
out = mha(x, x, x)
print(out.shape)  # torch.Size([2, 10, 512])
```

### Phase (LLM / Agents) — OpenAI API Pattern

```python
# Common pattern across agent/LLM phases
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def chat(messages: list[dict], model="gpt-4o", temperature=0.7) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def agent_with_tools(user_query: str) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": user_query}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message

# Usage
reply = chat([
    {"role": "system", "content": "You are an AI engineering tutor."},
    {"role": "user",   "content": "Explain backpropagation in simple terms."},
])
print(reply)
```

---

## Common Lesson Patterns

### 1. "Build from Scratch" Pattern

Every phase starts by implementing core concepts manually before using libraries:

```python
# Pattern: implement manually → verify against library → then use library
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Build it yourself
my_model = LinearRegressionScratch(lr=0.001, n_iters=2000)
my_model.fit(X_train, y_train)

# 2. Verify against sklearn
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

# 3. Compare results
print(f"My MSE:  {np.mean((my_model.predict(X_test) - y_test)**2):.4f}")
print(f"SK MSE:  {np.mean((sk_model.predict(X_test) - y_test)**2):.4f}")
```

### 2. Experiment Tracking Pattern

```python
import json
from pathlib import Path
from datetime import datetime

def log_experiment(phase, lesson, config, metrics):
    log_dir = Path("experiments") / phase / lesson
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
    }
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(entry, indent=2))
    print(f"Experiment logged to {log_file}")

# Usage
log_experiment(
    phase="02-ml-fundamentals",
    lesson="02-linear-regression",
    config={"lr": 0.001, "n_iters": 2000},
    metrics={"train_mse": 0.023, "test_mse": 0.031},
)
```

### 3. Reusable Artifact Pattern

Each lesson outputs something others can import:

```python
# phases/02-ml-fundamentals/02-linear-regression/__init__.py
from .solution import LinearRegressionScratch

__all__ = ["LinearRegressionScratch"]
```

```python
# Use a prior lesson's output in a later lesson
from phases.ml_fundamentals.linear_regression import LinearRegressionScratch
```

### 4. GPU-Aware Training Pattern

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

model = MyModel().to(device)
X = X.to(device)
```

---

## Contributing a New Lesson

Follow the directory structure convention:

```bash
# Create lesson directory
mkdir -p phases/NN-phase-name/NN-lesson-name

# Required files
touch phases/NN-phase-name/NN-lesson-name/README.md
touch phases/NN-phase-name/NN-lesson-name/solution.py
touch phases/NN-phase-name/NN-lesson-name/notebook.ipynb
touch phases/NN-phase-name/NN-lesson-name/requirements.txt

# Optional
mkdir -p phases/NN-phase-name/NN-lesson-name/tests
```

### Lesson README Template

```markdown
# Lesson NN: Lesson Title

## What You'll Build
Brief description of the output artifact.

## Concepts Covered
- Concept 1
- Concept 2

## Prerequisites
- Phase XX, Lesson YY

## Running
\`\`\`bash
pip install -r requirements.txt
python solution.py
\`\`\`

## Output
Description of what gets produced.
```

---

## Troubleshooting

### CUDA / GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# For Google Colab GPU
!nvidia-smi

# Install CUDA-specific PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Notebook Kernel Issues

```bash
# Register venv as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=ai-eng --display-name "AI Engineering"

# Clear notebook outputs before committing
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb
```

### Dependency Conflicts

```bash
# Use per-lesson virtual environments to avoid conflicts
python -m venv phases/03-deep-learning-core/.venv
source phases/03-deep-learning-core/.venv/bin/activate
pip install -r phases/03-deep-learning-core/requirements.txt
```

### Julia Not Found (math lessons)

```bash
# Install Julia via juliaup
curl -fsSL https://install.julialang.org | sh
julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "Statistics", "Plots"])'
```

### HuggingFace Model Downloads Failing

```bash
# Set cache directory
export HF_HOME=/path/to/large/disk/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Use offline mode after first download
export TRANSFORMERS_OFFLINE=1
```

---

## Quick Reference: Phase Map

| Phase | Topic | Key Tools |
|-------|-------|-----------|
| 00 | Setup & Tooling | Python, Node, Docker, Git |
| 01 | Math Foundations | NumPy, Julia, SciPy |
| 02 | ML Fundamentals | scikit-learn, pandas |
| 03 | Deep Learning Core | PyTorch, JAX |
| 04 | Computer Vision | torchvision, OpenCV, YOLO |
| 05 | NLP | HuggingFace, spaCy, NLTK |
| 06 | Speech & Audio | librosa, Whisper |
| 07+ | Transformers → LLMs → Agents → MCP → Swarms | OpenAI, Anthropic, LangChain |

---

## Key Resources

- **Roadmap**: [`ROADMAP.md`](ROADMAP.md) — planned future lessons
- **Glossary**: [`glossary/terms.md`](glossary/terms.md) — AI terminology reference
- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to add lessons
- **License**: MIT — free to use, fork, and build upon
```
