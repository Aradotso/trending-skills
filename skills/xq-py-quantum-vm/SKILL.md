```markdown
---
name: xq-py-quantum-vm
description: Python implementation of the Quip Network's quantum virtual machine (xqvm)
triggers:
  - quantum virtual machine python
  - xqvm quip network
  - quantum circuit simulation python
  - xq-py quantum vm
  - quip network quantum python
  - simulate quantum gates python
  - quantum vm xqvm
  - xqvm-py quantum circuit
---

# xq-py Quantum Virtual Machine

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

`xqvm-py` is a Python implementation of the Quip Network's quantum virtual machine (xqvm). It provides quantum circuit simulation, gate operations, and qubit state management for quantum computing workflows on classical hardware.

---

## Installation

```bash
# Clone from GitLab
git clone https://gitlab.com/piqued/xqvm-py.git
cd xqvm-py

# Install dependencies
pip install -r requirements.txt

# Install as a package (if setup.py/pyproject.toml present)
pip install -e .
```

---

## Core Concepts

- **Qubit**: The fundamental unit of quantum information. xqvm-py models qubits as statevectors.
- **Gate**: Quantum operations applied to qubits (Hadamard, CNOT, Pauli-X/Y/Z, etc.).
- **Circuit**: An ordered sequence of gate operations applied to a register of qubits.
- **Measurement**: Collapse the quantum state into a classical bit outcome.

---

## Basic Usage

### Initialize the VM and Create a Circuit

```python
from xqvm import QuantumVM, QuantumCircuit

# Create a quantum virtual machine
vm = QuantumVM()

# Create a 2-qubit circuit
circuit = QuantumCircuit(num_qubits=2)
```

### Apply Quantum Gates

```python
from xqvm import QuantumCircuit
from xqvm.gates import H, X, Y, Z, CNOT, CZ, T, S

circuit = QuantumCircuit(num_qubits=2)

# Apply Hadamard gate to qubit 0 (creates superposition)
circuit.apply(H, target=0)

# Apply Pauli-X (NOT) gate to qubit 1
circuit.apply(X, target=1)

# Apply CNOT (controlled-NOT): control=0, target=1
circuit.apply(CNOT, control=0, target=1)
```

### Run the Circuit

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, CNOT

# Build Bell state circuit
circuit = QuantumCircuit(num_qubits=2)
circuit.apply(H, target=0)
circuit.apply(CNOT, control=0, target=1)

# Execute on the VM
vm = QuantumVM()
result = vm.run(circuit)

print(result.statevector)   # Complex amplitude vector
print(result.probabilities)  # Measurement probabilities per basis state
```

### Measure Qubits

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H

circuit = QuantumCircuit(num_qubits=3)
circuit.apply(H, target=0)
circuit.apply(H, target=1)
circuit.apply(H, target=2)

vm = QuantumVM()
result = vm.run(circuit)

# Measure all qubits (collapses state, returns classical bits)
bits = result.measure()
print(bits)  # e.g. [0, 1, 0]

# Measure a specific qubit
bit = result.measure_qubit(0)
print(bit)  # 0 or 1
```

### Sample Multiple Shots

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, CNOT

circuit = QuantumCircuit(num_qubits=2)
circuit.apply(H, target=0)
circuit.apply(CNOT, control=0, target=1)

vm = QuantumVM()

# Run 1024 shots and collect measurement histogram
counts = vm.sample(circuit, shots=1024)
print(counts)  # e.g. {'00': 512, '11': 512}
```

---

## Common Quantum Patterns

### Bell State (Maximum Entanglement)

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, CNOT

def bell_state():
    circuit = QuantumCircuit(num_qubits=2)
    circuit.apply(H, target=0)
    circuit.apply(CNOT, control=0, target=1)
    return circuit

vm = QuantumVM()
result = vm.run(bell_state())
counts = vm.sample(bell_state(), shots=2048)
print(counts)  # Should be ~50% '00', ~50% '11'
```

### GHZ State (3-Qubit Entanglement)

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, CNOT

def ghz_state():
    circuit = QuantumCircuit(num_qubits=3)
    circuit.apply(H, target=0)
    circuit.apply(CNOT, control=0, target=1)
    circuit.apply(CNOT, control=0, target=2)
    return circuit

vm = QuantumVM()
counts = vm.sample(ghz_state(), shots=1024)
print(counts)  # ~50% '000', ~50% '111'
```

### Quantum Teleportation Circuit

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, X, Z, CNOT

def teleportation_circuit():
    # 3 qubits: [message, alice, bob]
    circuit = QuantumCircuit(num_qubits=3)

    # Prepare message qubit in |+> state
    circuit.apply(H, target=0)

    # Create Bell pair between Alice and Bob
    circuit.apply(H, target=1)
    circuit.apply(CNOT, control=1, target=2)

    # Alice's operations
    circuit.apply(CNOT, control=0, target=1)
    circuit.apply(H, target=0)

    # Classically conditioned corrections on Bob's qubit
    # (In full teleportation, measure qubits 0 and 1 first)
    circuit.apply(X, target=2)
    circuit.apply(Z, target=2)

    return circuit

vm = QuantumVM()
result = vm.run(teleportation_circuit())
print(result.statevector)
```

### Quantum Fourier Transform (QFT)

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H, CPhase
import math

def qft(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits=num_qubits)
    for i in range(num_qubits):
        circuit.apply(H, target=i)
        for j in range(i + 1, num_qubits):
            angle = math.pi / (2 ** (j - i))
            circuit.apply(CPhase, control=j, target=i, theta=angle)
    return circuit

vm = QuantumVM()
result = vm.run(qft(4))
print(result.probabilities)
```

---

## Parameterized Gates

```python
from xqvm import QuantumCircuit
from xqvm.gates import Rx, Ry, Rz
import math

circuit = QuantumCircuit(num_qubits=1)

# Rotation gates with angle parameter
circuit.apply(Rx, target=0, theta=math.pi / 2)
circuit.apply(Ry, target=0, theta=math.pi / 4)
circuit.apply(Rz, target=0, theta=math.pi)
```

---

## Inspecting State

```python
from xqvm import QuantumVM, QuantumCircuit
from xqvm.gates import H

circuit = QuantumCircuit(num_qubits=2)
circuit.apply(H, target=0)
circuit.apply(H, target=1)

vm = QuantumVM()
result = vm.run(circuit)

# Full statevector (complex numpy array)
sv = result.statevector
print("Statevector:", sv)

# Probability of each basis state
probs = result.probabilities
for state, prob in enumerate(probs):
    print(f"|{state:02b}>: {prob:.4f}")

# Density matrix
dm = result.density_matrix
print("Density matrix shape:", dm.shape)
```

---

## Configuration

```python
from xqvm import QuantumVM

# Configure VM options
vm = QuantumVM(
    backend="statevector",   # 'statevector' or 'density_matrix'
    precision="complex128",  # NumPy dtype for amplitudes
    seed=42,                 # RNG seed for reproducible measurements
)
```

### Environment Variables

```bash
# Optional: override default backend
export XQVM_BACKEND=statevector

# Optional: set global random seed
export XQVM_SEED=42

# Optional: enable debug/verbose output
export XQVM_DEBUG=1
```

---

## Gate Reference

| Gate | Class | Parameters | Description |
|------|-------|------------|-------------|
| Hadamard | `H` | `target` | Superposition |
| Pauli-X | `X` | `target` | Bit flip (NOT) |
| Pauli-Y | `Y` | `target` | Y rotation |
| Pauli-Z | `Z` | `target` | Phase flip |
| CNOT | `CNOT` | `control, target` | Controlled-NOT |
| CZ | `CZ` | `control, target` | Controlled-Z |
| T Gate | `T` | `target` | π/8 gate |
| S Gate | `S` | `target` | Phase gate |
| Rx | `Rx` | `target, theta` | X-axis rotation |
| Ry | `Ry` | `target, theta` | Y-axis rotation |
| Rz | `Rz` | `target, theta` | Z-axis rotation |
| CPhase | `CPhase` | `control, target, theta` | Controlled phase |
| SWAP | `SWAP` | `qubit_a, qubit_b` | Swap two qubits |

---

## Troubleshooting

### ImportError on gates module
```bash
# Ensure you installed from the repo root
pip install -e .
# Or add to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Statevector norm not 1.0
```python
import numpy as np
result = vm.run(circuit)
norm = np.linalg.norm(result.statevector)
assert abs(norm - 1.0) < 1e-9, f"Unnormalized state: norm={norm}"
```

### Reproducible random measurements
```python
vm = QuantumVM(seed=123)
counts1 = vm.sample(circuit, shots=100)
vm2 = QuantumVM(seed=123)
counts2 = vm2.sample(circuit, shots=100)
assert counts1 == counts2  # Deterministic with same seed
```

### Qubit index out of range
```python
circuit = QuantumCircuit(num_qubits=3)
# Valid targets: 0, 1, 2
# This will raise IndexError:
# circuit.apply(H, target=3)
```

---

## Development & Testing

```bash
# Run tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_gates.py

# Lint
flake8 xqvm/
```

---

## Resources

- GitLab Repository: https://gitlab.com/piqued/xqvm-py
- Quip Network: https://quipnetwork.io (check for official docs)
- Open Issues: https://gitlab.com/piqued/xqvm-py/-/issues
```
