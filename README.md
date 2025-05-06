# Quantum Recurrent Neural Network (QRNN) Circuit Overview

This repository implements a **Variational Quantum Circuit** used in a **Quantum Recurrent Neural Network (QRNN)**. The quantum circuit processes input data through encoding, variational transformations, entanglement, and measurement to produce quantum-based output representations.

---

## Quantum Circuit Architecture

### 1. Input Encoding

- Each qubit $\( q_0 \)$ to $\( q_4 \)$ is initialized in the ground state $\( |0\rangle \)$.
- Input values $\([0.5, 0.6, 0.7, 0.8, 0.9]\)$ are encoded using **RX gates**.
- The RX gate applies a rotation around the X-axis of the Bloch sphere:

  $\[
  RX(\theta) = e^{-i \frac{\theta}{2} X}
  \]$

  where $\( X \)$ is the Pauli-X matrix, and $\( \theta \)$ is proportional to the input value (e.g., $\( \theta = 0.5 \) for \( q_0 \)$).

---

### 2. Variational Layers

- Each qubit receives a series of **trainable RX and RZ gates**:
  
  $\[
  RZ(\phi) = e^{-i \frac{\phi}{2} Z}
  \]$

  where $\( Z \)$ is the Pauli-Z matrix, and $\( \phi \)$ is a trainable parameter.
- For example, qubit $\( q_0 \)$ might be transformed as:

  $\[
  RX(0.1) \rightarrow RZ(0.2) \rightarrow RX(0.3)
  \]$

- These parameters are optimized during training to learn temporal patterns.

---

### 3. Entanglement

- **CNOT gates** (controlled-NOT) are applied to entangle adjacent qubits, creating a **linear chain** of entanglement.
- Entanglement enables the model to capture complex correlations between input features.

---

### 4. Measurement

- After the variational and entangling operations, all qubits are **measured** in the computational basis $\( \{ |0\rangle, |1\rangle \} \)$.
- Each measurement produces a classical bitstring (e.g., `10010`), representing the collapsed quantum state.

---

## Understanding Measurement Results

### 1. Quantum State Evolution

The state of the quantum system evolves as:

$\[
|\psi\rangle = \sum_{i=0}^{2^n - 1} \alpha_i |i\rangle
\]$

where:
- $\( n \)$ is the number of qubits,
- $\( |i\rangle \)$ is a basis state (bitstring),
- $\( \alpha_i \)$ is the complex amplitude of that state,
- $\( |\alpha_i|^2 \)$ is the probability of measuring $\( |i\rangle \)$.

---

### 2. Measurement Probabilities

After running the circuit multiple times (shots), we collect bitstring counts. The **probability** of a bitstring $\( |i\rangle \)$ is:

$\[
P(|i\rangle) = \frac{\text{count}(|i\rangle)}{\text{total counts}}
\]$

For example, if `00000` is observed 220 times out of 1024 shots:

$\[
P(|00000\rangle) = \frac{220}{1024} \approx 0.215
\]$

---

### 3. Example Measurement Output

```
{
  "00000": 220,
  "10001": 180,
  "11110": 150,
  ...
}
```

- The **most frequent bitstrings** correspond to the **most probable quantum states** influenced by input encoding and learned parameters.

---

## References

- *Quantum Recurrent Neural Networks for Sequential Learning*, arXiv: [2302.03244](https://arxiv.org/abs/2302.03244)