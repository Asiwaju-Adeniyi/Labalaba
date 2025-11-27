# GPU Butterfly Transform — Fast Structured XW Multiplication

This project implements efficient GPU kernels for the **Butterfly Transform**, a structured alternative to dense matrix–matrix multiplication used in FFTs, classical parallel algorithms, and modern ML architectures.

## What is the Butterfly Pattern?

The Butterfly pattern models hierarchical pairwise interactions between elements.  
It appears in:

- The Cooley–Tukey FFT  
- Multiprocessor interconnect networks  
- Efficient structured neural network layers  
- Algorithms with recursive "swap + combine" phases  

For an input of width **L**, the standard dense multiplication **XW** (shape: *F×L · L×L*) can be replaced by:

\[
W = B_1 B_2 \dots B_e,\quad e=\log_2 L
\]

where each **Butterfly factor** \(B_i\) is a very sparse matrix (only two non-zero entries per row/column, spaced \(2^i\) apart).

This structure reduces both **compute** and **memory** complexity.

---

## 🎯 Project Goal

Efficiently compute:

\[
Y = XW
\]

using:

- **X**: a tensor of shape **(B, F, L)**
- **W**: a tensor of Butterfly factors, shape **(e, L, 2)**  
- **Y**: output tensor of shape **(B, F, L)**



## 🚀 Implementation Plan

- Build CUDA kernels for each Butterfly stage  
- Exploit sparsity for coalesced reads and minimal memory traffic  
- Benchmark against dense GEMM for comparison  
- Validate correctness vs. CPU implementation  



