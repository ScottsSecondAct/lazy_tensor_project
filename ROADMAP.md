# Tensor Library Architecture Roadmap

This roadmap outlines the strategic phases to elevate the current C++23 lazy-evaluation engine into a production-grade framework capable of handling massive scientific compute workloads, AI primitives, and bioinformatics alignments.

## Phase 1: Concurrency & Hardware Utilization
The current evaluation loop is single-threaded. To achieve true high performance, the evaluation engine must utilize modern CPU architectures fully.
* [ ] **Parallel Execution Policies:** Integrate `std::execution::par_unseq` into the `Tensor::operator=` assignment loop to thread the AST evaluation automatically across all CPU cores.
* [ ] **SIMD Vectorization:** Specialize the `TensorExpr` nodes to evaluate batches of data using AVX2/AVX-512 intrinsics, computing 4-8 double-precision floats per clock cycle.
* [ ] **Cache-Aware MatMul:** Replace the naive $O(N^3)$ eager matrix multiplication with a block-tiled algorithm to minimize L1/L2 cache misses, or wrap an optimized BLAS backend (OpenBLAS/Intel MKL).

## Phase 2: API & Syntactic Sugar
Enhance the developer experience to mimic the ergonomics of Python-based frameworks (like PyTorch/NumPy) while maintaining C++ performance.
* [ ] **Advanced Slicing:** Implement a slicing API that returns zero-copy `TensorView` objects for sub-regions (e.g., extracting a 2x2 block from a 10x10 matrix by adjusting the initial memory offset and strides).
* [ ] **Reduction Operations:** Add eager reduction functions (`sum()`, `mean()`, `max()`) that can collapse tensors along specific axes.
* [ ] **Formatting:** Integrate C++20 `<format>` or `std::print` (C++23) to enable beautiful console rendering of N-dimensional matrices.

## Phase 3: Automatic Differentiation (Autodiff)
To support neural networks or complex gradient descents, the library must track the mathematical operations performed.
* [ ] **The Computational Graph:** Extend the AST nodes to dynamically allocate a graph tracking inputs, outputs, and operations (Forward pass).
* [ ] **Reverse-Mode Autodiff:** Implement `.backward()` to traverse the computational graph in reverse, utilizing the Chain Rule to calculate gradients for all leaf tensors.

## Phase 4: Domain-Specific Accelerators
* [ ] **GPU Backend:** Abstract the memory allocator within the `Tensor` class to support device memory (VRAM). Implement a CUDA or SYCL execution backend to offload the evaluation of the lazy AST to the GPU.
* [ ] **Bioinformatics Primitives:** Introduce domain-specific eager operations optimized for sequence alignment (e.g., vectorized Smith-Waterman or Needleman-Wunsch scoring matrices) that operate natively on the tensor memory layout.