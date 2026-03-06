# High-Performance Lazy Tensor Library (C++23)

[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-green.svg)](https://github.com/ScottsSecondAct/some) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Problem Statement:** Create a High-Performance Tensor Math Library (Lazy Evaluation)

- High-Level Goal: Implement a NumPy-like library for C++ where operations like A + B * C do not create temporary matrices but instead build an expression tree evaluated only upon assignment.
- Core Concepts: Expression Templates, Move Semantics (Rule of 5), std::span (C++20), Smart Pointers.
- Architectural Challenges: Managing object lifetimes within expression trees. Optimizing for "Return Value Optimization" (RVO).
- Tooling: BLAS/LAPACK (for backend linkage), Google Benchmark.

## Solution
A modern, header-only C++23 tensor library designed to demonstrate zero-overhead abstraction, lazy evaluation, and multi-dimensional memory mapping.

This project bypasses legacy C++ idioms (like the Curiously Recurring Template Pattern / CRTP) in favor of cutting-edge C++23 features to build a highly optimized, compile-time Abstract Syntax Tree (AST) for mathematical operations.

### Architectural Highlights

* **Zero-Overhead Expression Templates:** Uses C++23 "Deducing `this`" (Explicit Object Parameters - P0847) to build lazy-evaluation trees without virtual function dispatch or CRTP boilerplate.
* **Zero-Copy Broadcasting & Transposition:** Separates physical 1D memory from logical N-dimensional shapes using programmable strides.
* **Multidimensional Subscripting:** Implements the C++23 multidimensional subscript operator (P2128) for native `tensor[row, col]` syntax.
* **Eager vs. Lazy Separation:** Intelligently separates element-wise operations (evaluated lazily in a single loop) from tensor contractions (evaluated eagerly to optimize CPU cache locality).

### Prerequisites

To build and run this project, you must have a compiler that fully supports C++23 "Deducing `this`".
* **Compiler:** Clang 18.0 or newer (Strictly required)
* **Build System:** CMake 3.20 or newer
* **OS:** Linux (Ubuntu 18.1.3 recommended), macOS, or Windows (via Clang-cl)

### Build Instructions

This project uses CMake to orchestrate the build and automatically fetches GoogleTest for unit testing.

```bash
# 1. Ensure Clang 18 is the active compiler
export CXX=clang++-18
export CC=clang-18

# 2. Configure the build directory
cmake -B build -S .

# 3. Compile the demonstration and test binaries
cmake --build build -j$(nproc)

# 4. Run the unit tests
cd build
ctest --output-on-failure -V

# 5. Run the demonstration
./tensor_demo
```

### Acknowledgments
The problem statement, architectural blueprints, C++23 metaprogramming patterns, and build system configurations for this project were developed in collaboration with an AI assistant as part of an advanced systems-programming exercise.
