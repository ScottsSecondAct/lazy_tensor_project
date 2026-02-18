#include <iostream>
#include <vector>
#include "tensor.hpp"

int main() {
    std::cout << "=== Comprehensive Tensor Operations (C++23) ===\n\n";

    // Base Memory 
    // We use some negative numbers in A to make the ReLU operation obvious
    std::vector<double> memA = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> memB = {10.0, 20.0, 30.0, 40.0};
    
    // 2x2 Matrix Views (Shape: [2, 2], Strides: [2, 1])
    TensorView A(memA, {2, 2}, {2, 1});
    TensorView B(memB, {2, 2}, {2, 1});

    // ---------------------------------------------------------
    // Operation 1: Standard Chained Lazy Addition
    // ---------------------------------------------------------
    std::cout << "1. Chained Lazy Addition (A + B + A)\n";
    // Builds: TensorAdd<TensorAdd<TensorView, TensorView>, TensorView>
    auto tree_add = A + B + A; 
    
    Tensor C({2, 2}, {2, 1});
    C = tree_add; // Trigger evaluation loop
    
    std::cout << "   Result Matrix C:\n";
    C.print(); // Expected: [12, 16] \n [36, 32] 

    // ---------------------------------------------------------
    // Operation 2: Complex Lazy AST (Unary + Element-wise)
    // ---------------------------------------------------------
    std::cout << "2. Complex Lazy AST (ReLU(A) + B)\n";
    std::cout << "   A contains negative numbers. ReLU zeros them out.\n";
    
    // Builds: TensorAdd<TensorMap<TensorView>, TensorView>
    auto tree_relu = relu(A) + B; 
    
    Tensor D({2, 2}, {2, 1});
    D = tree_relu; // Trigger evaluation loop
    
    std::cout << "   Result Matrix D:\n";
    D.print(); // Expected: [11, 20] \n [33, 40]

    // ---------------------------------------------------------
    // Operation 3: Zero-Copy Broadcasting
    // ---------------------------------------------------------
    std::cout << "3. Zero-Copy Broadcasting (A + Bias)\n";
    std::vector<double> mem_bias = {100.0, 200.0}; // 1D Vector

    // Magic happens here: Row stride is 0. 
    // Moving to the next row doesn't advance the memory pointer!
    TensorView Bias(mem_bias, {2, 2}, {0, 1}); 
    
    auto tree_broadcast = A + Bias;
    
    Tensor E({2, 2}, {2, 1});
    E = tree_broadcast; // Trigger evaluation loop
    
    std::cout << "   Result Matrix E:\n";
    E.print(); // Expected: [101, 198] \n [103, 196]

    // ---------------------------------------------------------
    // Operation 4: Tensor Contraction (Matrix Multiplication)
    // ---------------------------------------------------------
    std::cout << "4. Eager Tensor Contraction (MatMul A * B)\n";
    std::cout << "   Computing A * B eagerly to avoid redundant cache misses...\n";
    
    Tensor F = matmul_2x2(A, B);
    
    std::cout << "   Result Matrix F:\n";
    F.print(); // Expected: [-50, -60] \n [-90, -100]

    return 0;
}