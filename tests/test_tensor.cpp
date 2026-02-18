#include <gtest/gtest.h>
#include <vector>
#include "tensor.hpp"

// ==========================================
// Test Fixture & Memory Setup
// ==========================================
class TensorTest : public ::testing::Test {
protected:
    // 2x2 Matrix data
    std::vector<double> memA = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> memB = {10.0, 20.0, 30.0, 40.0};
    
    // 1D Vectors for broadcasting and dimension tests
    std::vector<double> mem_bias_row = {100.0, 200.0};
    std::vector<double> mem_bias_col = {1000.0, 2000.0};
    std::vector<double> mem_1d = {5.0, 10.0, 15.0};
};

// ==========================================
// 1. Multidimensional Access & Logical Mapping
// ==========================================
TEST_F(TensorTest, Cpp23MultidimensionalSubscript) {
    TensorView A(memA, {2, 2}, {2, 1});
    
    EXPECT_DOUBLE_EQ((A[0, 0]), 1.0);
    EXPECT_DOUBLE_EQ((A[0, 1]), -2.0);
    EXPECT_DOUBLE_EQ((A[1, 0]), 3.0);
    EXPECT_DOUBLE_EQ((A[1, 1]), -4.0);
}

// ==========================================
// 2. 1D and 2D Lazy Addition
// ==========================================
TEST_F(TensorTest, LazyAddition1DAnd2D) {
    // 2D Addition
    TensorView A(memA, {2, 2}, {2, 1});
    TensorView B(memB, {2, 2}, {2, 1});
    auto tree2d = A + B;
    EXPECT_DOUBLE_EQ(tree2d.eval_at(0), 11.0);  // 1.0 + 10.0
    EXPECT_DOUBLE_EQ(tree2d.eval_at(3), 36.0);  // -4.0 + 40.0

    // 1D Addition (Boundary Test)
    TensorView V1(mem_1d, {3}, {1});
    TensorView V2(mem_1d, {3}, {1});
    auto tree1d = V1 + V2;
    EXPECT_DOUBLE_EQ(tree1d.eval_at(2), 30.0);  // 15.0 + 15.0
    EXPECT_EQ(tree1d.size(), 3);
}

// ==========================================
// 3. Unary Operations (ReLU)
// ==========================================
TEST_F(TensorTest, UnaryReLU) {
    TensorView A(memA, {2, 2}, {2, 1});
    auto tree_relu = relu(A);
    
    EXPECT_DOUBLE_EQ(tree_relu.eval_at(0), 1.0);  // Positive stays positive
    EXPECT_DOUBLE_EQ(tree_relu.eval_at(1), 0.0);  // -2.0 becomes 0.0
    EXPECT_DOUBLE_EQ(tree_relu.eval_at(2), 3.0);
    EXPECT_DOUBLE_EQ(tree_relu.eval_at(3), 0.0);  // -4.0 becomes 0.0
}

// ==========================================
// 4. Complex AST Evaluation to Memory Tensor
// ==========================================
TEST_F(TensorTest, ComplexASTEvaluation) {
    TensorView A(memA, {2, 2}, {2, 1});
    TensorView B(memB, {2, 2}, {2, 1});
    
    // C = ReLU(A) + B
    auto tree = relu(A) + B;
    
    Tensor C({2, 2}, {2, 1});
    C = tree; // Trigger Eager Loop
    
    EXPECT_DOUBLE_EQ((C[0]), 11.0); // max(0, 1) + 10
    EXPECT_DOUBLE_EQ((C[1]), 20.0); // max(0, -2) + 20
    EXPECT_DOUBLE_EQ((C[2]), 33.0); // max(0, 3) + 30
    EXPECT_DOUBLE_EQ((C[3]), 40.0); // max(0, -4) + 40
}

// ==========================================
// 5. Zero-Copy Broadcasting (Row & Column)
// ==========================================
TEST_F(TensorTest, ZeroCopyBroadcasting) {
    TensorView A(memA, {2, 2}, {2, 1});

    // Row Broadcast: Stretch a 1x2 vector across 2 rows. Row stride is 0.
    TensorView BiasRow(mem_bias_row, {2, 2}, {0, 1});
    auto tree_row = A + BiasRow;
    Tensor D_row({2, 2}, {2, 1});
    D_row = tree_row;

    EXPECT_DOUBLE_EQ((D_row[0]), 101.0); // 1.0 + 100.0
    EXPECT_DOUBLE_EQ((D_row[1]), 198.0); // -2.0 + 200.0
    EXPECT_DOUBLE_EQ((D_row[2]), 103.0); // 3.0 + 100.0 (Row 1 uses same bias)
    EXPECT_DOUBLE_EQ((D_row[3]), 196.0); // -4.0 + 200.0

    // Column Broadcast: Stretch a 2x1 vector across 2 columns. Col stride is 0.
    // mem_bias_col = {1000.0, 2000.0}. We want 1000 applied to Row 0, 2000 applied to Row 1.
    TensorView BiasCol(mem_bias_col, {2, 2}, {1, 0});
    auto tree_col = A + BiasCol;
    Tensor D_col({2, 2}, {2, 1});
    D_col = tree_col;

    EXPECT_DOUBLE_EQ((D_col[0]), 1001.0); // 1.0 + 1000.0
    EXPECT_DOUBLE_EQ((D_col[1]), 998.0);  // -2.0 + 1000.0
    EXPECT_DOUBLE_EQ((D_col[2]), 2003.0); // 3.0 + 2000.0
    EXPECT_DOUBLE_EQ((D_col[3]), 1996.0); // -4.0 + 2000.0
}

// ==========================================
// 6. Eager Matrix Multiplication
// ==========================================
TEST_F(TensorTest, EagerMatMul) {
    TensorView A(memA, {2, 2}, {2, 1});
    TensorView B(memB, {2, 2}, {2, 1});
    
    Tensor C = matmul_2x2(A, B);
    
    // A[0,:] * B[:,0] = (1*10) + (-2*30) = 10 - 60 = -50
    EXPECT_DOUBLE_EQ((C[0]), -50.0);
    // A[0,:] * B[:,1] = (1*20) + (-2*40) = 20 - 80 = -60
    EXPECT_DOUBLE_EQ((C[1]), -60.0);
    // A[1,:] * B[:,0] = (3*10) + (-4*30) = 30 - 120 = -90
    EXPECT_DOUBLE_EQ((C[2]), -90.0);
    // A[1,:] * B[:,1] = (3*20) + (-4*40) = 60 - 160 = -100
    EXPECT_DOUBLE_EQ((C[3]), -100.0);
}

// ==========================================
// 7. Boundary Test: Shape Mismatch Assertion
// ==========================================
TEST_F(TensorTest, ShapeMismatchAssertion) {
    TensorView A(memA, {2, 2}, {2, 1});
    TensorView V(mem_1d, {3}, {1}); // 1D Vector, Size 3
    
    // In a debug build, attempting to add a 2x2 to a 1D vector should kill the process.
#ifndef NDEBUG
    EXPECT_DEATH({
        [[maybe_unused]]auto invalid_expr = A + V;
    }, "Tensor sizes must match for addition");
#endif
}