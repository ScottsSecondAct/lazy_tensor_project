#pragma once

#include <vector>
#include <span>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <iostream>

// ==========================================
// 1. The Expression Template Base (C++23)
// ==========================================
struct TensorExpr {
    template <typename Self>
    double eval_at(this const Self& self, size_t i) {
        return self.eval_impl(i); 
    }

    template <typename Self>
    size_t size(this const Self& self) {
        return self.size_impl();
    }
};

// ==========================================
// 2. The Leaf Node (Zero-Copy View)
// ==========================================
class TensorView : public TensorExpr {
    std::span<const double> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t math_size; // Tracks the mathematical size (e.g., 2x2 = 4)

public:
    TensorView(std::span<const double> buffer, std::vector<size_t> shp, std::vector<size_t> str)
        : data(buffer), shape(std::move(shp)), strides(std::move(str)) {
        
        // Calculate the true mathematical size
        math_size = 1;
        for (auto s : shape) { math_size *= s; }
    }

    // Translates the logical 1D loop index into the physical strided memory index
    double eval_impl(size_t i) const { 
        size_t physical_idx = 0;
        size_t remaining = i;
        
        // Generalized N-dimensional un-flattening using strides
        for (int k = shape.size() - 1; k >= 0; --k) {
            size_t coord = remaining % shape[k];
            physical_idx += coord * strides[k];
            remaining /= shape[k];
        }
        return data[physical_idx]; 
    }
    
    // Return the mathematical size, NOT the physical buffer size
    size_t size_impl() const { return math_size; }

    // C++23 Multidimensional Subscript Operator [row, col]
    template <std::integral... Args>
    double operator[](Args... indices) const {
        constexpr size_t N = sizeof...(Args);
        assert(N == shape.size() && "Incorrect number of dimensions accessed");
        
        size_t idx_array[N] = { static_cast<size_t>(indices)... };
        size_t flat_index = 0;
        
        for (size_t k = 0; k < N; ++k) {
            flat_index += idx_array[k] * strides[k];
        }
        return data[flat_index];
    }
};

// ==========================================
// 3. AST Nodes (Lazy Operations)
// ==========================================

// Binary Node: Addition
template <typename Lhs, typename Rhs>
class TensorAdd : public TensorExpr {
    const Lhs& lhs;
    const Rhs& rhs;

public:
    TensorAdd(const Lhs& l, const Rhs& r) : lhs(l), rhs(r) {
        assert(l.size() == r.size() && "Tensor sizes must match for addition");
    }

    double eval_impl(size_t i) const { return lhs.eval_at(i) + rhs.eval_at(i); }
    size_t size_impl() const { return lhs.size(); }
};

// Unary Node: Maps a function over a tensor
template <typename Child, typename Func>
class TensorMap : public TensorExpr {
    const Child& child;
    Func op;

public:
    TensorMap(const Child& c, Func f) : child(c), op(f) {}

    double eval_impl(size_t i) const { return op(child.eval_at(i)); }
    size_t size_impl() const { return child.size(); }
};

// ==========================================
// 4. Operator Overloads & Helper Functions
// ==========================================

// Lazy Addition (+)
template <typename Lhs, typename Rhs>
requires std::derived_from<std::decay_t<Lhs>, TensorExpr> && 
         std::derived_from<std::decay_t<Rhs>, TensorExpr>
auto operator+(const Lhs& lhs, const Rhs& rhs) {
    return TensorAdd<Lhs, Rhs>(lhs, rhs);
}

// Lazy ReLU Activation
template <typename Child>
requires std::derived_from<std::decay_t<Child>, TensorExpr>
auto relu(const Child& c) {
    // We let CTAD (Class Template Argument Deduction) deduce the types automatically!
    return TensorMap(c, [](double x) { return std::max(0.0, x); });
}

// ==========================================
// 5. The Memory-Owning Tensor
// ==========================================
class Tensor {
    std::vector<double> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

public:
    Tensor(std::vector<size_t> shp, std::vector<size_t> str) 
        : shape(std::move(shp)), strides(std::move(str)) {
        size_t total_size = 1;
        for (auto s : shape) total_size *= s;
        data.resize(total_size, 0.0);
    }

    // THE TRIGGER: Ingests a lazy AST and evaluates it in a single loop
    template <typename Expr>
    requires std::derived_from<std::decay_t<Expr>, TensorExpr>
    Tensor& operator=(const Expr& expr) {
        assert(expr.size() == data.size() && "Size mismatch during evaluation");
        
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = expr.eval_at(i);
        }
        return *this;
    }

    double operator[](size_t flat_index) const { return data[flat_index]; }
    
    // Helper to print a 2x2 matrix for demonstration purposes
    void print() const {
        if (data.size() >= 4) {
            std::cout << "[ " << data[0] << ", " << data[1] << " ]\n"
                      << "[ " << data[2] << ", " << data[3] << " ]\n\n";
        }
    }
};

// ==========================================
// 6. Eager Operations
// ==========================================

// Hardcoded 2x2 Matrix Multiplication (Demonstrates eager necessity)
inline Tensor matmul_2x2(const TensorView& A, const TensorView& B) {
    Tensor C({2, 2}, {2, 1});
    
    // Eagerly compute the dot products using the multidimensional subscript operator
    std::vector<double> result = {
        A[0,0]*B[0,0] + A[0,1]*B[1,0],  A[0,0]*B[0,1] + A[0,1]*B[1,1],
        A[1,0]*B[0,0] + A[1,1]*B[1,0],  A[1,0]*B[0,1] + A[1,1]*B[1,1]
    };
    
    // Assign via a temporary view
    C = TensorView(result, {2, 2}, {2, 1});
    return C;
}