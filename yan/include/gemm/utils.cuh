#pragma once

#include <exception>

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}