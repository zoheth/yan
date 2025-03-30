#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

template <class T>
struct SumOp
{
    __device__ __forceinline__ T
    operator()(const T &a, const T &b) const
    {
        return a + b;
    }

    __device__ __forceinline__ T
    identity() const
    {
        return T(0);
    }
};

template <class T>
struct MaxOp;

template<>
struct MaxOp<float>
{
    __device__ __forceinline__ float
    operator()(const float &a, const float &b) const
    {
        return fmaxf(a, b);
    }

    __device__ __forceinline__ float
    identity() const
    {
        return -__FLT_MAX__;
    }
};

template <class T, class ReduceOp>
__device__ __forceinline__ T
warp_reduce(T val, ReduceOp op)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <class T, class ReduceOp>
__device__ __forceinline__ T
warp_reduce(T val, ReduceOp op)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Convert from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

template <int kNumRows>
struct Softmax {
    Tensor row_max = make_tensor<float>(make_shape<kNumRows>());
    Tensor row_sum = make_tensor<float>(make_shape<kNumRows>());

    __forceinline__ __device__ Softmax() {};

    template<bool kIsFirst, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &accum_s, Tensor1 &accum_o, float softmax_scale)
    {
        Tensor scores = make_tensor(accum_s.data(), convert_layout_acc_rowcol(accum_s.layout()));
        if(kIsFirst)
        {

        }
        
        Tensor output = make_tensor(accum_o.data(), convert_layout_acc_rowcol(accum_o.layout()));


    }
};