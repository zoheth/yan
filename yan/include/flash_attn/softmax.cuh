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

template <int kNumThread, class T, class ReduceOp>
__device__ __forceinline__ T
warp_reduce(T val, ReduceOp op)
{
#pragma unroll
    for (int offset = kNumThread / 2; offset > 0; offset /= 2)
    {
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template<bool kIsFirst = true, typename Tensor0, typename Tensor1, class ReduceOp>
__device__ __forceinline__ void row_reduce(Tensor0 const& tensor, Tensor1 &row_summary, ReduceOp op)
{
    #pragma unroll
    for(int row = 0; row < size<0>(tensor); ++row)
    {
        row_summary(row) = kIsFirst ? tensor(row, 0) : op(row_summary(row), tensor(row, 0));
        #pragma unroll
        for(int col = 1; col < size<1>(tensor); ++col)
        {
            row_summary(row) = op(row_summary(row), tensor(row, col));
        }
    }

    #pragma unroll
    for(int row = 0; row < size<0>(tensor); ++row)
    {
        row_summary(row) = warp_reduce<4>(row_summary(row), op);
    }
} 

// Convert from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

template<typename Tensor0, typename Tensor1>
__forceinline__ __device__ void scale_apply_exp2(Tensor0 &tensor, Tensor1 &row_max, float scale_log2)
{
    #pragma unroll
    for(int row = 0; row < size<0>(tensor); ++row)
    {
        const float row_max_val = row_max(row) == -INFINITY ? 0.0f : row_max(row) * scale_log2;
        #pragma unroll
        for(int col = 0; col < size<1>(tensor); ++col)
        {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(row, col) = __exp2f(tensor(row, col) * scale_log2 - row_max_val);
        }
    }
}

template <int kNumRows>
struct Softmax {
    Tensor row_max = make_tensor<float>(make_shape<kNumRows>());
    Tensor row_sum = make_tensor<float>(make_shape<kNumRows>());

    __forceinline__ __device__ Softmax() {};

    template<bool kIsFirst, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &accum_s, Tensor1 &accum_o, float softmax_scale_log2)
    {
        Tensor scores = make_tensor(accum_s.data(), convert_layout_acc_rowcol(accum_s.layout()));
        if(kIsFirst)
        {
            row_reduce<kIsFirst>(scores, row_max, MaxOp<float>());
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            row_reduce<kIsFirst>(scores, row_sum, SumOp<float>());
        }
        else
        {
            Tensor scores_max_prev = make_fragment_like(row_max);
            copy(row_max, scores_max_prev);
            row_reduce<kIsFirst>(scores, row_max, MaxOp<float>());

            Tensor accum_o_rowcol = make_tensor(accum_o.data(), convert_layout_acc_rowcol(accum_o.layout()));
            static_assert(decltype(size<0>(accum_o_rowcol))::value == kNumRows);
            #pragma unroll
            for(int row = 0; row < size(row_max); ++row)
            {
                float row_max_cur = row_max(row);
                float scores_scale = exp2f((scores_max_prev(row) - row_max_cur) * softmax_scale_log2);
                row_sum(row) *= scores_scale;
                #pragma unroll
                for(int col = 0; col < size<1>(scores); ++col)
                {
                    accum_o_rowcol(row, col) *= scores_scale;
                }
            }
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            row_reduce<kIsFirst>(scores, row_sum, SumOp<float>());
        }
        


    }
};