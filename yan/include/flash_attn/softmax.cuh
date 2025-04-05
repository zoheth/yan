#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

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

template <>
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
    CUTE_UNROLL
    for (int offset = kNumThread / 2; offset > 0; offset /= 2)
    {
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <bool kIsFirst = true, typename Tensor0, typename Tensor1, class ReduceOp>
__device__ __forceinline__ void row_thread_reduce(Tensor0 const &tensor, Tensor1 &row_summary, ReduceOp op)
{
    CUTE_STATIC_ASSERT_V(size<0>(row_summary) == size<0>(tensor));
    CUTE_UNROLL
    for (int row = 0; row < size<0>(tensor); ++row)
    {
        row_summary(row) = kIsFirst ? tensor(row, 0) : op(row_summary(row), tensor(row, 0));
        CUTE_UNROLL
        for (int col = 1; col < size<1>(tensor); ++col)
        {
            row_summary(row) = op(row_summary(row), tensor(row, col));
        }
    }
}

template <typename Tensor1, class ReduceOp>
__device__ __forceinline__ void row_warp_reduce(Tensor1 &row_summary, ReduceOp op)
{
    CUTE_STATIC_ASSERT_V(size(row_summary) == size(row_summary));
    CUTE_UNROLL
    for (int row = 0; row < size(row_summary); ++row)
    {
        // MMA Tiled N=16 线程排布以第一行为例，则为 0011223300112233 001122...
        // 每一行四个连续线程，此处做跨线程reduce
        row_summary(row) = warp_reduce<4>(row_summary(row), op);
    }
}

template <bool kIsFirst = true, typename Tensor0, typename Tensor1, class ReduceOp>
__device__ __forceinline__ void row_reduce(Tensor0 const &tensor, Tensor1 &row_summary, ReduceOp op)
{
    row_thread_reduce<kIsFirst>(tensor, row_summary, op);
    row_warp_reduce(row_summary, op);
}

// Convert from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
{
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

template <typename Tensor0, typename Tensor1>
__forceinline__ __device__ void scale_apply_exp2(Tensor0 &tensor, Tensor1 &row_max, float scale_log2)
{
    CUTE_UNROLL
    for (int row = 0; row < size<0>(tensor); ++row)
    {
        const float row_max_val = row_max(row) == -INFINITY ? 0.0f : row_max(row) * scale_log2;
        CUTE_UNROLL
        for (int col = 0; col < size<1>(tensor); ++col)
        {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(row, col) = ::exp2f(tensor(row, col) * scale_log2 - row_max_val);
        }
    }
}

template <int kNumRows>
struct Softmax
{
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNumRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template <bool kIsFirst, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void rescale_output(Tensor0 &accum_s, Tensor1 &accum_o, float softmax_scale_log2)
    {
        Tensor scores = make_tensor(accum_s.data(), convert_layout_acc_rowcol(accum_s.layout()));
        if (kIsFirst)
        {
            row_reduce<kIsFirst>(scores, row_max, MaxOp<float>());
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            row_thread_reduce<kIsFirst>(scores, row_sum, SumOp<float>());
        } else
        {
            Tensor scores_max_prev = make_fragment_like(row_max);
            copy(row_max, scores_max_prev);
            row_reduce<kIsFirst>(scores, row_max, MaxOp<float>());

            Tensor accum_o_rowcol = make_tensor(accum_o.data(), convert_layout_acc_rowcol(accum_o.layout()));
            static_assert(decltype(size<0>(accum_o_rowcol))::value == kNumRows);
            CUTE_UNROLL
            for (int row = 0; row < size(row_max); ++row)
            {
                float row_max_cur  = row_max(row);
                float scores_scale = ::exp2f((scores_max_prev(row) - row_max_cur) * softmax_scale_log2);
                row_sum(row) *= scores_scale;
                CUTE_UNROLL
                for (int col = 0; col < size<1>(accum_o_rowcol); ++col)
                {
                    accum_o_rowcol(row, col) *= scores_scale;
                }
            }
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            row_thread_reduce<kIsFirst>(scores, row_sum, SumOp<float>());
        }
    }

    template <typename Tensor0>
    __forceinline__ __device__ TensorT normalize_output(Tensor0 &accum_o, float softmax_scale)
    {
        row_warp_reduce(row_sum, SumOp<float>());
        TensorT lse            = make_fragment_like(row_sum);
        Tensor  accum_o_rowcol = make_tensor(accum_o.data(), convert_layout_acc_rowcol(accum_o.layout()));
        static_assert(decltype(size<0>(accum_o_rowcol))::value == kNumRows);
        CUTE_UNROLL
        for (int row = 0; row < size(row_max); ++row)
        {
            float val     = row_sum(row);
            float inv_val = (val == 0.f || val != val) ? 1.0f : 1.0f / val;
            lse(row)      = (val == 0.f || val != val) ? INFINITY : row_max(row) * softmax_scale + __logf(val);

            CUTE_UNROLL
            for (int col = 0; col < size<1>(accum_o_rowcol); ++col)
            {
                accum_o_rowcol(row, col) *= inv_val;
            }
        }
        return lse;
    }
};