#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "softmax.cuh"

#define CUTE_DEBUG

using namespace cute;
using Element = cutlass::half_t;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor)
{
    using From_type                                                 = typename Engine::value_type;
    constexpr int                                             numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <class Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout layout)
{
    using namespace cute;
    auto l = logical_divide(layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

template <int d, typename CtaTiler,
          typename SmemLayoutQ, typename SmemLayoutKV, typename SmemLayoutVtransposed,
          typename TiledMma,
          typename TiledCopyQKV_g2s, typename TiledCopyQ_s2r,
          typename TiledCopyK_s2r, typename TiledCopyVT_s2r,
          typename TiledCopyO_r2s, typename TiledCopyO_s2g>
__global__ void flash_attn_cute(CtaTiler cta_tiler, Element *Q, Element *K, Element *V, Element *O,
                                int seq_len, TiledCopyQKV_g2s copy_g2s, TiledCopyQ_s2r copyQ_s2r,
                                TiledCopyK_s2r copyK_s2r, TiledCopyVT_s2r copyVT_s2r,
                                TiledCopyO_r2s copyO_r2s, TiledCopyO_s2g copyO_s2g,
                                TiledMma tiled_mma)
{
    using namespace cute;
    int offset = blockIdx.x * seq_len * d;

    Tensor mQ = make_tensor(make_gmem_ptr(Q + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(K + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(V + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mO = make_tensor(make_gmem_ptr(O + offset), make_shape(seq_len, d), make_stride(d, _1{}));

    Tensor gQ = local_tile(mQ, select<0, 2>(cta_tiler), make_coord(_, 0)); // (Br, d, Tr)
    Tensor gK = local_tile(mK, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, Tc)
    Tensor gV = local_tile(mV, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, Tc)
    Tensor gO = local_tile(mO, select<0, 2>(cta_tiler), make_coord(_, 0)); // (Br, d, Tr)

    static_assert(is_static<SmemLayoutQ>::value);
    static_assert(is_static<SmemLayoutKV>::value);

    __shared__ Element smemQ[cosize_v<SmemLayoutQ>];
    __shared__ Element smemK[cosize_v<SmemLayoutKV>];
    __shared__ Element smemV[cosize_v<SmemLayoutKV>];

    Tensor sQ = make_tensor(make_smem_ptr(smemQ), SmemLayoutQ{});  // (Br, d)  ((8, _), (block_d, _))
    Tensor sK = make_tensor(make_smem_ptr(smemK), SmemLayoutKV{}); // (Bc, d)  ((8, _), (block_d, _))

    Tensor sV           = make_tensor(make_smem_ptr(smemV), SmemLayoutKV{});                             // (Bc, d)  ((8, _), (block_d, _))
    Tensor sVt          = make_tensor(sV.data(), SmemLayoutVtransposed{});                               // (d, Bc) ((block_d, _), Bc)
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), get_nonswizzle_portion(SmemLayoutVtransposed{})); // (d, Bc) ((block_d, _), Bc)

    ThrCopy thr_copy_q = copy_g2s.get_slice(threadIdx.x);
    Tensor  tQgQ       = thr_copy_q.partition_S(gQ);
    Tensor  tQsQ       = thr_copy_q.partition_D(sQ);

    ThrCopy thr_copy_k = copy_g2s.get_slice(threadIdx.x);
    Tensor  tKgK       = thr_copy_k.partition_S(gK);
    Tensor  tKsK       = thr_copy_k.partition_D(sK);

    ThrCopy thr_copy_v = copy_g2s.get_slice(threadIdx.x);
    Tensor  tVgV       = thr_copy_v.partition_S(gV);
    Tensor  tVsV       = thr_copy_v.partition_D(sV);

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor tPrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));
    Tensor tPrK = thr_mma.partition_fragment_B(gK(_, _, 0));

    ThrCopy thr_copy_q_s2r = copyQ_s2r.get_slice(threadIdx.x);
    Tensor  tPsQ           = thr_copy_q_s2r.partition_S(sQ);
    Tensor  tPrQ_view      = thr_copy_q_s2r.retile_D(tPrQ);

    ThrCopy thr_copy_k_s2r = copyK_s2r.get_slice(threadIdx.x);
    Tensor  tPsK           = thr_copy_k_s2r.partition_S(sK);
    Tensor  tPrK_view      = thr_copy_k_s2r.retile_D(tPrK);

    ThrCopy thr_copy_vt_s2r = copyVT_s2r.get_thread_slice(threadIdx.x);
    Tensor  tOsVt           = thr_copy_vt_s2r.partition_S(sVt);
    Tensor  tOrVt           = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor  tOrVt_view      = thr_copy_vt_s2r.retile_D(tOrVt);

    auto  K_BLOCK_MAX = size<2>(tPrQ);
    float scale_log2  = M_LOG2E;

    // Tensor accum_o = partition_fragment_C(tiled_mma, select<0, 2>(cta_tiler));
    // Tensor lse = make_fragment_like(Shape<Int<2 * size<1>(accum_o)>>{});

    for (int i = 0; i < size<2>(gQ); i++)
    {
        copy(copy_g2s, tQgQ(_, _, _, i), tQsQ(_, _, _));
        cp_async_fence();
        cp_async_wait<0>();
        Tensor accum_o = partition_fragment_C(tiled_mma, select<0, 2>(cta_tiler));
        clear(accum_o);

        Softmax<2 * size<1>(accum_o)> softmax_op;

        for (int j = 0; j < size<2>(gK); j++)
        {
            Tensor accum_s = partition_fragment_C(tiled_mma, select<0, 1>(cta_tiler));

            if (thread0() && i == 0 && j == 0)
            {
                print(accum_s);
                print("\n\n");
            }

            clear(accum_s);

            copy(copy_g2s, tKgK(_, _, _, j), tKsK(_, _, _));
            copy(copy_g2s, tVgV(_, _, _, j), tVsV(_, _, _));
            cp_async_fence();
            cp_async_wait<0>();

            __syncthreads();

            for (int k = 0; k < K_BLOCK_MAX; k++)
            {
                copy(copyQ_s2r, tPsQ(_, _, k), tPrQ_view(_, _, k));
                copy(copyK_s2r, tPsK(_, _, k), tPrK_view(_, _, k));

                gemm(tiled_mma, accum_s, tPrQ(_, _, k), tPrK(_, _, k), accum_s);
            }

            __syncthreads();
            // if (j == 0)
            // {
            //     softmax_op.softmax_rescale_o<true>(accum_s, accum_o, scale_log2);
            // } else
            // {
            //     softmax_op.softmax_rescale_o<false>(accum_s, accum_o, scale_log2);
            // }

            Tensor rP   = convert_type<Element>(accum_s);
            Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(rP.layout()));

            auto K_BLOCK_PV = size<2>(tOrP);
            for (int k = 0; k < K_BLOCK_PV; k++)
            {
                copy(copyVT_s2r, tOsVt(_, _, k), tOrVt_view(_, _, k));
                gemm(tiled_mma, accum_o, tOrP(_, _, k), tOrVt(_, _, k), accum_o);
            }
            __syncthreads();
        }

        // lse = softmax_op.normalize_softmax_lse(accum_o, 1.0f);
        // softmax_op.normalize_softmax_lse(accum_o, 1.0f);

        Tensor  rO             = convert_type<Element>(accum_o);
        Tensor  sO             = make_tensor(sQ.data(), sQ.layout());
        ThrCopy thr_copy_o_r2s = copyO_r2s.get_slice(threadIdx.x);
        Tensor  tOrO_r2s       = thr_copy_o_r2s.retile_S(rO);
        Tensor  tOsO_r2s       = thr_copy_o_r2s.partition_D(sO);

        copy(copyO_r2s, tOrO_r2s, tOsO_r2s);
        __syncthreads();

        ThrCopy thr_copy_o_s2g = copyO_s2g.get_slice(threadIdx.x);
        Tensor  tOsO_s2g       = thr_copy_o_s2g.partition_S(sO);
        Tensor  tOgO_view      = thr_copy_o_s2g.partition_D(gO(_, _, i));

        copy(copyO_s2g, tOsO_s2g, tOgO_view);
    }
}

template <int HEAD_DIM>
void flash_attn_func(half *query, half *key, half *value, half *output, int batch_size, int num_heads, int seq_len, cudaStream_t stream)
{
    // key: [batch_size, num_heads, seq_len, head_dim]
    // value: [batch_size, num_heads, seq_len, head_dim]
    // query: [batch_size, num_heads, seq_len, head_dim]
    // output: [batch_size, num_heads, seq_len, head_dim]
    // temp: [batch_size, num_heads, seq_len, seq_len]
    using namespace cute;

    auto prob_shape = make_shape(batch_size, num_heads, seq_len, HEAD_DIM);

    auto Br       = Int<64>{};
    auto Bc       = Int<64>{};
    auto kHeadDim = Int<HEAD_DIM>{};

    constexpr int kBlockHeadDim = 64;
    static_assert(HEAD_DIM % kBlockHeadDim == 0);

    auto cta_tiler = make_shape(Br, Bc, kHeadDim);

    using SmemLayoutAtom = decltype(make_layout(make_shape(Int<8>{}, Int<kBlockHeadDim>{}),
                                                make_stride(Int<kBlockHeadDim>{}, Int<1>{})));

    auto sQ           = tile_to_shape(SmemLayoutAtom{}, make_shape(Br, kHeadDim));
    auto sKV          = tile_to_shape(SmemLayoutAtom{}, make_shape(Bc, kHeadDim));
    auto sVtransposed = composition(sKV, make_layout(make_shape(kHeadDim, Bc), GenRowMajor{}));

    TiledMMA  tiled_mma   = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                           Layout<Shape<_4, _1, _1>>{},
                                           Tile<_64, _16, _16>{});
    TiledCopy copyQKV_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                                            Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout
                                            Layout<Shape<_1, _8>>{});                 // Val layout

    TiledCopy copyQ_s2r  = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    TiledCopy copyK_s2r  = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    TiledCopy copyVT_s2r = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, tiled_mma);

    TiledCopy copyO_r2s = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, Element>{}, tiled_mma);
    TiledCopy copyO_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<int>, Element>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout

    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(batch_size * num_heads);

    flash_attn_cute<HEAD_DIM, decltype(cta_tiler),
                    decltype(sQ), decltype(sKV), decltype(sVtransposed),
                    decltype(tiled_mma),
                    decltype(copyQKV_g2s), decltype(copyQ_s2r),
                    decltype(copyK_s2r), decltype(copyVT_s2r),
                    decltype(copyO_r2s), decltype(copyO_s2g)>
        <<<dimGrid, dimBlock, 0, stream>>>(cta_tiler, reinterpret_cast<Element *>(query), reinterpret_cast<Element *>(key), reinterpret_cast<Element *>(value), reinterpret_cast<Element *>(output),
                                           seq_len,
                                           copyQKV_g2s, copyQ_s2r,
                                           copyK_s2r, copyVT_s2r,
                                           copyO_r2s, copyO_s2g,
                                           tiled_mma);
}