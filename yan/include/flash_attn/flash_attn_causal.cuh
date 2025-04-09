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
__forceinline__ __device__ auto convert_reg_layout_c2a(Layout layout)
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
                                TiledMma tiled_mma, float scale, float scale_log2) __launch_bounds__(128, 2)
{
    using namespace cute;

    int offset = blockIdx.y * seq_len * d;

    Tensor mQ = make_tensor(make_gmem_ptr(Q + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(K + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(V + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mO = make_tensor(make_gmem_ptr(O + offset), make_shape(seq_len, d), make_stride(d, _1{}));

    Tensor gQ = local_tile(mQ, select<0, 2>(cta_tiler), make_coord(blockIdx.x, 0)); // (Br, d, Tr)
    Tensor gK = local_tile(mK, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, Tc)
    Tensor gV = local_tile(mV, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, Tc)
    Tensor gO = local_tile(mO, select<0, 2>(cta_tiler), make_coord(blockIdx.x, 0)); // (Br, d, Tr)

    static_assert(is_static<SmemLayoutQ>::value);
    static_assert(is_static<SmemLayoutKV>::value);

    extern __shared__ Element smem[];

    Tensor sQ = make_tensor(make_smem_ptr(smem), SmemLayoutQ{});   // (Br, d)  ((8, _), (block_d, _))
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{}); // (Bc, d)  ((8, _), (block_d, _))

    Tensor sV           = make_tensor(sK.data() + size(sK), SmemLayoutKV{});                             // (Bc, d)  ((8, _), (block_d, _))
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

    copy(copy_g2s, tQgQ, tQsQ);
    copy(copy_g2s, tKgK(_, _, _, 0), tKsK(_, _, _));
    cp_async_fence();

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);

    ThrCopy thr_copy_q_s2r = copyQ_s2r.get_slice(threadIdx.x);
    Tensor  tSsQ           = thr_copy_q_s2r.partition_S(sQ);
    Tensor  tSrQ_view      = thr_copy_q_s2r.retile_D(tSrQ);

    ThrCopy thr_copy_k_s2r = copyK_s2r.get_slice(threadIdx.x);
    Tensor  tSsK           = thr_copy_k_s2r.partition_S(sK);
    Tensor  tSrK_view      = thr_copy_k_s2r.retile_D(tSrK);

    ThrCopy thr_copy_vt_s2r = copyVT_s2r.get_thread_slice(threadIdx.x);
    Tensor  tOsVt           = thr_copy_vt_s2r.partition_S(sVt);
    Tensor  tOrVt           = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor  tOrVt_view      = thr_copy_vt_s2r.retile_D(tOrVt);

    Tensor  sO             = make_tensor(sQ.data(), sQ.layout());
    ThrCopy thr_copy_o_r2s = copyO_r2s.get_slice(threadIdx.x);
    Tensor  tOsO_r2s       = thr_copy_o_r2s.partition_D(sO);

    ThrCopy thr_copy_o_s2g = copyO_s2g.get_slice(threadIdx.x);
    Tensor  tOsO_s2g       = thr_copy_o_s2g.partition_S(sO);
    Tensor  tOgO      = thr_copy_o_s2g.partition_D(gO);

    const auto K_BLOCK_MAX = size<2>(tSrQ);

    Tensor accum_o = partition_fragment_C(tiled_mma, select<0, 2>(cta_tiler));
    clear(accum_o);

    Softmax<2 * size<1>(accum_o)> softmax_op;

    for (int j = 0; j < size<2>(gK); j++)
    {
        Tensor accum_s = partition_fragment_C(tiled_mma, select<0, 1>(cta_tiler));
        clear(accum_s);

        cp_async_wait<0>();
        __syncthreads();

        copy(copy_g2s, tVgV(_, _, _, j), tVsV(_, _, _));
        cp_async_fence();

        copy(copyK_s2r, tSsK, tSrK_view);
        copy(copyQ_s2r, tSsQ, tSrQ_view);
        gemm(tiled_mma, accum_s, tSrQ, tSrK, accum_s);

        cp_async_wait<0>();
        __syncthreads();

        if (j < size<2>(gK) - 1)
        {
            copy(copy_g2s, tKgK(_, _, _, j + 1), tKsK(_, _, _));
            cp_async_fence();
        }

        if (j == 0)
        {
            softmax_op.rescale_output<true>(accum_s, accum_o, scale_log2);
        } else
        {
            softmax_op.rescale_output<false>(accum_s, accum_o, scale_log2);
        }

        Tensor rP   = convert_type<Element>(accum_s);
        Tensor tOrP = make_tensor(rP.data(), convert_reg_layout_c2a(rP.layout()));

        auto K_BLOCK_PV = size<2>(tOrP);

        copy(copyVT_s2r, tOsVt(_, _, _), tOrVt_view(_, _, _));
        gemm(tiled_mma, accum_o, tOrP, tOrVt, accum_o);
    }
    softmax_op.normalize_output(accum_o, scale);

    Tensor  rO             = convert_type<Element>(accum_o);
    Tensor  tOrO_r2s       = thr_copy_o_r2s.retile_S(rO);

    copy(copyO_r2s, tOrO_r2s, tOsO_r2s);
    __syncthreads();

    copy(copyO_s2g, tOsO_s2g, tOgO);
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

    auto Br       = Int<128>{};
    auto Bc       = Int<32>{};
    auto kHeadDim = Int<HEAD_DIM>{};

    constexpr int kBlockHeadDim = 64;
    static_assert(HEAD_DIM % kBlockHeadDim == 0);

    auto cta_tiler = make_shape(Br, Bc, kHeadDim);

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kBlockHeadDim>{}),
                    make_stride(Int<kBlockHeadDim>{}, Int<1>{}))));

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
    TiledCopy copyO_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout

    int  seq_blocks = seq_len / Br;
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(seq_blocks, batch_size * num_heads);

    float scale      = 1.0f / sqrtf(HEAD_DIM);
    float scale_log2 = M_LOG2E * scale;

    static constexpr int kShmSize = (cosize(sQ) + cosize(sKV) * 2) * sizeof(Element);
    std::cout << "kShmSize: " << kShmSize / 1024.f << "KB" << std::endl;

    cudaFuncSetAttribute(flash_attn_cute<HEAD_DIM, decltype(cta_tiler),
                                         decltype(sQ), decltype(sKV), decltype(sVtransposed),
                                         decltype(tiled_mma),
                                         decltype(copyQKV_g2s), decltype(copyQ_s2r),
                                         decltype(copyK_s2r), decltype(copyVT_s2r),
                                         decltype(copyO_r2s), decltype(copyO_s2g)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    flash_attn_cute<HEAD_DIM, decltype(cta_tiler),
                    decltype(sQ), decltype(sKV), decltype(sVtransposed),
                    decltype(tiled_mma),
                    decltype(copyQKV_g2s), decltype(copyQ_s2r),
                    decltype(copyK_s2r), decltype(copyVT_s2r),
                    decltype(copyO_r2s), decltype(copyO_s2g)>
        <<<dimGrid, dimBlock, kShmSize, stream>>>(cta_tiler, reinterpret_cast<Element *>(query), reinterpret_cast<Element *>(key), reinterpret_cast<Element *>(value), reinterpret_cast<Element *>(output),
                                                  seq_len,
                                                  copyQKV_g2s, copyQ_s2r,
                                                  copyK_s2r, copyVT_s2r,
                                                  copyO_r2s, copyO_s2g,
                                                  tiled_mma, scale, scale_log2);
}