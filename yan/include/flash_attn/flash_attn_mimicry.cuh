#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

using namespace cute;

struct FlashParams
{
    using index_t = int64_t;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int batch_size;
    int seq_len;
    int num_heads;

    void *__restrict__ p_ptr;
};

template <int kHeadDim_, int kBlockM_, int kBlockN_,
          int kNumWarps_, bool kIsQInRegs_ = false, typename ElementType_ = cutlass::half_t>
struct FlashTraits
{
    using ElementType  = ElementType_;
    using ElementAccum = float;

    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kBlockM   = kBlockM_;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kNumWarps = kNumWarps_;

    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    static constexpr int kNumThreads = kNumWarps * 32;

    static_assert(kHeadDim % 32 == 0, "head维度必须是32的倍数");
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle    = kBlockKSmem == 32 ? 2 : 3;

    // MMA Tile的N和K维度相等，可以让A和C的shape相同，放便将矩阵乘法的结果C作为下一次乘法的A
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              Layout<Shape<Int<kNumWarps>{}, _1, _1>>,
                              Tile<Int<16 * kNumWarps>, _16, _16>>;

    using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                 Layout<Shape<_8, Int<kBlockKSmem>>,
                                                        Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV_Transposed = decltype(composition(
        SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutV_Transposed_NoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutV_Transposed{}));

    using SmemCopyAtom            = Copy_Atom<SM75_U32x4_LDSM_N, ElementType>;
    using SmemCopyAtom_Transposed = Copy_Atom<SM75_U16x8_LDSM_T, ElementType>;

    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                 Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                                                        Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutO         = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO       = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementType>;
    using SmemCopyAtomO_Accum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>;

    static constexpr int kSmemQSize  = size(SmemLayoutQ{}) * sizeof(ElementType);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(ElementType);
    static constexpr int kSmemSize   = kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(uint128_t) / sizeof(ElementType);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be divisible by kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNumThreads % kGmemThreadPerRow == 0, "kNumThreads must be divisible by kGmemThreadPerRow");
    using GmemLayoutAtom = Layout<Shape<Int<kNumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementType>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>()));

    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementType>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>()));
};

template <int kHeadDim, int kBlockM, int kBlockN, int kNumWarps>
inline __device__ void compute_attn_1rowblock(const FlashParams &params)
{
    using Traits  = FlashTraits<kHeadDim, kBlockM, kBlockN, kNumWarps>;
    using Element = typename Traits::ElementType;

    extern __shared__ char smem_bytes[];

    const int tid = threadIdx.x;

    const int row_block_idx = blockIdx.x;
    const int batch_idx     = blockIdx.y;
    const int head_idx      = blockIdx.z;

    const int64_t qkv_stride = params.num_heads * params.seq_len * kHeadDim;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + batch_idx * qkv_stride),
                            make_shape(params.seq_len, params.num_heads, kHeadDim),
                            make_stride(params.q_row_stride, params.q_head_stride, 1));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + batch_idx * qkv_stride),
                            make_shape(params.seq_len, params.num_heads, kHeadDim),
                            make_stride(params.k_row_stride, params.k_head_stride, 1));
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + batch_idx * qkv_stride),
                            make_shape(params.seq_len, params.num_heads, kHeadDim),
                            make_stride(params.v_row_stride, params.v_head_stride, 1));

    Tensor gQ = local_tile(mQ(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(row_block_idx, 0)); // (kBlockM, kHeadDim)

    Tensor gK = local_tile(mK(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0)); // (kBlockN, kHeadDim, numBlocksN)
    Tensor gV = local_tile(mV(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0)); // (kBlockN, kHeadDim, numBlocksN)

    int num_col_block = ceil_div(params.seq_len, kBlockN);

    const int64_t row_offset_p = (batch_idx * params.num_heads + head_idx) * params.seq_len + row_block_idx * kBlockM;
    row_offset_p               = row_offset_p * params.seq_len + (num_col_block - 1) * kBlockN;

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seq_len, 1));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_bytes)),
                            typename Traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Traits::SmemLayoutKV{});

    Tensor sV_transposed            = make_tensor(sV.data(),
                                                  typename Traits::SmemLayoutV_Transposed{});
    Tensor sV_transposed_no_swizzle = make_tensor(sV.data().get(),
                                                  typename Traits::SmemLayoutV_Transposed_NoSwizzle{});

    typename Traits::GmemTiledCopyQKV copy_qkv_g2s;
    ThrCopy                           thr_copy_qkv_g2s = copy_qkv_g2s.get_thread_slice(tid);

    Tensor tQgQ = thr_copy_qkv_g2s.partition_S(gQ);
    Tensor tQsQ = thr_copy_qkv_g2s.partition_D(sQ);
    Tensor tKgK = thr_copy_qkv_g2s.partition_S(gK);
    Tensor tKsK = thr_copy_qkv_g2s.partition_D(sK);
    Tensor tVgV = thr_copy_qkv_g2s.partition_S(gV);
    Tensor tVsV = thr_copy_qkv_g2s.partition_D(sV);

    typename Traits::TiledMma tiled_mma;

    auto   thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor tSrQ    = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK    = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt   = thr_mma.partition_fragment_C(sV_transposed_no_swizzle);

    Tensor tSgS = thr_mma.partition_c(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto   tiled_copyQ_s2r = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto   thr_copyQ_s2r   = tiled_copyQ_s2r.get_thread_slice(tid);
    Tensor tSsQ            = thr_copyQ_s2r.partition_S(sQ);

    auto   tiled_copyK_s2r = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto   thr_copyK_s2r   = tiled_copyK_s2r.get_thread_slice(tid);
    Tensor tSsK            = thr_copyK_s2r.partition_S(sK);

    auto   tiled_copyV_s2r = make_tiled_copy_B(typename Traits::SmemCopyAtom_Transposed{}, tiled_mma);
    auto   thr_copyV_s2r   = tiled_copyV_s2r.get_thread_slice(tid);
    Tensor tSsVt           = thr_copyV_s2r.partition_S(sV_transposed);

    copy(copy_qkv_g2s, tQgQ, tQsQ);

    int col_block_idx = num_col_block - 1;

    copy(copy_qkv_g2s, tKgK(_, _, _, col_block_idx), tKsK);
    copy(copy_qkv_g2s, tVgV(_, _, _, col_block_idx), tVsV);
    cp_async_fence();

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape < Int<kBlockM>, Int<kBlockN>{});
    clear(acc_s);

    Tensor tSrQ_copy_view = thr_copyQ_s2r.retile_D(tSrQ);
    Tensor tSrK_copy_view = thr_copyK_s2r.retile_D(tSrK);
    copy(tiled_copyQ_s2r, tSsQ(_, _, _0{}), tSrQ_copy_view(_, _, _0{}));
    copy(tiled_copyK_s2r, tSsK(_, _, _0{}), tSrK_copy_view(_, _, _0{}));
    for (int i = 0; i < size<2>(tSrQ); ++i)
    {
        if (i < size<2>(tSrQ) - 1)
        {
            copy(tiled_copyQ_s2r, tSsQ(_, _, i + 1), tSrQ_copy_view(_, _, i + 1));
            copy(tiled_copyK_s2r, tSsK(_, _, i + 1), tSrK_copy_view(_, _, i + 1));
        }
        gemm(tiled_mma, acc_s, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
    }


}