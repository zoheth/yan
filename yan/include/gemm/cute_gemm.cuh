#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using cute::half_t;

template <class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TiledCopyA, class CopyAtomA_s2r,
          class SmemLayoutB, class TiledCopyB, class CopyAtomB_s2r,
          class SmemLayoutC, class TiledCopyC, class CopyAtomC_r2s,
          class TiledMma>

__global__ void
fp16_gemm_cute(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    half_t const *A, SmemLayoutA sA_layout, TiledCopyA copy_a, CopyAtomA_s2r copy_a_s2r,
    half_t const *B, SmemLayoutB sB_layout, TiledCopyB copy_b, CopyAtomB_s2r copy_b_s2r,
    half_t *C, SmemLayoutC sC_layout, TiledCopyC copy_c, CopyAtomC_r2s copy_c_r2s,
    TiledMma mma)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), make_stride(select<2>(shape_MNK), _1));
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), make_stride(select<2>(shape_MNK), _1));
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), make_stride(select<1>(shape_MNK), _1));

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLM_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    __shared__ half_t smemA[cosize_v<ASmemLayout>];
    __shared__ half_t smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA_copy = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA_copy = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB_copy = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB_copy = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    auto K_PIPE_MAX = size<3>(tAsA_copy);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA_copy);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
    {
        copy(copy_a, tAgA_copy(_, _, _, k_tile_next), tAsA_copy(_, _, _, k_pipe));
        copy(copy_b, tBgB_copy(_, _, _, k_tile_next), tBsB_copy(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgc = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N,MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(tCgc);        // (MMA, MMA_M,MMA_N)
    // Clear the accumulators
    clear(tCrC);

    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto tiled_copy_a_s2r = make_tiled_copy_A(copy_a_s2r, mma);
    auto thr_copy_a_s2r = tiled_copy_a_s2r.get_slice(threadIdx.x);
    auto tAsA = thr_copy_a_s2r.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = thr_copy_a_s2r.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto tiled_copy_b_s2r = make_tiled_copy_B(copy_b_s2r, mma);
    auto thr_copy_b_s2r = tiled_copy_b_s2r.get_slice(threadIdx.x);
    auto tBsB = thr_copy_b_s2r.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
    auto tCrB_view = thr_copy_b_s2r.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();

    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Pipe slice
    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1)
    {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_, _, Int<0>{}), tCrA_view(_, _, Int<0>{}));
        copy(tCsB_p(_, _, Int<0>{}), tCrB_view(_, _, Int<0>{}));
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                tCsB_p = tCsB(_, _, _, smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
            copy(tiled_copy_a_s2r, tCsA_p(_, _, k_block_next), tCrA_view(_, _, k_block_next));
            copy(tiled_copy_b_s2r, tCsB_p(_, _, k_block_next), tCrB_view(_, _, k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copy_a, tAgA_copy(_, _, _, k_tile_next), tAsA_copy(_, _, _, smem_pipe_write));
                copy(copy_b, tBgB_copy(_, _, _, k_tile_next), tBsB_copy(_, _, _, smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
            }
            // Thread-level register gemm for k_block
            gemm(mma, tCrC, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    auto sC = make_tensor(sA(_, _, smem_pipe_read).data(), sC_layout);

    auto tiled_copy_c_r2s = make_tiled_copy_C(copy_c_r2s, mma);
    auto thr_copy_c_r2s = tiled_copy_c_r2s.get_slice(threadIdx.x);
    auto tCrC_r2s = thr_copy_c_r2s.retile_S(tCrC);
    auto tCsC_r2s = thr_copy_c_r2s.partition_D(sC);

    auto thr_copy_c_s2g = copy_c.get_slice(threadIdx.x);
    auto tCsC_s2g = thr_copy_c_s2g.partition_S(sC);
    auto tCgC_s2g = thr_copy_c_s2g.partition_D(gC);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    int step = size<3>(tCsC_r2s);
    CUTE_UNROLL
    for (int i = 0; i<size<1>(tCrC_r2sx); i += step)
    {
        CUTE_UNROLL
        for (int j = 0; i<step; ++j)
        {
            auto t = make_tensor_like<half_t>(tCrC_r2sx(_, i+ j));
            copy(tCrC_r2sx(_, i + j), t);

            copy(tiled_copy_c_r2s, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

        CUTE_UNROLL
        for(int j=0; j<step; ++j)
        {
            copy(copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i+j));
        }
        __syncthreads();
    }
}

void
gemm_nt(int m, int n, int k,
        half_t const *A,
        half_t const *B,
        half_t *C,
        cudaStream_t stream = 0)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto bM = Int<128>{};
    auto bN = Int<256>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<2>{}; // Pipeline
    auto kSmemLayoutCBatch = Int<4>{};

    using SmemLayoutAtomAnB = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<bK>{}),
                    make_stride(Int<bK>{}, Int<1>{}))));

    auto sA = tile_to_shape(SmemLayoutAtomAnB{},
                            make_shape(Int<bM>{}, Int<bK>{}, Int<bP>{}));

    auto sB = tile_to_shape(SmemLayoutAtomAnB{},
                            make_shape(Int<bN>{}, Int<bK>{}, Int<bP>{}));

    TiledCopy copyA_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                          Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout
    TiledCopy copyB_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                          Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout

    auto copyAtomA_s2r = Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
    auto copyAtomB_s2r = Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};

    TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                   Layout<Shape<_2, _2, _1>>{},
                                   Tile<_32, _32, _16>{});
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<32>{}),
                    make_stride(Int<16>{}, Int<1>{}))));
    auto sC = tile_to_shape(SmemLayoutAtomC{},
                            make_shape(Int<bM>{}, Int<bN>{}, Int<kSmemLayoutCBatch>{}));
    TiledCopy copyC_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
                                          Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout
    auto copyAtomC_r2s = Copy_Atom<UniversalCopy<int>, half_t>{};

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(N, bN)),
                 size(ceil_div(M, bM)));

    fp16_gemm_cute<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler,
                                                     A, sA, copyA_g2s, copyAtomA_s2r,
                                                     B, sB, copyB_g2s, copyAtomB_s2r,
                                                     C, sC, copyC_s2g, copyAtomC_r2s,
                                                     mmaC);
}