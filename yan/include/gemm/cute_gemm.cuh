#pragma once

#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BStride, class BSmemLayout, class TiledCopyB,
          class CStride, class CSmemLayout, class TiledMma>

__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_cute(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    float const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    float *C, CStride dC, CSmemLayout, TiledMma mma,
    float alpha, float beta)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (K,N)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLM_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
    {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0)
        {
            ++k_tile_next;
        }
    }

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0)); // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0)); // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

    // Clear the accumulators
    clear(tCrC);

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
        copy(tCsA_p(_, _, Int<0>{}), tCrA(_, _, Int<0>{}));
        copy(tCsB_p(_, _, Int<0>{}), tCrB(_, _, Int<0>{}));
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
            copy(tCsA_p(_, _, k_block_next), tCrA(_, _, k_block_next));
            copy(tCsB_p(_, _, k_block_next), tCrB(_, _, k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
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
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    axpby(alpha, tCrC, beta, tCgC);
}

void
gemm_nt(int m, int n, int k,
        float alpha,
        float const *A, int ldA,
        float const *B, int ldB,
        float beta,
        float *C, int ldC,
        cudaStream_t stream = 0)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto bM = Int<128>{};
    auto bN = Int<256>{};
    auto bK = Int<32>{};
    auto bP = Int<2>{}; // Pipeline

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<bK>{}),
                    make_stride(Int<bK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<bM>{}, Int<bK>{}, Int<bP>{})));

    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<bN>{}, Int<bK>{}, Int<bP>{})));



    TiledCopy copyA_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                      Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                      Layout<Shape<_1, _8>>{});                 // Val layout
    TiledCopy copyB_g2s  = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                      Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                      Layout<Shape<_1, _8>>{});                 // Val layout
    
    using CopyAtomA_s2r = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using CopyAtomB_s2r = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    using CopyAtomC_r2s = Copy_Atom<UniversalCopy<int>, half_t>;

    TiledCopy copyC_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
                                      Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout
                                      Layout<Shape<_1, _8>>{});                   // Val layout

    TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                   Layout<Shape<_2, _2, _1>>{},
                                   Tile<_32, _32, _16>{});

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(N, bN)),
                 size(ceil_div(M, bM)));
    gemm_cute<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler,
                                                A, dA, sA, copyA,
                                                B, dB, sB, copyB,
                                                C, dC, sC, mmaC,
                                                alpha, beta);
}