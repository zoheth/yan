#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

__global__ void
scaleKernel(const float *input, float *output, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = input[i] * scale;
    }
}

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

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

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

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

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

    CUTE_STATIC_ASSERT_V((shape(tCrA) == take<0, 3>(shape(tCsA)))); // (MMA,MMA_M,MMA_K)
    CUTE_STATIC_ASSERT_V((shape(tCrB) == take<0, 3>(shape(tCsB)))); // (MMA,MMA_N,MMA_K)
    CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0, 3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));         // MMA_M
    CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));         // MMA_N
    CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));         // MMA_K

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
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    // auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
    auto dC = make_stride(ldC, Int<1>{}); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<3>{};                      // Pipeline

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK, bP)); // (m,k,p) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK, bP)); // (n,k,p) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));     // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)

    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
                                      Layout<Shape<_4, _1>>{}); // Val layout  4x1 m-major
    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                                      Layout<Shape<_4, _1>>{}); // Val layout  4x1 n-major

    TiledMMA mmaC = make_tiled_mma(UniversalFMA<float, float, float>{},
                                   Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));
    gemm_cute<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler,
                                                A, dA, sA, copyA,
                                                B, dB, sB, copyB,
                                                C, dC, sC, mmaC,
                                                alpha, beta);
}