#include <cstdint>
#include <cute/numeric/integral_constant.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cutlass/device_kernel.h>

using namespace cute;

template <class ElementA,
          class ElementB,
          class SmemLayoutA, //(M,K,P)
          class SmemLayoutB> // (N,K,P)
struct SharedStorage
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smemA;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smemB;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ void gemm_sm90(ProblemShape shape_MNK, CtaTiler cta_tiler,
                          TA const *A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                          TB const *B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                          TC *C, CStride dC, TiledMma mma,
                          Alpha alpha, Beta beta)
{
    auto [M, N, K] = shape_MNK;
    Tensor mA      = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB      = tma_b.get_tma_tensor(make_shape(N, K));
    Tensor mC      = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto   cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA        = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB        = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC        = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor         sA   = make_tensor(make_smem_ptr(smem.smemA.data()), SmemLayoutA{});
    Tensor         sB   = make_tensor(make_smem_ptr(smem.smemB.data()), SmemLayoutB{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sB), group_modes<0, 2>(gB));

    constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                         CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);

    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier; // MMA

    CUTE_UNROLL
    for(int pipe=0; pipe < K_PIPE_MAX; ++pipe)
    {
        if((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }

    cluster_sync();

    CUTE_UNROLL
    for(int pipe=0; pipe < K_PIPE_MAXl ++pipe)
    {
        if((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCsC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_NO_UNROLL
    while(k_tile_count > -K_PIPE_MAX)
    {
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        warpgroup_commit_batch();

        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if((warp_idx == 0) && lane_predicate)
        {
            int pipe = write_state.index();

            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());

            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    axpby(alpha, tCrC, beta, tCgC);

}