#pragma once

#include <cstdint>
#include <sys/types.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd>
__device__ __host__ void outer_launch_k_iterations(const auto &inner_launch_k_iterations, const auto &func, uint32_t num_former_iters)
{
    if (num_former_iters < kNumFormerIters)
    {
        inner_launch_k_iterations(func, cute::Int<kNumFormerIters>{});
        return;
    }
    if constexpr (kNumFormerIters + kGap <= kEnd)
    {
        outer_launch_k_iterations<kNumFormerIters + kGap, kGap, kEnd>(inner_launch_k_iterations, func, num_former_iters);
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_N_PADDING,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
    fp8_gemm_kernel(float *scales_b, int *grouped_layout,
                    uint32_t                            shape_m,
                    const __grid_constant__ CUtensorMap tensor_map_a,
                    const __grid_constant__ CUtensorMap tensor_map_b,
                    const __grid_constant__ CUtensorMap tensor_map_scales_a,
                    const __grid_constant__ CUtensorMap tensor_map_d)
{
    static_assert(BLOCK_K == 128);
    static_assert(ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    using WGMMA   = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static_assert(BLOCK_M % WGMMA::M == 0);

    static constexpr bool     kMustUseUniformedScaleB      = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE                  = BLOCK_M * (BLOCK_N + BLOCK_N_PADDING) * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE        = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE        = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES               = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE           = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads       = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads   = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations    = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t     warp_idx          = cutlass::canonical_warp_idx_sync();
    const uint32_t     lane_idx          = get_lane_id();

    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&tensor_map_scales_a));

        if constexpr (kSwizzleDMode > 0)
        {
            cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&tensor_map_d));
        }
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    static_assert(SMEM_D_SIZE % 1024 == 0);

    auto           smem_d = reinterpret_cast<__nv_bfloat16 *>(smem_buffer);
    __nv_fp8_e4m3 *smem_a[kNumStages];
    __nv_fp8_e4m3 *smem_b[kNumStages];
    float         *smem_scales_a[kNumStages];
    float         *smem_scales_b;

    Barrier *full_barriers[kNumStages];
    Barrier *empty_barriers[kNumStages];

#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        smem_a[i]        = reinterpret_cast<__nv_fp8_e4m3 *>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i]        = reinterpret_cast<__nv_fp8_e4m3 *>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
                                                      i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float *>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
                                                     i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float *>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    auto barrier_start_ptr = reinterpret_cast<Barrier *>(reinterpret_cast<uint8_t *>(smem_scales_b) + SMEM_SCALES_B_SIZE);
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        full_barriers[i]  = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    static_assert(kNumTMAMulticast <= 32);
    if (threadIdx.x == kNumMathThreads)
    {
// NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
// even with TMA multicast disabled, we want to make the behavior aligned
#pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            // todo why
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    struct DivisibleK
    {
    };
    struct NonDivisibleK
    {
    };
    struct SkipComputation
    {
    };
    struct NotSkipComputation
    {
    };
    auto lauch_k_iteration = [](const auto &func, bool skip_computation, uint32_t num_former_iters) {
        // todo why
        constexpr bool     kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
        constexpr uint32_t kGap            = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
        constexpr uint32_t kEnd            = kShouldOptimize ? BLOCK_K / 8 : 0;

        outer_launch_k_iteration<0, kGap, kEnd>([=](const auto &func, auto num_former_iters_type) {
            if (skip_computation)
            {
                for (uint32_t k_iter = 0; k_iter < kNumIterations; ++k_iter)
                    func(k_iter, DivisibleK{}, SkipComputation{}, num_former_iters_type);
            } else if (SHAPE_K % kFullKOfAllStages == 0)
            {
                for (uint32_t k_iter = 0; k_iter < kNumIterations; ++k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            } else
            {
                for (uint32_t k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
                func(kNumIterations - 1, NonDivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            }
        },
                                                func, kShouldOptimize ? num_former_iters : 0);
    };

    constexpr uint32_t kNumTMARegisters  = 40;
    constexpr uint32_t kNumMathRegisters = 232; // 192+40

    uint32_t m_block_idx, n_block_idx;

    auto scheduler = Scheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA>(
        shape_m, grouped_layout);

    if (threadIdx.x >= kNumMathThreads)
    {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (threadIdx.x == kNumMathThreads)
        {
            while (scheduler.get_nexr_block(m_block_idx, n_block_idx))
            {
                lauch_k_iteration([&](uint32_t k_iter, auto divisible_type, auto _, auto __) {
                    constexpr bool     kHasDivisibleStages = std::is_same_v<decltype(divisible_type), DivisibleK>;
                    constexpr uint32_t kNumInnerStages     = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;

                    const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                    const 
                });
            }
        }
    }
}