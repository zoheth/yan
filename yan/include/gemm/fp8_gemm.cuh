#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include "scheduler.cuh"

namespace yan {

enum class Layout {
  kRowMajor,
  kColMajor
};

template <uint32_t kNumLoaderThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
  static_assert(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
  return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumLoaderThreads;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumLoaderThreads, uint32_t kNumMathThreadsPerGroup,
          GemmType kGemmType>

__global__ void __launch_bounds__(get_num_threads_per_sm<kNumLoaderThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
    fp8_gemm_kernel(__nv_bfloat16 *gmem_d, float *scales_b, int *grouped_layout,
                    uint32_t shape_m,
                    __nv_fp8_e4m3 *gmem_a, __nv_fp8_e4m3 *gmem_b, float *scales_a) {

  using MMA = typename FP8MMASelector<BLOCK_N>::type;

  // Shared memory
  static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
  static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_K * BLOCK_N * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
  static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
  static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);

  // Configs
  constexpr uint32_t kFullOfAllStages = kNumStages * BLOCK_K;
  constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumLoaderThreads, kNumMathThreadsPerGroup>(BLOCK_M);
  constexpr uint32_t kNumMathThreads = kNumThreads - kNumLoaderThreads;
  constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullOfAllStages);
  const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane_idx = get_lane_id();

  __syncwarp();

  // Align to 1024 bytes for swizzle-128B
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  static_assert(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

  // Data on shared memory
  auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
  __nv_fp8_e4m3* smem_a[kNumStages];
  __nv_fp8_e4m3* smem_b[kNumStages];
  float* smem_scales_a[kNumStages];
  float* smem_scales_b;

  
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          GemmType kGemmType>
class Gemm {
public:
  Gemm() = default;

  static void run(__nv_bfloat16 *gmem_d,
                  float *scales_b,
                  int *grouped_layout,
                  uint32_t shape_m,
                  __nv_fp8_e4m3 *gmem_a,
                  __nv_fp8_e4m3 *gmem_b,
                  float *scales_a,
                  cudaStream_t stream,
                  int num_sms,
                  uint32_t smem_size) {
    // 使用4个warp作为加载器
    constexpr uint32_t kNumLoaderThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K,
                                  BLOCK_M, BLOCK_N, BLOCK_K,
                                  kNumGroups, kNumStages,
                                  kNumLoaderThreads, kNumMathThreadsPerGroup,
                                  kGemmType>;
  }

private:
  using
};
} // namespace yan

#pragma clang diagnostic pop