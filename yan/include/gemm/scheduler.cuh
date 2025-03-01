#pragma once

#include "utils.cuh"
#include <cinttypes>

namespace yan {
enum class GemmType {
  Normal,
  GroupedContiguous,
  GroupedMasked
};
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"

template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct Scheduler {
  int current_iter = -1;
  uint32_t num_aligned_m_blocks;

  uint32_t num_blocks;

  __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m) {
    num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
    if constexpr (kGemmType == GemmType::Normal) {
      num_blocks = num_aligned_m_blocks * kNumNBlocks;
    }
  }
};

} // namespace yan