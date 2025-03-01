#pragma once

#include <cuda.h>

#include "utils.cuh"

namespace yan {

// SM89 MMA operator for FP8 E4M3 x E4M3 -> FP32
struct SM89_16x8x32_F32E4M3E4M3_SS {
  __device__ static void mma(uint32_t const (&a)[4], uint32_t const (&b)[2],
                             float &d0, float &d1, float &d2, float &d3, float &c0, float &c1, float &c2, float &c3) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  }
};

__forceinline__ __device__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

template <int N>
struct FP8MMASelector {
  static_assert(N % 8 == 0, "N must be a multiple of 8 for SM89 FP8 MMA operations");
  static_assert(N >= 8, "N must be at least 8 for SM89 FP8 MMA operations");

  using type = SM89_16x8x32_F32E4M3E4M3_SS;
};
} // namespace yan