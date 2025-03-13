#pragma once

#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

#include "utils.cuh"

namespace cute
{

// MMA 16x8x32 TN for FP8 (e4m3)
struct SM89_16x8x32_E4M3E4M3F32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM89_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_E4M3E4M3F32F32_TN without CUTE_ARCH_MMA_SM89_ENABLED");
#endif
  }
};

template <>
struct MMA_Traits<SM89_16x8x32_E4M3E4M3F32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  
  // Layouts for FP8 e4m3 data with k=32
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  // Row major layout for F32 accumulator (4 registers for 16x8 output)
  using CLayout = Layout<Shape <Shape < _4,_2>>,
                         Stride<Stride<_1,_4>>>;
};
}