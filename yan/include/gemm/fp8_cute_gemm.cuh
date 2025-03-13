#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

#include "mma_utils.cuh"

template <class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TiledCopyA_g2s, class TiledCopyA_s2r,
          class SmemLayoutB, class TiledCopyB_g2s, class TiledCopyB_s2r,
          class SmemLayoutC, class TiledCopyC_s2g, class TiledCopyB_r2s,
          class TiledMma>

__global__ void
fp8_gemm_cute(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    __nv_fp8_e4m3 const *A, SmemLayoutA sA_layout, TiledCopyA_g2s copy_a_g2s, TiledCopyA_s2r copy_a_s2r,
    __nv_fp8_e4m3 const *B, SmemLayoutB sB_layout, TiledCopyB_g2s copy_b_g2s, TiledCopyB_s2r copy_b_s2r,
    float *C, SmemLayoutC sC_layout, TiledCopyC_s2g copy_c_s2g, TiledCopyB_r2s copy_c_r2s,
    TiledMma tiled_mma)
{
    using namespace cute;

    
}