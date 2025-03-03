import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_col_major_tensor

# C++ code templates
includes = ('"gemm/fp8_gemm.cuh"', )
template = """
using namespace yan;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
GemmType::run(out, rhs_scales, nullptr,
              m,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""