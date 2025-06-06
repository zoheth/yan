import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import compare_tensors

includes = ('"gemm/cute_gemm.cuh"', )
template = """
// Templated args from Python JIT call
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};

gemm_tn<BLOCK_M, BLOCK_N, kNumStages>(m, n, k,
        A,
        B,
        C,
        stream);
"""

def gemm_fp16_tn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    m, k = a.shape
    n, k_ = b.shape
    m_, n_ = c.shape

    
    assert m == m_ and n == n_ and k == k_
    assert a.dtype == torch.half and b.dtype == torch.half
    
    if m == 0 or n == 0:
        return
    
    stream = torch.cuda.current_stream()
    

    global includes, template
    
    args = (m, n, k, a, b, c, stream)
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp16_tn',
        keys={'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3},
        space=(
            # {'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 2},
            # {'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2},
            # {'BLOCK_M': 128, 'BLOCK_N': 256, 'NUM_STAGES': 2},
            # {'BLOCK_M': 256, 'BLOCK_N': 64, 'NUM_STAGES': 2},
            # {'BLOCK_M': 256, 'BLOCK_N': 128, 'NUM_STAGES': 2},
            # {'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 3},
            # {'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3},
            # {'BLOCK_M': 128, 'BLOCK_N': 256, 'NUM_STAGES': 3},
            # {'BLOCK_M': 256, 'BLOCK_N': 64, 'NUM_STAGES': 3},
            # {'BLOCK_M': 256, 'BLOCK_N': 128, 'NUM_STAGES': 3},
        ),
        includes=includes,
        arg_defs=(('m', int), ('n', int), ('k', int),
                  ('A', torch.half),
                  ('B', torch.half),
                  ('C', torch.half),
                  ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        m, n, k = 4096, 7168, 2048
        a = torch.randn(m, k, dtype=torch.half, device='cuda')
        b = torch.randn(n, k, dtype=torch.half, device='cuda')
        c = torch.zeros(m, n, dtype=torch.half, device='cuda')
        # m, n, k = 128, 256, 256
        # a = torch.zeros(m, k, dtype=torch.half, device='cuda')
        # b = torch.zeros(n, k, dtype=torch.half, device='cuda')
        # a[:, ::4] = 1.0
        
        # b[:, ::3] = 1.0
        
        # c = torch.zeros(m, n, dtype=torch.half, device='cuda')
        
        gemm_fp16_tn(a, b, c)
        c_ref = a @ b.t()

        compare_tensors(c, c_ref, rtol=0.5, atol=0.1)
        

if __name__ == "__main__":
    accuracy_test()
