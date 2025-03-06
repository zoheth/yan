import torch
from typing import Tuple

from .tuner import jit_tuner

includes = ('"gemm/cute_gemm.cuh"', )
template = """
gemm_tn(m, n, k,
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
        keys={},
        space=(),
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
        m, n, k = 16384, 7168, 256
        a = torch.randn(m, k, dtype=torch.half, device='cuda')
        b = torch.randn(n, k, dtype=torch.half, device='cuda')
        c = torch.zeros(m, n, dtype=torch.half, device='cuda')
        # m, n, k = 128, 256, 256
        # a = torch.zeros(m, k, dtype=torch.half, device='cuda')
        # b = torch.zeros(n, k, dtype=torch.half, device='cuda')
        # a[:, ::4] = 1.0
        
        # b[:, ::3] = 1.0
        
        # c = torch.zeros(m, n, dtype=torch.half, device='cuda')
        
        c_ref = a @ b.t()
        gemm_fp16_tn(a, b, c)
        print(c)
        print(c_ref)
        assert torch.allclose(c, c_ref, rtol=0.5, atol=0.1)
        
        print("Simple GEMM test passed!")

if __name__ == "__main__":
    accuracy_test()
