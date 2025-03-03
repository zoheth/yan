import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms

includes = ('"gemm/simple_gemm.cuh"',  )
template = """
gemm_nt(m, n, k, 
        alpha, 
        static_cast<float const*>(A), ldA, 
        static_cast<float const*>(B), ldB, 
        beta, 
        static_cast<float*>(C), ldC, 
        stream);
"""

def gemm_float_nt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
                 alpha: float = 1.0, beta: float = 0.0) -> None:
    m, k = a.shape
    n, k_ = b.shape
    m_, n_ = c.shape
    
    assert m % 128 == 0 and n % 128 == 0
    
    assert m == m_ and n == n_ and k == k_
    assert a.dtype == torch.float32 and b.dtype == torch.float32 and c.dtype == torch.float32
    
    if m == 0 or n == 0:
        return
    
    stream = torch.cuda.current_stream()
    
    ldA = m
    ldB = n
    ldC = n

    global includes, template
    
    args = (m, n, k, alpha, a, ldA, b, ldB, beta, c, ldC, stream)
    runtime = jit_tuner.compile_and_tune(
        name='gemm_float_nt',
        keys={},
        space=(),
        includes=includes,
        arg_defs=(('m', int), ('n', int), ('k', int), 
                  ('alpha', float), 
                  ('A', torch.float), ('ldA', int), 
                  ('B', torch.float), ('ldB', int), 
                  ('beta', float), 
                  ('C', torch.float), ('ldC', int), 
                  ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )
    
    runtime(*args)

def test_simple_gemm_():
    torch.manual_seed(42)
    m, n, k = 5120, 2048, 4096
    a = torch.randn(m, k, dtype=torch.float32, device='cuda')
    b = torch.randn(n, k, dtype=torch.float32, device='cuda')
    c = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    c_ref = a @ b.t()
    a = a.t().contiguous().t()
    b = b.t().contiguous().t()
    gemm_float_nt(a, b, c)
    print(c)
    print(c_ref)
    assert torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2)
    
    print("Simple GEMM test passed!")

if __name__ == "__main__":
    test_simple_gemm_()
