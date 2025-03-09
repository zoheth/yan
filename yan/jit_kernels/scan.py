import torch
from typing import Tuple

from .tuner import jit_tuner

includes = ('"scan/naive_scan.cuh"', )
template = """
// Templated args from Python JIT call
constexpr auto BLOCK_SIZE = {BLOCK_SIZE};

naive_scan_c<BLOCK_SIZE>(X, Y, N);
"""

def naive_scan(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.shape[0]
    assert N == y.shape[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    global includes, template
    
    args = (x, y, N)
    runtime = jit_tuner.compile_and_tune(
        name='naive_scan',
        keys={'BLOCK_SIZE': 1024},
        space=(),
        includes=includes,
        arg_defs=(('X', torch.float), ('Y', torch.float), ('N', int)),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        N = 1024*4096
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y = torch.zeros(N, dtype=torch.float, device='cuda')
        
        y_ref = torch.cumsum(x, 0)
        
        naive_scan(x, y)
        
        print(y)
        print(y_ref)
        
        assert torch.allclose(y, y_ref, rtol=0.5, atol=0.1)
        
        print("Test passed!")

if __name__ == "__main__":
    accuracy_test()
