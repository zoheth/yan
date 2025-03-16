import torch
import torch.nn.functional as F
from .tuner import jit_tuner

includes = ('"softmax/online_softmax.cuh"', )
template = """
// Templated args from Python JIT call
online_softmax_c(X, Y, B, N);
"""

def softmax(x: torch.Tensor, y: torch.Tensor) -> None:
    B, N = x.shape[0], x.shape[1]
    assert x.shape == y.shape
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    global includes, template
    
    args = (x, y, B, N)
    runtime = jit_tuner.compile_and_tune(
        name='softmax',
        keys={},
        space=(),
        includes=includes,
        arg_defs=(('X', torch.float), ('Y', torch.float),('B', int), ('N', int)),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        N = 16384
        B = 512
        x = torch.randn(B, N, dtype=torch.float, device='cuda')
        y = torch.zeros(B, N, dtype=torch.float, device='cuda')
        
        y_ref = F.softmax(x, dim=1)
        
        softmax(x, y)
        
        assert torch.allclose(y, y_ref, rtol=0.0003, atol=0.0001)
        
        print("Test passed!")

if __name__ == "__main__":
    accuracy_test()
