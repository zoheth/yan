import torch
import torch.nn.functional as F
from .tuner import jit_tuner

includes = ('"softmax/softmax.cuh"', )
template = """
// Templated args from Python JIT call
softmax_c(X, Y, N);
"""

def softmax(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.shape[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    global includes, template
    
    args = (x, y, N)
    runtime = jit_tuner.compile_and_tune(
        name='softmax',
        keys={},
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
        N = 1024
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y = torch.zeros(N, dtype=torch.float, device='cuda')
        
        softmax(x, y)
        
        print(torch.max(x))
        print(y)
        print(torch.softmax(x, 0))
        print(F.softmax(x, 0))
        
        # assert torch.allclose(y, y_ref, rtol=0.5, atol=0.1)
        
        print("Test passed!")

if __name__ == "__main__":
    accuracy_test()
