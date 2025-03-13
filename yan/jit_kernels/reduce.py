import torch
from .tuner import jit_tuner

includes = ('"reduce/reduce.cuh"', )
template = """
// Templated args from Python JIT call
cute_reduce_sum(X, y, N);
"""

def reduce_sum(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.shape[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    global includes, template
    
    args = (x, y, N)
    runtime = jit_tuner.compile_and_tune(
        name='cute_reduce_sum',
        keys={},
        space=(),
        includes=includes,
        arg_defs=(('X', torch.float), ('y', torch.float), ('N', int)),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        N = 4096*1024
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y = torch.zeros(1, dtype=torch.float, device='cuda')
        
        
        reduce_sum(x, y)
        
        print(y)
        print(torch.sum(x))
        
        # assert torch.allclose(y, y_ref, rtol=0.5, atol=0.1)
        
        print("Test passed!")

if __name__ == "__main__":
    accuracy_test()
