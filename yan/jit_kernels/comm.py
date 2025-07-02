import torch
from .tuner import jit_tuner
from .utils import compare_tensors

includes = ('"comm/simple_p2p.cuh"', )
template = """
// Templated args from Python JIT call

simple_p2p(X, Y, N);
"""

def simple_p2p(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.shape[0]
    assert N == y.shape[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    global includes, template
    
    args = (x, y, N)
    runtime = jit_tuner.compile_and_tune(
        name='simple_p2p',
        space=(),
        includes=includes,
        arg_defs=(('X', torch.float), ('Y', torch.float), ('N', int)),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available. Found {gpu_count} GPU(s).")
    
    for _ in range(1):
        torch.manual_seed(42)
        N = 1024*4096
        x = torch.randn(N, dtype=torch.float, device='cuda:0')
        y = torch.zeros(N, dtype=torch.float, device='cuda:1')
        
        y_ref = torch.ones_like(y, device='cuda:1')
        y_ref = y_ref * 2
        simple_p2p(x, y)
        
        compare_tensors(y, y_ref, rtol=1e-5, atol=1e-6)
        
        print("Test passed!")

if __name__ == "__main__":
    accuracy_test()
