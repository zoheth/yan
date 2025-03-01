import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms

includes = ('"gemm/simple_gemm.cuh"',  )
template = """
constexpr auto BLOCK_SIZE = {BLOCK_SIZE};
int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
scaleKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, scale, n);
"""

def scale_tensor(input_tensor: torch.Tensor, 
                 output_tensor: torch.Tensor, 
                 scale: float) -> None:
    assert input_tensor.shape == output_tensor.shape
    assert input_tensor.dtype == output_tensor.dtype == torch.float32
    assert input_tensor.is_contiguous() and output_tensor.is_contiguous()
    
    n = input_tensor.numel()
    if n == 0:
        return
    
    block_size = 256
    
    num_sms = get_num_sms()
    
    args = (input_tensor, output_tensor, scale, n, torch.cuda.current_stream())
    runtime = jit_tuner.compile_and_tune(
        name='scale_tensor',
        keys={'BLOCK_SIZE': block_size},
        space=(),
        includes=includes,
        arg_defs=(('input', torch.float), ('output', torch.float), 
                 ('scale', float), ('n', int), ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )
    
    runtime(*args)

def test_jit():
    n = 1000000
    input_tensor = torch.rand(n, dtype=torch.float32, device='cuda')
    output_tensor = torch.zeros(n, dtype=torch.float32, device='cuda')
    scale_factor = 2.5

    scale_tensor(input_tensor, output_tensor, scale_factor)

    expected = input_tensor * scale_factor
    assert torch.allclose(output_tensor, expected), "JIT内核测试失败!"
    print("JIT内核测试成功!")

if __name__ == "__main__":
    test_jit()
