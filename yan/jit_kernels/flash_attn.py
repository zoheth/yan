import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .tuner import jit_tuner
from .utils import compare_tensors

includes = ('"flash_attn/flash_attn.cuh"', )
template = """
// Templated args from Python JIT call
constexpr auto d = {D};

flash_attn_func<d>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
"""

def flash_attn_cute(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor) -> None:
    batch_size = Q.shape[0]
    num_heads = Q.shape[1]
    seq_len = Q.shape[2]
    d = Q.shape[3]
    
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Output = Output.contiguous()
    
    # assert Q.shape == K.shape == V.shape == Output.shape
    # assert Q.dtype == K.dtype == V.dtype == Output.dtype == torch.half
    assert Q.device.type == "cuda"
    
    stream = torch.cuda.current_stream()

    global includes, template

    args = (Q, K, V, Output, batch_size, num_heads, seq_len, stream)
    runtime = jit_tuner.compile_and_tune(
        name='flash_attn_func',
        keys={'D': d},
        space=(),
        includes=includes,
        arg_defs=(('Q', torch.half), ('K', torch.half), ('V', torch.half), ('O', torch.half), ('batch_size', int), ('num_heads', int), ('seq_len', int), ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )

    runtime(*args)

def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        Q = torch.randn(32, 64, 1024, 128, device='cuda', dtype=torch.half)
        K = torch.randn(32, 64, 1024, 128, device='cuda', dtype=torch.half)
        V = torch.randn(32, 64, 1024, 128, device='cuda', dtype=torch.half)
        Output = torch.zeros(32, 64, 1024, 128, device='cuda', dtype=torch.half)
        # Temp = torch.zeros(32, 64, 1024, 1024, device='cuda', dtype=torch.half)
        flash_attn_cute(Q, K, V, Output)
        
        # Calculate expected output: softmax(Q * K^T / sqrt(d)) * V
        # expected_output = torch.zeros_like(Output)
        # for b in range(32):
        #     for h in range(64):
        #         scale_factor = 1.0 / math.sqrt(Q.shape[-1])
        #         temp = torch.matmul(Q[b, h], K[b, h].transpose(0, 1)) * scale_factor
        #         temp = torch.softmax(temp, dim=-1)
        #         expected_output[b, h] = torch.matmul(temp, V[b, h])

        expected_output = F.scaled_dot_product_attention(Q, K, V)
                
        compare_tensors(Output, expected_output, rtol=1e-3, atol=1e-3)