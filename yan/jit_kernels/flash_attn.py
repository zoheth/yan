import torch
from typing import Tuple

from .tuner import jit_tuner

includes = ('"flash_attn/flash_attn.cuh"', )
template = """
// Templated args from Python JIT call
constexpr auto d = {D};

flash_attn_func<d>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
"""

def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor) -> None:
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
        Output = torch.zeros(32, 96, 1024, 128, device='cuda', dtype=torch.half)
        # Temp = torch.zeros(32, 64, 1024, 1024, device='cuda', dtype=torch.half)
        flash_attn(Q, K, V, Output)
        
        print(Output[15, 53])

    # expected_temp = torch.zeros_like(Temp)
    # for b in range(32):
    #     for h in range(64):
    #         expected_temp[b, h] = torch.matmul(Q[b, h], K[b, h].transpose(0, 1))
            
    # print(Temp[15, 53])
    # print(expected_temp[15, 53])
    
    # diff = torch.abs(Temp - expected_temp)
    # max_diff = diff.max()
    # mean_diff = diff.mean()


    # max_indices = (diff == max_diff).nonzero()[0].tolist()
    # b, h, r, c = max_indices

    # diff_flat = diff.flatten()
    # values, indices = diff_flat.topk(2)
    # second_max_indices = torch.unravel_index(indices[1], diff.shape)
    # b2, h2, r2, c2 = [i.item() for i in second_max_indices]

    # print(f"Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
    # print(f"Max diff at [{b},{h},{r},{c}]: {Temp[b,h,r,c]:.6f} vs {expected_temp[b,h,r,c]:.6f}")
    # print(f"Second max diff at [{b2},{h2},{r2},{c2}]: {Temp[b2,h2,r2,c2]:.6f} vs {expected_temp[b2,h2,r2,c2]:.6f}")

    # tolerance = 0.2
    # print("Verification", "PASSED" if max_diff < tolerance else "FAILED")