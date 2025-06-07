import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .tuner import jit_tuner
from .utils import compare_tensors

includes = ('"flash_attn/flash_attn_tk.cuh"',)
template = """
// Templated args from Python JIT call
constexpr auto d = {D};

flash_attn_func<d>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
"""


def flash_attn_tk(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor
) -> None:
    batch_size = Q.shape[0]
    num_heads = Q.shape[2]
    seq_len = Q.shape[1]
    d = Q.shape[3]

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Output = Output.contiguous()

    # assert Q.shape == K.shape == V.shape == Output.shape
    # assert Q.dtype == K.dtype == V.dtype == Output.dtype == torch.bfloat16
    assert Q.device.type == "cuda"

    stream = torch.cuda.current_stream()

    global includes, template

    args = (Q, K, V, Output, batch_size, num_heads, seq_len, stream)
    runtime = jit_tuner.compile_and_tune(
        name="flash_attn_func",
        keys={"D": d},
        space=(),
        includes=includes,
        arg_defs=(
            ("Q", torch.bfloat16),
            ("K", torch.bfloat16),
            ("V", torch.bfloat16),
            ("O", torch.bfloat16),
            ("batch_size", int),
            ("num_heads", int),
            ("seq_len", int),
            ("stream", torch.cuda.Stream),
        ),
        template=template,
        args=args,
    )

    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        shape = (32, 1024, 1, 64)  # (batch_size, seq_len, num_heads, head_dim)
        dtype = torch.bfloat16
        
        Q = torch.randn(*shape, device="cuda", dtype=dtype)
        K = torch.randn(*shape, device="cuda", dtype=dtype)
        V = torch.randn(*shape, device="cuda", dtype=dtype)
        Output = torch.zeros(*shape, device="cuda", dtype=dtype)
        flash_attn_tk(Q, K, V, Output)

        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()
        Output = Output.permute(0, 2, 1, 3)
        expected_output = F.scaled_dot_product_attention(
            Q, K, V
        )

        passed = compare_tensors( Output, expected_output, mode="similarity", diff_tolerance=1e-4)
        # if(not passed):
        #     print("Expected output:")
        #     print(expected_output[31, 63, 1023, :10])
        #     print("Actual output:")
        #     print(Output[31, 63, 1023, :10])