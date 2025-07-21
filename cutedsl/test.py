import time
from typing import Type

import torch
import torch.nn.functional as F
import cutlass
import cutlass.torch as cutlass_torch

from simple_softmax import softmax

if __name__ == "__main__":
    torch_dtype = cutlass_torch.dtype(cutlass.Float32)

    device = "cuda"
    M=500
    N=2048
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    out = softmax(x)
    ref = F.softmax(x, dim=1)
    
    if torch.allclose(out, ref, rtol=0.01, atol=0.001):
        print("Verification: PASSED ✅")
    else:
        print("Verification: FAILED ❌\n")
        print(x)
        print(out)