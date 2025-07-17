import math
import torch
from typing import Type, Callable

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cutedsl.reduction_base import ReductionBase, torch2cute_dtype_map

TensorConverter = Callable[[torch.Tensor], cute.Tensor]

class Softmax(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=cutlass.Float32 if not online_softmax else cutlass.Int64,
        )
        self.online_softmax = online_softmax
        
    def _set_cluster_n(self):
        N = self.N
        
        
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self.
        
        
def _softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    """Softmax forward pass.
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Softmax output tensor of same shape as x
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    M, N = x.shape
    out = torch.empty_like(x)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack: TensorConverter = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N)
    if not hasattr(_softmax_fwd, "compile_cache"):
        _softmax_fwd.compile_cache = {}
    if compile_key not in _softmax_fwd.compile_cache:
        softmax_op = Softmax(dtype, N)
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            softmax_op, x_tensor, out_tensor, current_stream
        )
    _softmax_fwd.compile_cache[compile_key](x_tensor, out_tensor, current_stream)
    return out
        
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        
        
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Backward pass for softmax is not implemented")
        
def softmax(x: torch.Tensor) -> torch.Tensor:
    return SoftmaxFunction.apply(x)

