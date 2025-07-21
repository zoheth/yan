import math
import torch
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from reduction_base import ReductionBase, torch2cute_dtype_map

import utils

class Softmax(ReductionBase):
    def __init__(
        self, dtype: Type[cutlass.Numeric], N: int, reduction_dtype=cutlass.Float32
    ):
        self.dtype = dtype
        self.N = N
        self.reduction_dtype = reduction_dtype
    
    def _get_tv_layout(self):
        copy_bits = 128
        vecsize = copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        
        num_blocks_N = cute.ceil_div(self.N // vecsize, 128)
        
        tiler = (1, self.N)
        tv_layout = cute.make_layout(
            shape=((num_threads, 1), (vecsize, num_blocks_N)),
            stride=(
                (vecsize, 1),
                (1, vecsize * num_threads)
            )
        )
        return tiler, tv_layout
        
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        tiler, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        
        self.kernel(mX, mO, tv_layout, tiler).launch(
            grid=[mX.shape[0], 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )
        
    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        smem = cutlass.utils.SmemAllocator()
        gX, gO = [cute.local_tile(mT, tiler, (bidx, 0)) for mT in (mX, mO)]
        
        copy_atom_load_X = cute.make_copy_atom(
            # cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )
        
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler).get_slice(tidx)
        
        tXgX = thr_copy_X.partition_S(gX)
        tXgO = thr_copy_O.partition_D(gO)
        
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]
        
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            cute.make_layout((num_warps)),
            byte_alignment=4
        )
        
        cute.copy(copy_atom_load_X, tXgX, tXrX)
        x = tXrX.load().to(cute.Float32)
        max_x = utils.row_reduce(
            x,
            cute.ReductionOp.MAX,
            reduction_buffer,
            init_val=-cutlass.Float32.inf,
        )
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
        denom = utils.row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            reduction_buffer,
            init_val=0.0
        )
        y = exp_x * (1.0 / denom)
        tXrO.store(y.to(tXrO.element_type))
        cute.copy(copy_atom_store_O, tXrO, tXgO)
        

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
    convert_from_dlpack = lambda tensor: (
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
    def forward(ctx, x):
        y = _softmax_fwd(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        pass

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax forward pass with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N)

    Returns:
        Softmax output tensor of same shape as x
    """
    return SoftmaxFunction.apply(x)