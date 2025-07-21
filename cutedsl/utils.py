import operator
import math
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import from_dlpack

@cute.jit
def warp_reduce(
    val: cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1<<i))
    return val

@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    if lane_idx == 0:
        reduction_buffer[warp_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < reduction_buffer.shape:
        block_reduce_val = reduction_buffer[warp_idx]
    return warp_reduce(block_reduce_val, op)

@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    op: cute.ReductionOp,
    reduction_buffer: Optional[cute.Tensor] = None,
    init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    val = x.reduce(op, init_val=init_val, reduction_profile=0)
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if cutlass.const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = warp_reduce(
        val,
        warp_op,
    )
    if cutlass.const_expr(reduction_buffer is not None):
        if cutlass.const_expr(reduction_buffer.shape > 1):
            val = block_reduce(val, warp_op, reduction_buffer, init_val)
    return val
    

@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(
        flat_stride
    ), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)