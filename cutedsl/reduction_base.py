# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import torch
from typing import Type, Tuple, Optional

import cutlass
import cutlass.cute as cute


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

class ReductionBase:
    def __init__(
        self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=cutlass.Float32
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype
        self.cluster_n = 1

    def _calculate_threads_per_row(self):
        raise NotImplementedError()
    
    def _set_cluster_n(self):
        raise NotImplementedError()
    
    def _get_num_threads(self):
        return 128 if self.N <= 16384 else 256
    
    def _get_tv_layout(self):
        copy_bits = 128
        vec_size = copy_bits // self.dtype.width
        assert self.N % vec_size == 0, f"N {self.N} must be divisible by {vec_size}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        
        block_x = self._calculate_threads_per_row()
        block_y =  cute.ceil_div(self.N // vec_size, block_x * self.cluster_n)
        cols_per_block = 
        
    
    