import random
import torch
from typing import Tuple

import yan
from yan import bench_kineto, get_col_major_tensor, calc_diff


def construct(m: int, k: int, n: int) -> \
Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    y = torch.randn(n, k, device='cuda', dtype=torch.float32)
    out = torch.zeros(m, n, device='cuda', dtype=torch.float32)
    ref_out = x @ y.t()
    
    x, y = get_col_major_tensor(x), get_col_major_tensor(y)
    return x, y, out, ref_out

def test_simple_gemm():
    print('Testing simple GEMM:')
    
    for m in (128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x, y, out, ref_out = construct(m, k, n)
            yan.gemm_float_nt(x, y, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            
            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x, y, out, ref_out = construct(m, k, n)
                yan.gemm_float_nt(x, y, out)
            
            t = bench_kineto(test_func, 'gemm_cute', suppress_kineto_output=True, flush_l2=True)
            # t = bench_kineto(test_func, 'tensorop_s1688gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) * 4 / 1e9 / t:4.0f} GB/s')
    print()       

if __name__ == "__main__":
    # yan.jit_kernels.test_simple_gemm_()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    random.seed(42)

    print('Library path:')
    print(f' > {yan.__path__}\n')

    test_simple_gemm()