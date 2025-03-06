import random
import torch
from typing import Tuple

import yan
from yan import bench_kineto, get_col_major_tensor, calc_diff


def construct(m: int, k: int, n: int) -> \
Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(m, k, device='cuda', dtype=torch.half)
    y = torch.randn(n, k, device='cuda', dtype=torch.half)
    out = torch.zeros(m, n, device='cuda', dtype=torch.half)
    ref_out = x @ y.t()
    
    # x, y = get_col_major_tensor(x), get_col_major_tensor(y)
    return x, y, out, ref_out

def test_cute_gemm():
    print('Testing CuTe GEMM:')
    
    for m in (256, 4096):
        for k, n in [(5120, 5120), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x, y, out, ref_out = construct(m, k, n)
            yan.gemm_fp16_tn(x, y, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            
            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x, y, out, ref_out = construct(m, k, n)
                yan.gemm_fp16_tn(x, y, out)
            
            t = bench_kineto(test_func, 'fp16_gemm_cute', suppress_kineto_output=True, flush_l2=False)
            # t = bench_kineto(test_func, 'ampere', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) * 2 / 1e9 / t:4.0f} GB/s')
    print()       

if __name__ == "__main__":
    # yan.jit_kernels.accuracy_test()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    random.seed(42)

    print('Library path:')
    print(f' > {yan.__path__}\n')

    test_cute_gemm()
    