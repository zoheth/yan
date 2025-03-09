import random
import torch
from typing import Tuple

import yan
from yan import bench_kineto
  

def test_naive_scan():
    print('Testing naive scan:')
    
    
            
    # noinspection PyShadowingNames
    def test_func():
        N = 1024*4096
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y = torch.zeros(N, dtype=torch.float, device='cuda')
        
        yan.jit_kernels.naive_scan(x, y)

    
    t = bench_kineto(test_func, ('scan_phase1','scan_phase2','scan_phase3'), suppress_kineto_output=True, flush_l2=True)
    # t = bench_kineto(test_func, 'tensorop_s1688gemm', suppress_kineto_output=True)
    for i, time in enumerate(t):
        print(f' > Performance {i}: {time * 1e6:4.0f} us')
    total_time = sum(t)
    print(f' > Total Performance: {total_time * 1e6:4.0f} us')


if __name__ == "__main__":
    yan.jit_kernels.accuracy_test()
    test_naive_scan()
    