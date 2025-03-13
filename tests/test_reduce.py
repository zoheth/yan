import torch

import yan
from yan import bench_kineto
  

def test_sum_reduce():
    print('Testing reduce sum:')

    # noinspection PyShadowingNames
    def test_func():
        N = 1024*4096
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y = torch.zeros(1, dtype=torch.float, device='cuda')
        
        yan.jit_kernels.reduce_sum(x, y)

    
    t = bench_kineto(test_func, ('cute_reduce_kernel','cute_final_reduce_kernel'), suppress_kineto_output=True)
    # t = bench_kineto(test_func, 'tensorop_s1688gemm', suppress_kineto_output=True)
    for i, time in enumerate(t):
        print(f' > Performance {i}: {time * 1e6:4.0f} us')
    total_time = sum(t)
    print(f' > Total Performance: {total_time * 1e6:4.0f} us')


if __name__ == "__main__":
    yan.jit_kernels.accuracy_test()
    test_sum_reduce()
    