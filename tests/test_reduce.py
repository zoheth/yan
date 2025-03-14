import torch

import yan
from yan import bench_kineto
  

def test_sum_reduce():
    print('Testing reduce sum:')

    # noinspection PyShadowingNames
    def test_func():
        N = 1024*4096*128
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y0 = torch.zeros(1, dtype=torch.float, device='cuda')
        y1 = torch.zeros(1, dtype=torch.float, device='cuda')
        
        yan.jit_kernels.reduce_sum_max(x, y0, y1)

    
    t = bench_kineto(test_func, ('SumOp<float>', 'MaxOp<float>'), suppress_kineto_output=True)
    for i, time in enumerate(t):
        print(f' > Performance {i}: {time * 1e6:4.0f} us')
    total_time = sum(t)
    print(f' > Total Performance: {total_time * 1e6:4.0f} us')
    
    # print(f' > Performance {t * 1e6:4.0f} us')


if __name__ == "__main__":
    yan.jit_kernels.accuracy_test()
    test_sum_reduce()
    