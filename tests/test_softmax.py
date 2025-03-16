import torch
import torch.nn.functional as F

import yan
from yan import bench_kineto

def test_softmax():
    print('Testing softmax:')

    # noinspection PyShadowingNames
    def test_func():
        B = 512
        N = 16384
        x = torch.randn(B, N, dtype=torch.float, device='cuda')
        y = torch.zeros(B, N, dtype=torch.float, device='cuda')
        yan.jit_kernels.softmax(x, y)
        y_ref = torch.softmax(x, dim=1)
        assert torch.allclose(y, y_ref, rtol=0.0003, atol=0.0001)

    
    t = bench_kineto(test_func, ('online_softmax_kernel'), suppress_kineto_output=True, flush_l2=True)
    # t = bench_kineto(test_func, ('cunn_SoftMaxForward'), suppress_kineto_output=True, flush_l2=True)
    # for i, time in enumerate(t):
    #     print(f' > Performance {i}: {time * 1e6:4.0f} us')
    # total_time = sum(t)
    # print(f' > Total Performance: {total_time * 1e6:4.0f} us')
    
    print(f' > Performance {t * 1e6:4.0f} us')

if __name__ == "__main__":
    yan.jit_kernels.accuracy_test()
    test_softmax()
    