import torch
from torch.profiler import profile, record_function, ProfilerActivity

import yan
from yan.jit_kernels import simple_p2p


if __name__ == "__main__":
    yan.jit_kernels.simple_p2p_accuracy_test()
    torch.manual_seed(42)
    for _ in range(10):
        N = 1024*4096
        x = torch.randn(N, dtype=torch.float, device='cuda:0')
        y = torch.zeros(N, dtype=torch.float, device='cuda:1')
        simple_p2p(x, y)

    N = 1024*4096
    x = torch.randn(N, dtype=torch.float, device='cuda:0')
    y = torch.zeros(N, dtype=torch.float, device='cuda:1')
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        simple_p2p(x, y)
    prof.export_chrome_trace("trace.json")