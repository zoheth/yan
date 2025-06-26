import torch
from flash_attn import flash_attn_func

import yan
from yan import bench_kineto, calc_diff
from yan.jit_kernels import flash_attn_tk

def construct_run(num_heads: int, seq_len: int, head_dim: int):
    batch_size = 32
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    Output = torch.zeros(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    timings = torch.zeros([128, 64], device="cuda", dtype=torch.int32)
    flash_attn_tk(Q, K, V, Output, timings)

    # expected_output = flash_attn_func(Q, K, V, causal=True)
    expected_output = flash_attn_func(Q, K, V, causal=False)
    

def test_flash_attn():
    for seq_len in (1024,2048, ):
        for num_heads, head_dim in [(16, 64), (64, 128) ]:
            def test_func():
                construct_run(num_heads, seq_len, head_dim)
                
            kernel_names = ('attend_ker', 'flash_fwd_kernel')
            t = bench_kineto(test_func, kernel_names, suppress_kineto_output=True)
            for i, time in enumerate(t):
                print(f' > Performance (N={seq_len:5}, h={num_heads:5}, d={head_dim:5}) {kernel_names[i]}: {time * 1e6:4.0f} us')
                
            print()

if __name__ == "__main__":
    yan.jit_kernels.flash_attn_tk_accuracy_test()
    test_flash_attn()

    