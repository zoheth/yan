import torch
from flash_attn import flash_attn_func

import yan
from yan import bench_kineto, calc_diff
from yan.jit_kernels import flash_attn_cute

def construct_run(num_heads: int, seq_len: int, head_dim: int):
    Q = torch.randn(32, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
    K = torch.randn(32, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
    V = torch.randn(32, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
    Output = torch.zeros(32, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
    flash_attn_cute(Q, K, V, Output)
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    # expected_output = flash_attn_func(Q, K, V, causal=True)
    expected_output = flash_attn_func(Q, K, V, causal=False)
    expected_output = expected_output.permute(0, 2, 1, 3)

def test_flash_attn():
    for seq_len in (1024,2048, ):
        for num_heads, head_dim in [(16, 64), (64, 128) ]:
            def test_func():
                construct_run(num_heads, seq_len, head_dim)
                
            kernel_names = ('flash_attn_cute', 'flash_fwd_kernel')
            t = bench_kineto(test_func, kernel_names, suppress_kineto_output=True)
            for i, time in enumerate(t):
                print(f' > Performance (N={seq_len:5}, h={num_heads:5}, d={head_dim:5}) {kernel_names[i]}: {time * 1e6:4.0f} us')
                
            print()

if __name__ == "__main__":
    yan.jit_kernels.flash_attn_cute_accuracy_test()
    test_flash_attn()
    