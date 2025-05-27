import torch
import torch.nn.functional as F
import time
import numpy as np

import yan
from yan.jit_kernels import tirplane_sampler

def initialize_data(H, W, C, N, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    weightxy = torch.randn(H, W, C, device='cuda', dtype=torch.float)
    weightyz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
    weightxz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
    
    grid_in = torch.randn(N, 2, device='cuda', dtype=torch.float)
    grid_mid = torch.randn(N, 2, device='cuda', dtype=torch.float)
    grid_out = torch.randn(N, 2, device='cuda', dtype=torch.float)
    
    return weightxy, weightyz, weightxz, grid_in, grid_mid, grid_out

def run_torch_implementation(weightxy_perm, weightyz_perm, weightxz_perm, grid_in, grid_mid, grid_out):
    grids = [grid_in, grid_mid, grid_out] * 3
    weights = [weightxy_perm, weightxy_perm, weightxy_perm, 
              weightyz_perm, weightyz_perm, weightyz_perm, 
              weightxz_perm, weightxz_perm, weightxz_perm]
    
    features = []
    for grid, weight in zip(grids, weights):
        sampled = F.grid_sample(
            weight.unsqueeze(0), 
            grid.unsqueeze(0).unsqueeze(0), 
            align_corners=True,
        )  # (1, C, 1, N)
        
        sampled = sampled.squeeze(0).squeeze(-2).permute(1, 0)
        features.append(sampled)

    result = torch.cat([
            features[0], features[3], features[6],
            features[1], features[4], features[7],
            features[2], features[5], features[8],
        ], dim=1)
    
    return result

def print_statistics(method_name, times):
    print(f"\n{method_name}:")
    print(f"  Mean: {np.mean(times):.4f} ms")
    print(f"  Median: {np.median(times):.4f} ms")
    print(f"  Min: {np.min(times):.4f} ms")
    print(f"  Max: {np.max(times):.4f} ms")
    print(f"  Std Dev: {np.std(times):.4f} ms")

def benchmark_test(num_runs=100, warmup_runs=10):
    
    C = 4
    H = 2443
    W = 3
    N = 1003264
    
    method1_times = []
    method2_times = []
    
    print("Performing warmup runs...")
    for _ in range(warmup_runs):
        weightxy, weightyz, weightxz, grid_in, grid_mid, grid_out = initialize_data(H, W, C, N)
        
        sample_output = torch.zeros(9, N, C, device='cuda', dtype=torch.float)
        final_output = torch.zeros(N, C*9, device='cuda', dtype=torch.float)
        grid_cute = torch.cat([grid_in, grid_mid, grid_out] * 3, dim=0)
        
        weightxy_perm = weightxy.permute(2, 0, 1).float().contiguous()
        weightyz_perm = weightyz.permute(2, 0, 1).float().contiguous()
        weightxz_perm = weightxz.permute(2, 0, 1).float().contiguous()
        
        tirplane_sampler(weightxy, weightyz, weightxz, grid_cute, sample_output, final_output)
        torch.cuda.empty_cache()
        
        run_torch_implementation(weightxy_perm, weightyz_perm, weightxz_perm, grid_in, grid_mid, grid_out)
        torch.cuda.empty_cache()
    
    print(f"Performing {num_runs} benchmark runs...")
    for run in range(num_runs):
        torch.manual_seed(run)
        weightxy, weightyz, weightxz, grid_in, grid_mid, grid_out = initialize_data(H, W, C, N, seed=run)
        
        assert weightxy.is_contiguous()
        assert weightyz.is_contiguous()
        assert weightxz.is_contiguous()
        
        sample_output = torch.zeros(9, N, C, device='cuda', dtype=torch.float)
        final_output = torch.zeros(N, C*9, device='cuda', dtype=torch.float)
        grid_cute = torch.cat([grid_in, grid_mid, grid_out] * 3, dim=0)
        assert grid_cute.is_contiguous()
        
        weightxy_perm = weightxy.permute(2, 0, 1).float().contiguous()
        weightyz_perm = weightyz.permute(2, 0, 1).float().contiguous()
        weightxz_perm = weightxz.permute(2, 0, 1).float().contiguous()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        tirplane_sampler(weightxy, weightyz, weightxz, grid_cute, sample_output, final_output)
        
        torch.cuda.synchronize()
        end_time = time.time()
        method1_times.append((end_time - start_time) * 1000)  
        torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        result2 = run_torch_implementation(weightxy_perm, weightyz_perm, weightxz_perm, grid_in, grid_mid, grid_out)
        
        torch.cuda.synchronize()
        end_time = time.time()
        method2_times.append((end_time - start_time) * 1000)
        torch.cuda.empty_cache()
        
        if run % 10 == 0:
            is_close = torch.allclose(final_output, result2, rtol=0.0003, atol=0.0001)
            if not is_close:
                print(f"Warning: Results don't match on run {run}")
    
    
    print("\n--- Benchmark Results (time in ms) ---")
    print_statistics("Yan's tirplane_sampler", method1_times)
    print_statistics("Torch implementation", method2_times)
    
    speedup = np.mean(method2_times) / np.mean(method1_times)
    print(f"\nYan is {speedup:.2f}x faster than Torch" if speedup > 1 else 
          f"\nTorch is {1/speedup:.2f}x faster than Yan")

    return np.mean(method1_times), np.mean(method2_times)

if __name__ == "__main__":
    yan.jit_kernels.tirplane_accuracy_test()
    benchmark_test(num_runs=100, warmup_runs=10)