import torch
import torch.nn.functional as F
import time
import numpy as np

import yan
from yan.jit_kernels import tirplane_sampler

def benchmark_test(num_runs=100, warmup_runs=10):
    
    C = 4
    H = 2443
    W = 3
    N = 1003264
    # N = 334464
    
    # Initialize timing arrays
    method1_times = []
    method2_times = []
    
    # Warmup runs
    print("Performing warmup runs...")
    for _ in range(warmup_runs):
        # Initialize data
        # torch.manual_seed(42)
        weightxy = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        weightyz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        weightxz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        
        grid_in = torch.randn(N, 2, device='cuda', dtype=torch.float)
        grid_mid = torch.randn(N, 2, device='cuda', dtype=torch.float)
        grid_out = torch.randn(N, 2, device='cuda', dtype=torch.float)
        
        sample_output = torch.zeros(9, N, C, device='cuda', dtype=torch.float)
        final_output = torch.zeros(N, C*9, device='cuda', dtype=torch.float)
        grid_cute = torch.cat([grid_in, grid_mid, grid_out] * 3, dim=0)
        
        # Run both methods once to warm up GPU
        tirplane_sampler(weightxy, weightyz, weightxz, grid_cute, sample_output, final_output)
        torch.cuda.empty_cache()
        weightxy = weightxy.permute(2, 0, 1).float().contiguous()
        weightyz = weightyz.permute(2, 0, 1).float().contiguous()
        weightxz = weightxz.permute(2, 0, 1).float().contiguous()
        
        # Method 2print
        grids = [grid_in, grid_mid, grid_out] * 3
        weights = [weightxy, weightxy, weightxy, weightyz, weightyz, weightyz, weightxz, weightxz, weightxz]
        
        features = []
        for grid, weight in zip(grids, weights):
            sampled = F.grid_sample(
                weight.unsqueeze(0), 
                grid.unsqueeze(0).unsqueeze(0), 
                align_corners=True,
            ) # (1, C, 1, N)
            # x_coords = grid[:, 0]
            # y_coords = grid[:, 1]
            # coord_sum = x_coords + y_coords
            # sampled = torch.zeros(1, C, 1, N, device='cuda', dtype=torch.float)
            # for c in range(C):
            #     sampled[0, c, 0, :] = coord_sum
            
            sampled = sampled.squeeze(0).squeeze(-2).permute(1, 0)
            features.append(sampled)
        torch.cuda.empty_cache()
    
    # Actual benchmark runs
    print(f"Performing {num_runs} benchmark runs...")
    for run in range(num_runs):
        # Initialize data with same seed for fair comparison
        torch.manual_seed(run)  # Different seed each run but same for both methods
        weightxy = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        weightyz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        weightxz = torch.randn(H, W, C, device='cuda', dtype=torch.float)
        
        grid_in = torch.randn(N, 2, device='cuda', dtype=torch.float)
        grid_mid = torch.randn(N, 2, device='cuda', dtype=torch.float)
        grid_out = torch.randn(N, 2, device='cuda', dtype=torch.float)
        
        # Method 1: tirplane_sampler + tensor chunking
        sample_output = torch.zeros(9, N, C, device='cuda', dtype=torch.float)
        final_output = torch.zeros(N, C*9, device='cuda', dtype=torch.float)
        grid_cute = torch.cat([grid_in, grid_mid, grid_out] * 3, dim=0)
        
        # Ensure contiguity
        assert weightxy.is_contiguous()
        assert weightyz.is_contiguous()
        assert weightxz.is_contiguous()
        assert grid_cute.is_contiguous()
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.time()

        tirplane_sampler(weightxy, weightyz, weightxz, grid_cute, sample_output, final_output)

        
        # Synchronize after timing
        torch.cuda.synchronize()
        end_time = time.time()
        method1_times.append(end_time - start_time)
        torch.cuda.empty_cache()
        weightxy = weightxy.permute(2, 0, 1).float().contiguous()
        weightyz = weightyz.permute(2, 0, 1).float().contiguous()
        weightxz = weightxz.permute(2, 0, 1).float().contiguous()
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        grids = [grid_in, grid_mid, grid_out] * 3
        weights = [weightxy, weightxy, weightxy, weightyz, weightyz, weightyz, weightxz, weightxz, weightxz]
        
        features = []
        for grid, weight in zip(grids, weights):
            sampled = F.grid_sample(
                weight.unsqueeze(0), 
                grid.unsqueeze(0).unsqueeze(0), 
                align_corners=True,
            ) # (1, C, 1, N)
            
            # x_coords = grid[:, 0]
            # y_coords = grid[:, 1]
            # coord_sum = x_coords + y_coords
            # sampled = torch.zeros(1, C, 1, N, device='cuda', dtype=torch.float)
            # for c in range(C):
            #     sampled[0, c, 0, :] = coord_sum
            
            sampled = sampled.squeeze(0).squeeze(-2).permute(1, 0)
            features.append(sampled)

        result = torch.cat([
                features[0],
                features[3],
                features[6],
                features[1],
                features[4],
                features[7],
                features[2],
                features[5],
                features[8],
            ], dim=1)
        
        # Synchronize after timing
        torch.cuda.synchronize()
        end_time = time.time()
        method2_times.append(end_time - start_time)
        torch.cuda.empty_cache()
        
        # Verify correctness occasionally (e.g., every 10 runs)
        if run % 10 == 0:
            is_close = torch.allclose(final_output, result, rtol=0.0003, atol=0.0001)
            if not is_close:
                print(f"Warning: Results don't match on run {run}")
    
    # Calculate statistics
    method1_times = np.array(method1_times) * 1000  # convert to ms
    method2_times = np.array(method2_times) * 1000  # convert to ms
    
    print("\n--- Benchmark Results (time in ms) ---")
    print(f"Yan's tirplane_sampler:")
    print(f"  Mean: {np.mean(method1_times):.4f} ms")
    print(f"  Median: {np.median(method1_times):.4f} ms")
    print(f"  Min: {np.min(method1_times):.4f} ms")
    print(f"  Max: {np.max(method1_times):.4f} ms")
    print(f"  Std Dev: {np.std(method1_times):.4f} ms")
    
    print(f"\nTorch implementation:")
    print(f"  Mean: {np.mean(method2_times):.4f} ms")
    print(f"  Median: {np.median(method2_times):.4f} ms")
    print(f"  Min: {np.min(method2_times):.4f} ms")
    print(f"  Max: {np.max(method2_times):.4f} ms")
    print(f"  Std Dev: {np.std(method2_times):.4f} ms")
    
    speedup = np.mean(method2_times) / np.mean(method1_times)
    print(f"\nYan is {speedup:.2f}x faster than Torch" if speedup > 1 else 
          f"\nTorch is {1/speedup:.2f}x faster than Yan")

    return np.mean(method1_times), np.mean(method2_times)

if __name__ == "__main__":
    # batch_size = 1
    # num_samples = 334415
    # weight_dim = (2433, 3)
    # feature_dim = 4
    # iterations = 1
    
    # avg_time = performance_test(batch_size, num_samples, weight_dim, feature_dim, iterations)
    # print(f"Average time per iteration: {avg_time:.6f} seconds")
    
    yan.jit_kernels.tirplane_accuracy_test()
    # benchmark_test(num_runs=100, warmup_runs=10)