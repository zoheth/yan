import torch
import torch.nn.functional as F
import time

import yan

def performance_test(batch_size=1, num_samples=334415, weight_dim=(2433, 3), feature_dim=4, iterations=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    weightxy = torch.randn(batch_size, feature_dim, weight_dim[0], weight_dim[1], device=device)
    weightyz = torch.randn(batch_size, feature_dim, weight_dim[0], weight_dim[1], device=device)
    weightxz = torch.randn(batch_size, feature_dim, weight_dim[0], weight_dim[1], device=device)
    
    grid_in = torch.randn(batch_size, 1, num_samples, 2, device=device)
    grid_mid = torch.randn(batch_size, 1, num_samples, 2, device=device)
    grid_out = torch.randn(batch_size, 1, num_samples, 2, device=device)

    grids = [grid_in, grid_mid, grid_out] * 3
    weights = [weightxy, weightxy, weightxy, weightyz, weightyz, weightyz, weightxz, weightxz, weightxz]
    
    total_time = 0
    
    for _ in range(iterations):
        features = []
        
        start_time = time.time()
        
        for grid, weight in zip(grids, weights):
            sampled = F.grid_sample(
                weight, grid, align_corners=True,
            )
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
        
        print(result.shape)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / iterations
    return avg_time

if __name__ == "__main__":
    # batch_size = 1
    # num_samples = 334415
    # weight_dim = (2433, 3)
    # feature_dim = 4
    # iterations = 1
    
    # avg_time = performance_test(batch_size, num_samples, weight_dim, feature_dim, iterations)
    # print(f"Average time per iteration: {avg_time:.6f} seconds")
    
    yan.jit_kernels.tirplane_accuracy_test()