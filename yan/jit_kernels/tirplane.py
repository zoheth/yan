import torch
import torch.nn.functional as F
from .tuner import jit_tuner

includes = ('"tirplane/tirplane.cuh"', )
template = """
// Templated args from Python JIT call
constexpr auto C = {C};
constexpr auto H = {H};
constexpr auto W = {W};

tirplane_sampler<C, H, W>(input0, input1, input2, grid, sample_o, final_o, N, stream);
"""

def tirplane_sampler(input0: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor, 
                     grid: torch.Tensor, 
                     sample_o: torch.Tensor, final_o: torch.Tensor) -> None:
    H, W, C = input0.shape
    N = final_o.shape[0]

    stream = torch.cuda.current_stream()

    global includes, template
    
    args = (input0, input1, input2, grid, sample_o, final_o, N, stream)
    runtime = jit_tuner.compile_and_tune(
        name='tirplane_sampler',
        keys={'C': C, 'H': H, 'W': W},
        space=(),
        includes=includes,
        arg_defs=(
            ('input0', torch.float),
            ('input1', torch.float),
            ('input2', torch.float),
            ('grid', torch.float),
            ('sample_o', torch.float),
            ('final_o', torch.float),
            ('N', int),
            ('stream', torch.cuda.Stream)
        ),
        template=template,
        args=args
    )
    
    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        C = 4
        H = 2443
        W = 3
        N = 1003294
        
        N_padded = ((N + 31) // 32) * 32 
        
        weightxy = torch.randn(H, W, C, device='cuda')
        weightyz = torch.randn(H, W, C, device='cuda')
        weightxz = torch.randn(H, W, C, device='cuda')
        
        grid_in = torch.randn(N_padded, 2, device='cuda', dtype=torch.float)
        grid_mid = torch.randn(N_padded, 2, device='cuda', dtype=torch.float)
        grid_out = torch.randn(N_padded, 2, device='cuda', dtype=torch.float)


        
        sample_output = torch.zeros(9, N_padded, C, device='cuda', dtype=torch.float)
        final_output = torch.zeros(N_padded, C*9, device='cuda', dtype=torch.float)
        grid_cute = torch.cat([grid_in, grid_mid, grid_out] * 3, dim=0)
        assert weightxy.is_contiguous()
        assert weightyz.is_contiguous()
        assert weightxz.is_contiguous()
        assert grid_cute.is_contiguous()
        tirplane_sampler(weightxy,weightyz, weightxz, grid_cute, sample_output, final_output)
        
        # tensor_chunks = [chunk.squeeze(0) for chunk in torch.chunk(sample_output, 9, dim=0)]
        # final_output_ = torch.cat([
        #     tensor_chunks[0],
        #     tensor_chunks[3],
        #     tensor_chunks[6],
        #     tensor_chunks[1],
        #     tensor_chunks[4],
        #     tensor_chunks[7],
        #     tensor_chunks[2],
        #     tensor_chunks[5],
        #     tensor_chunks[8],
        # ], dim=1)
        
        #print(sample_output)
        print(final_output[:N])
        
        weightxy = weightxy.permute(2, 0, 1).contiguous()
        weightyz = weightyz.permute(2, 0, 1).contiguous()
        weightxz = weightxz.permute(2, 0, 1).contiguous()
        
        grids = [grid_in[:N], grid_mid[:N], grid_out[:N]] * 3
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
        
        print(result)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        is_close = torch.allclose(final_output[:N], result, rtol=0.0003, atol=0.0001)
        print("Test passed!" if is_close else "Test failed!")
            

if __name__ == "__main__":
    accuracy_test()
