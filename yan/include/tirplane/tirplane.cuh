#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "reorder.cuh"

using namespace cute;

__constant__ half global_input[32 * 1024];

__device__ __forceinline__ float4 scalar_multiply(float4 vec, float scalar) {
    float4 result;
    result.x = vec.x * scalar;
    result.y = vec.y * scalar;
    result.z = vec.z * scalar;
    result.w = vec.w * scalar;
    return result;
}

__device__ __forceinline__ float4 vector_add(float4 a, float4 b) {
    float4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
}

template <int C, int H, int W>
__device__ __forceinline__ float4 bilinear_sample(float x, float y, const float *input)
{
    float ix = ((x + 1.0f) / 2.0f) * (W - 1);
    float iy = ((y + 1.0f) / 2.0f) * (H - 1);

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    bool nw_valid = (ix_nw >= 0 && ix_nw < W && iy_nw >= 0 && iy_nw < H);
    bool ne_valid = (ix_ne >= 0 && ix_ne < W && iy_ne >= 0 && iy_ne < H);
    bool sw_valid = (ix_sw >= 0 && ix_sw < W && iy_sw >= 0 && iy_sw < H);
    bool se_valid = (ix_se >= 0 && ix_se < W && iy_se >= 0 && iy_se < H);
    
    float4 val = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize val to zeros

    if(nw_valid)
    {
        int nw_idx = iy_nw * W + ix_nw;
        float4 feature = reinterpret_cast<const float4 *>(input)[nw_idx]; 
        val.x += feature.x * nw;
        val.y += feature.y * nw;
        val.z += feature.z * nw;
        val.w += feature.w * nw;
    }
    
    if(ne_valid)
    {
        int ne_idx = iy_ne * W + ix_ne;
        float4 feature = reinterpret_cast<const float4 *>(input)[ne_idx];
        val.x += feature.x * ne;
        val.y += feature.y * ne;
        val.z += feature.z * ne;
        val.w += feature.w * ne;
    }
    
    if(sw_valid)
    {
        int sw_idx = iy_sw * W + ix_sw;
        float4 feature = reinterpret_cast<const float4 *>(input)[sw_idx];
        val.x += feature.x * sw;
        val.y += feature.y * sw;
        val.z += feature.z * sw;
        val.w += feature.w * sw;
    }
    
    if(se_valid)
    {
        int se_idx = iy_se * W + ix_se;
        float4 feature = reinterpret_cast<const float4 *>(input)[se_idx];
        val.x += feature.x * se;
        val.y += feature.y * se;
        val.z += feature.z * se;
        val.w += feature.w * se;
    }
    
    return val;

}

template <int C, int H, int W>
__device__ __forceinline__ float4 bilinear_sample(float x, float y)
{
    float ix = ((x + 1.0f) / 2.0f) * (W - 1);
    float iy = ((y + 1.0f) / 2.0f) * (H - 1);

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    bool nw_valid = (ix_nw >= 0 && ix_nw < W && iy_nw >= 0 && iy_nw < H);
    bool ne_valid = (ix_ne >= 0 && ix_ne < W && iy_ne >= 0 && iy_ne < H);
    bool sw_valid = (ix_sw >= 0 && ix_sw < W && iy_sw >= 0 && iy_sw < H);
    bool se_valid = (ix_se >= 0 && ix_se < W && iy_se >= 0 && iy_se < H);
    
    float4 val = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize val to zeros

    if(nw_valid)
    {
        int nw_idx = iy_nw * W + ix_nw;
        val.x += __half2float(global_input[nw_idx * 4]) * nw;
        val.y += __half2float(global_input[nw_idx * 4 + 1]) * nw;
        val.z += __half2float(global_input[nw_idx * 4 + 2]) * nw;
        val.w += __half2float(global_input[nw_idx * 4 + 3]) * nw;
    }
    
    if(ne_valid)
    {
        int ne_idx = iy_ne * W + ix_ne;
        val.x += __half2float(global_input[ne_idx * 4]) * ne;
        val.y += __half2float(global_input[ne_idx * 4 + 1]) * ne;
        val.z += __half2float(global_input[ne_idx * 4 + 2]) * ne;
        val.w += __half2float(global_input[ne_idx * 4 + 3]) * ne;
    }
    
    if(sw_valid)
    {
        int sw_idx = iy_sw * W + ix_sw;
        val.x += __half2float(global_input[sw_idx * 4]) * sw;
        val.y += __half2float(global_input[sw_idx * 4 + 1]) * sw;
        val.z += __half2float(global_input[sw_idx * 4 + 2]) * sw;
        val.w += __half2float(global_input[sw_idx * 4 + 3]) * sw;
        
    }
    
    if(se_valid)
    {
        int se_idx = iy_se * W + ix_se;
        val.x += __half2float(global_input[se_idx * 4]) * se;
        val.y += __half2float(global_input[se_idx * 4 + 1]) * se;
        val.z += __half2float(global_input[se_idx * 4 + 2]) * se;
        val.w += __half2float(global_input[se_idx * 4 + 3]) * se;
    }
    
    return val;

}

template <int C, int H, int W>
__global__ void some_sampler_kernel(const float *input, const float *grid, float *output,
                                    int N)
{
    extern __shared__ float input_shared[];

    int idx_start = blockIdx.x * blockDim.x * 2 + threadIdx.x*2;
    int stride    = blockDim.x * gridDim.x * 2;

    for (int i = idx_start; i < N * 3; i += stride)
    {
        float4 G2 = reinterpret_cast<const float4 *>(grid)[i / 2];

        reinterpret_cast<float4 *>(output)[i] = bilinear_sample<4, H, W>(G2.x, G2.y, input); 

        if (i + 1 < N * 3)
        {
            reinterpret_cast<float4 *>(output)[i + 1] = bilinear_sample<4, H, W>(G2.z, G2.w, input); 
        }
    }

    // if(thread0())
    // {
    //     print("\n\n");
    //     print(gO);
    //     print("\n\n");
    // }
}

template <int C, int H, int W>
__global__ void some_sampler_kernel(const float *grid, float *output,
                                    int N)
{
    extern __shared__ float input_shared[];

    int idx_start = blockIdx.x * blockDim.x * 2 + threadIdx.x*2;
    int stride    = blockDim.x * gridDim.x * 2;

    for (int i = idx_start; i < N * 3; i += stride)
    {
        float4 G2 = reinterpret_cast<const float4 *>(grid)[i / 2];

        reinterpret_cast<float4 *>(output)[i] = bilinear_sample<4, H, W>(G2.x, G2.y); 

        if (i + 1 < N * 3)
        {
            reinterpret_cast<float4 *>(output)[i + 1] = bilinear_sample<4, H, W>(G2.z, G2.w); 
        }
    }
}

template <int C, int H, int W>
void tirplane_sampler(const float *input0, const float *input1, const float *input2,
                      float *grid,
                      float *sample_o, float *final_o,
                      int N, cudaStream_t stream = 0)
{
    // input: (H, W, C)
    // grid: (N*9, 2)
    // sample: (N*9, C)
    // output: (N, C*9)
    constexpr int kNumThreads = 256;
    constexpr int kNumBlocks  = 256;

    assert(N % 64 == 0);

    float *cur_grid = grid;
    float *output   = sample_o;
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, 0, stream>>>(input0, grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, 0, stream>>>(input1, cur_grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, 0, stream>>>(input2, cur_grid, output, N);

    cudaDeviceSynchronize();

    some_reorder<C, H, W>(sample_o, final_o, N, stream);
}

template <int C, int H, int W>
void tirplane_sampler(const half *input0, const half *input1, const half *input2,
                      float *grid,
                      float *sample_o, float *final_o,
                      int N, cudaStream_t stream = 0)
{
    // input: (H, W, C)
    // grid: (N*9, 2)
    // sample: (N*9, C)
    // output: (N, C*9)
    constexpr int kNumThreads = 256;
    constexpr int kNumBlocks  = 256;

    constexpr int smem_size = 0.f;

    float *cur_grid = grid;
    float *output   = sample_o;
    assert(H * W * C <= 32 * 1024);
    // cudaMemcpyToSymbol(global_input, input0, H * W * C * sizeof(half));
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    // cudaMemcpyToSymbol(global_input, input1, H * W * C * sizeof(half));
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(cur_grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    // cudaMemcpyToSymbol(global_input, input2, H * W * C * sizeof(half));
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(cur_grid, output, N);

    cudaDeviceSynchronize();

    some_reorder<C, H, W>(sample_o, final_o, N, stream);
}