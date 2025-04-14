#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "reorder.cuh"

using namespace cute;

__constant__ float global_input[16 * 1024];

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

        float4 val0;
        val0.x = G2.x + G2.y;
        val0.y = G2.x + G2.y;
        val0.z = G2.x + G2.y;
        val0.w = G2.x + G2.y;

        reinterpret_cast<float4 *>(output)[i] = val0;

        if (i + 1 < N * 3)
        {
            float4 val1;
            val1.x = G2.z + G2.w;
            val1.y = G2.z + G2.w;
            val1.z = G2.z + G2.w;
            val1.w = G2.z + G2.w;

            reinterpret_cast<float4 *>(output)[i + 1] = val1;
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
void tirplane_sampler(const float *input0, const float *input1, const float *input2,
                      float *grid,
                      float *sample_o, float *final_o,
                      int N, cudaStream_t stream = 0)
{
    // input: (C, H, W)
    // grid: (N*9, 2)
    // sample: (N*9, C)
    // output: (N, C*9)
    constexpr int kNumThreads = 256;
    constexpr int kNumBlocks  = 256;

    // cudaMemcpyToSymbol(global_input, input, 64 * 1024);

    constexpr int smem_size = sizeof(float) * C * H * W;

    cudaFuncSetAttribute(some_sampler_kernel<C, H, W>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    float *cur_grid = grid;
    float *output   = sample_o;
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(input0, grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(input1, cur_grid, output, N);

    cur_grid += (N * 3 * 2);
    output += (N * 3 * C);
    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, smem_size, stream>>>(input2, cur_grid, output, N);

    cudaDeviceSynchronize();

    TiledCopy copy_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                         Layout<Shape<_32, _1>>{}, // Thr layout
                                         Layout<Shape<_1, _4>>{}); // Val layout)

    std::cout<<extent<uint128_t[1]>::value<<std::endl;

    TiledCopy copy_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                         Layout<Shape<_32, _9>, Stride<_9, _1>>{}, // Thr layout
                                         Layout<Shape<_1, _4>>{});

    some_reorder_kernel<C><<<(N + 31) / 32, 32 * 9>>>(sample_o, final_o, N,
                                                      copy_g2s, copy_s2g);

    cudaDeviceSynchronize();
}