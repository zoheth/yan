#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

__constant__ float global_input[16 * 1024];

template <int C, int H, int W, class TiledCopy_g2s, class TiledCopy_g2r, class TiledCopy_r2g>
__global__ void some_sampler_kernel(const float *input, const float *grid, float *output,
                                    int N, int output_offset,
                                    TiledCopy_g2s copy_g2s,
                                    TiledCopy_g2r copyG_g2r,
                                    TiledCopy_r2g copyO_r2g)
{
    extern __shared__ float input_shared[];

    Tensor gI = make_tensor(make_gmem_ptr(input), make_layout(make_shape(C, H, W)));
    Tensor sI = make_tensor(make_smem_ptr(input_shared), make_layout(make_shape(C, H, W)));

    ThrCopy thr_copy = copy_g2s.get_slice(threadIdx.x);
    Tensor  tIgI     = thr_copy.partition_S(gI);
    Tensor  tIsI     = thr_copy.partition_D(sI);

    // copy(copy_g2s, tIgI, tIsI);
    // cp_async_fence();

    Tensor gG = make_tensor(make_gmem_ptr(grid), make_layout(
                                                     make_shape(
                                                         make_tuple(N, 3), 2),
                                                     make_stride(
                                                         make_tuple(2, 2 * N), 1)));

    Tensor gO = make_tensor(make_gmem_ptr(output), make_layout(
                                                       make_shape(
                                                           make_tuple(N, 3),
                                                           make_tuple(C, 3)),
                                                       make_stride(
                                                           make_tuple(C * 9, C * 3),
                                                           make_tuple(1, C))));

    // if (thread0())
    // {
    //     print("\n\ngO: ");
    //     print(gO);
    //     print("\n\n");
    // }

    int idx_start = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    int stride    = blockDim.x * gridDim.x * 2;

    for (int i = idx_start; i < N * 3; i += stride)
    {
        float4 G2 = reinterpret_cast<const float4 *>(gG.data().get())[i / 2];

        float4 val0;
        val0.x = G2.x + G2.y;
        val0.y = G2.x + G2.y;
        val0.z = G2.x + G2.y;
        val0.w = G2.x + G2.y;

        float *addr0                         = reinterpret_cast<float *>(gO.data().get()) + gO.layout()(i, make_tuple(0, output_offset));
        reinterpret_cast<float4 *>(addr0)[0] = val0;

        if (i + 1 < N * 3)
        {
            float4 val1;
            val1.x                               = G2.z + G2.w;
            val1.y                               = G2.z + G2.w;
            val1.z                               = G2.z + G2.w;
            val1.w                               = G2.z + G2.w;
            float *addr1                         = reinterpret_cast<float *>(gO.data().get()) + gO.layout()(i + 1, make_tuple(0, output_offset));
            reinterpret_cast<float4 *>(addr1)[0] = val1;
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
void tirplane_sampler(const float *input, const float *grid, float *output,
                      int N, int output_offset, cudaStream_t stream = 0)
{
    // input: (C, H, W)
    // grid: (N*3, 2)
    // output: (N, C*9)
    constexpr int kNumThreads = 256;
    constexpr int kNumBlocks  = 128;

    TiledCopy copy_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                         Layout<_1, _256>{}, // Thr layout
                                         Layout<_4, _1>{});  // Val layout)

    TiledCopy copyG_g2r = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                          Layout<_256, _1>{}, // Thr layout
                                          Layout<_4, _2>{});  // Val layout
    TiledCopy copyO_r2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                          Layout<_256, _1>{}, // Thr layout
                                          Layout<_4, _4>{});

    cudaMemcpyToSymbol(global_input, input, 64*1024);

    cudaFuncSetAttribute(some_sampler_kernel<C, H, W,
                                             decltype(copy_g2s), decltype(copyG_g2r), decltype(copyO_r2g)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(float) * C * H * W);

    some_sampler_kernel<C, H, W><<<kNumBlocks, kNumThreads, sizeof(float) * C * H * W, stream>>>(input, grid, output, N, output_offset,
                                                                                                 copy_g2s, copyG_g2r, copyO_r2g);
    cudaDeviceSynchronize();
}