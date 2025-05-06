#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

constexpr int ALIGNMENT = 64;

template <int C, class TiledCopy_g2s, class TiledCopy_s2g>
__global__ void some_reorder_kernel(const float *sample, float *output,
                                    int           N,
                                    TiledCopy_g2s copy_g2s,
                                    TiledCopy_s2g copy_s2g)
{
    int              idx_map[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    __shared__ float smem[ALIGNMENT * 9 * C];

    Tensor mS = make_tensor(make_gmem_ptr(sample),
                            make_layout(
                                make_shape(N, make_tuple(Int<C>{}, _9{})),
                                make_stride(Int<C>{}, make_tuple(_1{}, N * C))));
    // Tensor mS = make_tensor(make_gmem_ptr(sample), make_layout(make_shape(N, 36), make_stride(36, 1)));
    int wid = threadIdx.x / 32;

    Tensor gS_     = local_tile(mS, make_shape(ALIGNMENT, C), make_coord(blockIdx.x, _));
    auto   stride_ = size<1>(gS_.stride());
    Tensor gS      = make_tensor(gS_.data(), make_layout(make_shape(Int<ALIGNMENT>{}, Int<C>{}, _9{}), make_stride(Int<C>{}, _1{}, stride_)));
    Tensor sS      = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<ALIGNMENT>{}, Int<C>{}, _9{}), make_stride(Int<C * 9>{}, _1{}, Int<C>{})));

    int     lane     = threadIdx.x % 32;
    ThrCopy thr_copy = copy_g2s.get_slice(lane);
    Tensor  tSgS     = thr_copy.partition_S(gS(_, _, idx_map[wid]));
    Tensor  tSsS     = thr_copy.partition_D(sS(_, _, wid));
    // if(thread0())
    // {
    //     print("\n\ntSgS: ");
    //     print(tSgS);
    //     print("\n\ntSsS: ");
    //     print(tSsS);
    // }
    copy(copy_g2s, tSgS(_, _, _), tSsS(_, _, _));
    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();

    Tensor  mO        = make_tensor(make_gmem_ptr(output), make_layout(make_shape(N, Int<9 * C>{}), make_stride(Int<9 * C>{}, _1{})));
    Tensor  gO        = local_tile(mO, make_shape(Int<ALIGNMENT>{}, Int<9 * C>{}), make_coord(blockIdx.x, 0));
    Tensor  sO        = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<ALIGNMENT>{}, Int<9 * C>{}), make_stride(Int<9 * C>{}, _1{})));
    ThrCopy thr_copy2 = copy_s2g.get_slice(threadIdx.x);
    Tensor  tOsO      = thr_copy2.partition_S(sO);
    Tensor  tOgO      = thr_copy2.partition_D(gO);

    copy(copy_s2g, tOsO, tOgO);

    // if(thread0())
    // {
    //     print("\n\ngO: ");
    //     print(gO);
    //     print("\n\n");
    // }
}

template <int C, int H, int W>
inline void some_reorder(const float *sample, float *output,
                         int N, cudaStream_t stream = 0)
{
    // sample: (N*9, C)
    // output: (N, C*9)

    int kNumBlocks  = (N + ALIGNMENT - 1) / ALIGNMENT;
    int kNumThreads = 32 * 9;

    TiledCopy copy_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                         Layout<Shape<_32, _1>>{}, // Thr layout
                                         Layout<Shape<_1, _8>>{}); // Val layout)

    TiledCopy copy_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                         Layout<Shape<_32, _9>, Stride<_9, _1>>{}, // Thr layout
                                         Layout<Shape<_1, _4>>{});

    some_reorder_kernel<C><<<kNumBlocks, kNumThreads>>>(sample, output, N,
                                                      copy_g2s, copy_s2g);

    cudaDeviceSynchronize();
}