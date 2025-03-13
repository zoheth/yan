#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <class SmemLayout, class TiledCopyA>
__global__ void scan_phase1(float *input, float *output, const int N, SmemLayout layout, TiledCopyA copy_a)
{
    Tensor mA = make_tensor(make_gmem_ptr(input), make_shape(N));
    auto cta_coord = make_coord(blockIdx.x);
    Tensor gA = local_tile(mA, make_shape(4), cta_coord);

    __shared__ float smem[cosize_v<SmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smem), layout);

    ThrCopy thr_copy = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy.partition_S(gA);
    Tensor tAsA = thr_copy.partition_D(sA);

    copy(copy_a, tAgA, tAsA);

    __syncthreads();

    tAsA.data()[0] = 0;
}

void
cute_scan(float *input, float *output, unsigned int N)
{
    constexpr int block_size = 512;

    auto sA = make_layout(make_shape(block_size*4));
    
    TiledCopy copy_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                         Layout<Shape<Int<block_size>>>{},
                                         Layout<Shape<Int<4>>>{});

    

}