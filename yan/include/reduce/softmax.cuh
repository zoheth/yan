#pragma once

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_per_token_kernel(half* x, half* y, int N)
{
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    
}