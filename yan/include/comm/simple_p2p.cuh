#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "utils.cuh"

__global__ void p2p_access_kernel(const float* pSrc, float* pDst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        pDst[idx] = pSrc[idx] * 2.0f;
    }
}

void simple_p2p(float* d_input, float* d_output, size_t n)
{
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
    {
        std::cerr << "Error: Need at least 2 GPUs to run this test." << std::endl;
        return;
    }

    int dev0 = 0;
    int dev1 = 1;

    checkCuda(cudaSetDevice(dev0));
    int can_access_peer_0_1, can_access_peer_1_0;
    checkCuda(cudaDeviceCanAccessPeer(&can_access_peer_0_1, dev0, dev1));
    checkCuda(cudaDeviceCanAccessPeer(&can_access_peer_1_0, dev1, dev0));

    cudaSetDevice(dev0);
    // cudaDeviceEnablePeerAccess(dev1, 0);

    cudaSetDevice(dev1);
    cudaDeviceEnablePeerAccess(dev0, 0);

    const size_t dataSize = n * sizeof(float);

    checkCuda(cudaSetDevice(dev1));
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    p2p_access_kernel<<<grid, block, 0, stream>>>(d_input, d_output, n);
}