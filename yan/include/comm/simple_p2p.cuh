#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

inline void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorName(err));
        throw std::runtime_error(cudaGetErrorName(err));
    }
}

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
    int canAccessPeer = 0;
    checkCuda(cudaDeviceCanAccessPeer(&canAccessPeer, dev0, dev1));
    if (canAccessPeer)
    {
        std::cout << "GPU " << dev0 << " can access GPU " << dev1 << "'s memory." << std::endl;
        checkCuda(cudaDeviceEnablePeerAccess(dev1, 0)); // 标志位为0，是保留的
        std::cout << "Enabled P2P access from GPU " << dev0 << " to GPU " << dev1 << std::endl;
    } else
    {
        std::cerr << "Error: P2P access between GPU " << dev0 << " and " << dev1 << " is not supported." << std::endl;
        return;
    }

    const size_t dataSize = n * sizeof(float);



    checkCuda(cudaSetDevice(dev1));
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    p2p_access_kernel<<<grid, block, 0, stream>>>(d_input, d_output, n);
}