#include "utils.cuh"
#include <__clang_cuda_builtin_vars.h>

__device__ __forceinline__ float mock_compute(int n , int a)
{
    float temp = 0;
    for (int i = 0; i < n; ++i)
    {
        temp = sinf(float(i + a));
    }
    return temp;
}

__global__ void kernel0(float *data, int size, int val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = mock_compute(1000, idx);
    }
}

__global__ void kernel1(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = mock_compute(1000, idx) + input[idx];
    }
}

__global__ void kernel2(float *input1, float *input2, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = mock_compute(1000, idx) + input1[idx] + input2[idx];
    }
}

int simple_events(float *d_output, int n)
{
    const size_t dataSize = n * sizeof(float);

    cudaStream_t stream1, stream2;
    cudaEvent_t  event_k0_done, event_k10_done, event_k11_done;

    checkCuda(cudaStreamCreate(&stream1));
    checkCuda(cudaStreamCreate(&stream2));
    checkCuda(cudaEventCreate(&event_k0_done));
    checkCuda(cudaEventCreate(&event_k10_done));
    checkCuda(cudaEventCreate(&event_k11_done));

    float *d_temp0, *d_temp10, *d_temp11;
    checkCuda(cudaMalloc(&d_temp0, dataSize));
    checkCuda(cudaMalloc(&d_temp10, dataSize));
    checkCuda(cudaMalloc(&d_temp11, dataSize));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    kernel0<<<grid, block, 0, stream1>>>(d_temp0, n, 10);
    checkCuda(cudaEventRecord(event_k0_done, stream1));
    kernel1<<<grid, block, 0, stream1>>>(d_temp0, d_temp10, n);
    
    checkCuda(cudaStreamWaitEvent)
    checkCuda(cudaEventRecord(event_s2_done, stream2));
}