#pragma once

#include <cuda_runtime_api.h>
#include <cute/tensor.hpp>

__device__ __forceinline__ void
atomicMax(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int;
    int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ float
warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float
warp_reduce_max(float val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int NUM_THREADS = 256>
__device__ float
block_reduce_sum(float val)
{
    constexpr int NUM_WARPS = (NUM_THREADS + 32 - 1) / 32;
    static __shared__ float shared[NUM_WARPS];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();
    
    if(wid == 0)
    {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

template <int NUM_THREADS = 256>
__device__ float
block_reduce_max(float val)
{
    constexpr int NUM_WARPS = (NUM_THREADS + 32 - 1) / 32;
    static __shared__ float shared[NUM_WARPS];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_max(val);
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();
    
    if(wid == 0)
    {
        val = (lane < NUM_WARPS) ? shared[lane] : 0;
        val = warp_reduce_max(val);
    }
    return val;
}

template <int NUM_THREADS = 256>
__global__ void
find_max_f32_kernel(float *x, float *global_max_sum, int n_vector_loads)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = -1e20f;
    if (idx < n_vector_loads)
    {
        float4 vals = reinterpret_cast<const float4 *>(x)[idx];
        float val = vals.x;
        val = fmaxf(val, vals.y);
        val = fmaxf(val, vals.z);
        val = fmaxf(val, vals.w);
        max_val = block_reduce_max<NUM_THREADS>(val);
    }
    
    if(tid == 0)
    {
        atomicMax(global_max_sum, max_val);
    }
}

template <int NUM_THREADS = 256>
__global__ void
find_exp_sum_f32_kernel(float *x, float *global_max_sum, int n_vector_loads)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float exp_sum = 0.0f;
    if (idx < n_vector_loads)
    {
        float4 vals = reinterpret_cast<const float4 *>(x)[idx];
        vals.x = expf(vals.x - global_max_sum[0]);
        vals.y = expf(vals.y - global_max_sum[0]);
        vals.z = expf(vals.z - global_max_sum[0]);
        vals.w = expf(vals.w - global_max_sum[0]);

        exp_sum = vals.x + vals.y + vals.z + vals.w;
        exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);
    }

    if(tid == 0)
    {
        atomicAdd(global_max_sum+1, exp_sum);
    }
}

template <const int NUM_THREADS = 256>
__global__ void
softmax_f32_kernel(float *x, float *y, float *global_max_sum, int n_vector_loads)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 vals = {0.0f, 0.0f, 0.0f, 0.0f};

    if (idx < n_vector_loads)
    {
        vals = reinterpret_cast<const float4 *>(x)[idx];
        vals.x = expf(vals.x - global_max_sum[0]) / global_max_sum[1];
        vals.y = expf(vals.y - global_max_sum[0]) / global_max_sum[1];
        vals.z = expf(vals.z - global_max_sum[0]) / global_max_sum[1];
        vals.w = expf(vals.w - global_max_sum[0]) / global_max_sum[1];
        reinterpret_cast<float4 *>(y)[idx] = vals;
        
    }
}


template <typename T, const int BLOCK_SIZE = 1024>
void
global_softmax_c(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t threads = BLOCK_SIZE;

    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

    assert(n_elements % N_ELEMS_PER_LOAD == 0);

    n_elements /= N_ELEMS_PER_LOAD;

    uint32_t blocks = (n_elements + threads - 1) / threads;

    float *global_max_sum;
    cudaMalloc(&global_max_sum, sizeof(float)*2);
    float h_max_sum[2] = {0.0f, 0.0f};
    cudaMemcpy(global_max_sum, h_max_sum, sizeof(float)*2, cudaMemcpyHostToDevice);

    find_max_f32_kernel<BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_input, global_max_sum, n_elements);
    cudaDeviceSynchronize();
    find_exp_sum_f32_kernel<BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_input, global_max_sum, n_elements);
    cudaDeviceSynchronize();
    softmax_f32_kernel<BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_input, d_output, global_max_sum, n_elements);

    cudaFree(global_max_sum);
}