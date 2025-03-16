#pragma once

#include <cute/tensor.hpp>

struct __align__(8) MD
{
    float m{-__FLT_MAX__};
    float d{0.0f};
};

__device__ __forceinline__ MD
warp_reduce_md_op(MD val)
{
#pragma unroll
    for (int stride = warpSize / 2; stride > 0; stride /= 2)
    {
        MD temp;
        temp.m = __shfl_xor_sync(0xffffffff, val.m, stride);
        temp.d = __shfl_xor_sync(0xffffffff, val.d, stride);
        bool temp_bigger = temp.m > val.m;
        MD bigger = temp_bigger ? temp : val;
        MD smaller = temp_bigger ? val : temp;
        val.d = bigger.d + smaller.d * expf(smaller.m - bigger.m);
        val.m = bigger.m;
    }
    return val;
}

__global__ void
online_softmax_kernel(const float *__restrict input, float *__restrict output, const uint32_t n_vector_loads)
{
    MD md;
    for(int vector_id = threadIdx.x; vector_id < n_vector_loads; vector_id += blockDim.x)
    {
        float4 val = reinterpret_cast<const float4 *>(input)[vector_id + blockIdx.x * n_vector_loads];
        float old_m = md.m;
        md.m = fmaxf(md.m, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        if (old_m != md.m) {
            md.d = md.d * expf(old_m - md.m);
        }
        md.d += expf(val.x - md.m) + expf(val.y - md.m) + expf(val.z - md.m) + expf(val.w - md.m);
    }

    md = warp_reduce_md_op(md);

    // 最大线程数为 1024 index对应warp id 非lane id
    static __shared__ MD sdata[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0)
    {
        sdata[wid] = md;
    }
    __syncthreads();

    if (wid == 0)
    {
        md = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : MD{0.0f, 0.0f};
        md = warp_reduce_md_op(md);
        if (lane == 0)
        {
            sdata[0] = md;
        }
    }
    __syncthreads();

    md = sdata[0];
    float d_inversed = 1.0f / md.d;

    for(int vector_id = threadIdx.x; vector_id < n_vector_loads; vector_id += blockDim.x)
    {
        float4 val = reinterpret_cast<const float4 *>(input)[vector_id + blockIdx.x * n_vector_loads];
        val.x = expf(val.x - md.m) * d_inversed;
        val.y = expf(val.y - md.m) * d_inversed;
        val.z = expf(val.z - md.m) * d_inversed;
        val.w = expf(val.w - md.m) * d_inversed;
        reinterpret_cast<float4 *>(output)[vector_id + blockIdx.x * n_vector_loads] = val;
    }
}

void
online_softmax_c(float *input, float *output, uint32_t batch_size, uint32_t n_elements)
{
    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(float);
    assert(n_elements % N_ELEMS_PER_LOAD == 0);

    const uint32_t n_vector_loads = n_elements / N_ELEMS_PER_LOAD;

    online_softmax_kernel<<<batch_size, 1024>>>(input, output, n_vector_loads);
}