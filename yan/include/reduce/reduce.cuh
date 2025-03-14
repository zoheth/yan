#include <cute/tensor.hpp>

using namespace cute;

template <class T>
struct SumOp
{
    __device__ __forceinline__ T
    operator()(const T &a, const T &b) const
    {
        return a + b;
    }

    __device__ __forceinline__ T
    identity() const
    {
        return T(0);
    }

    __device__ __forceinline__ void
    atomic_op(T *address, T val) const
    {
        atomicAdd(address, val);
    }
};

template <class T>
struct MaxOp;

template <class T, class ReduceOp>
__device__ __forceinline__ T
warp_reduce(T val, ReduceOp op)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T, typename T_OUT, class ReduceOp, int ITEMS_PER_THREAD = 1>
__global__ void
block_reduce(
    const uint32_t n_vector_loads,
    const ReduceOp reduce_op,
    const T *__restrict__ input,
    T_OUT *__restrict__ output,
    const uint32_t blocks_per_reduction)
{
    const uint32_t reduction_idx = blockIdx.x / blocks_per_reduction;
    const uint32_t sub_blocks_idx = blockIdx.x % blocks_per_reduction;

    const uint32_t base_idx = threadIdx.x + sub_blocks_idx * blockDim.x * ITEMS_PER_THREAD;
    const uint32_t block_offset = reduction_idx * n_vector_loads;

    // 最大线程数为 1024 index对应warp id 非lane id
    static __shared__ T_OUT sdata[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    using T_DECAYED = std::decay_t<T>;

    T_OUT val = reduce_op.identity();
    ;
    if constexpr (std::is_same_v<T_DECAYED, float>)
    {
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const uint32_t i = base_idx + item * blockDim.x;
            if (i < n_vector_loads)
            {
                float4 vals = reinterpret_cast<const float4 *>(input)[i + block_offset];
                T_OUT local_val = reduce_op(vals.x, vals.y);
                local_val = reduce_op(local_val, vals.z);
                local_val = reduce_op(local_val, vals.w);
                val = reduce_op(val, local_val);
            }
        }
    } else if constexpr (std::is_same_v<T_DECAYED, __half> || std::is_same_v<T_DECAYED, ::half>)
    {
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const uint32_t i = base_idx + item * blockDim.x;
            if (i < n_vector_loads)
            {
                ::half vals[8];
                *(int4 *)&vals[0] = *((int4 *)input + i + block_offset);

                T_OUT local_val = vals[0];

#pragma unroll
                for (int j = 1; j < 8; j++)
                {
                    local_val = reduce_op(local_val, vals[j]);
                }
                val = reduce_op(val, local_val);
            }
        }
    } else
    {
        assert(false);
    }

    val = warp_reduce(val, reduce_op);

    if (lane == 0)
    {
        sdata[wid] = val;
    }

    __syncthreads();

    if (wid == 0)
    {
        val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : reduce_op.identity();
        val = warp_reduce(val, reduce_op);

        if (lane == 0)
        {
            reduce_op.atomic_op(&output[reduction_idx], val);
        }
    }
}

template <typename T, const int BLOCK_SIZE = 1024, const int ITEMS_PER_THREAD = 16>
void
reduce_sum_c(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t threads = BLOCK_SIZE;

    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

    assert(n_elements % N_ELEMS_PER_LOAD == 0);

    n_elements /= N_ELEMS_PER_LOAD;

    uint32_t blocks = (n_elements + threads - 1) / threads;
    blocks = (blocks + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
    block_reduce<T, T, SumOp<T>, ITEMS_PER_THREAD><<<blocks, threads, 0, stream>>>(n_elements, SumOp<T>(), d_input, d_output, blocks);
}

template <typename T, const int BLOCK_SIZE = 1024, const int ITEMS_PER_THREAD = 16>
void
reduce_max_c(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t threads = BLOCK_SIZE;

    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

    assert(n_elements % N_ELEMS_PER_LOAD == 0);

    n_elements /= N_ELEMS_PER_LOAD;

    uint32_t blocks = (n_elements + threads - 1) / threads;
    blocks = (blocks + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
    block_reduce<T, T, MaxOp<T>, ITEMS_PER_THREAD><<<blocks, threads, 0, stream>>>(n_elements, MaxOp<T>(), d_input, d_output, blocks);
}

template <>
struct MaxOp<float>
{
    __device__ __forceinline__ float
    operator()(const float &a, const float &b) const
    {
        return fmaxf(a, b);
    }

    __device__ __forceinline__ float
    identity() const
    {
        return -1e20f;
    }

    __device__ __forceinline__ void
    atomic_op(float *address, float val) const
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
};

template <>
struct MaxOp<half>
{
    __device__ __forceinline__ half
    operator()(const half &a, const half &b) const
    {
        return __hgt(a, b) ? a : b;
    }

    __device__ __forceinline__ half
    identity() const
    {
        return __float2half(-1e4f);
    }

    __device__ __forceinline__ void
    atomic_op(half *address, half val) const
    {
        unsigned short *address_as_ushort = (unsigned short *)address;
        unsigned short old = *address_as_ushort;
        unsigned short assumed;

        do
        {
            assumed = old;
            half assumed_half = __ushort_as_half(assumed);
            half max_val = __hgt(val, assumed_half) ? val : assumed_half;
            old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(max_val));
        } while (assumed != old);
    }
};