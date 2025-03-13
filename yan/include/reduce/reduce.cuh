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
};

template <class T>
__device__ __forceinline__ T
warp_reduce(T val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T, typename T_OUT, typename F>
__global__ void
block_reduce(
    const uint32_t n_vector_loads,
    const F fun,
    const T *__restrict__ input,
    T_OUT *__restrict__ output,
    const uint32_t blocks_per_reduction)
{
    const uint32_t reduction_idx = blockIdx.x / blocks_per_reduction;
    const uint32_t sub_blocks_idx = blockIdx.x % blocks_per_reduction;

    const uint32_t i = threadIdx.x + sub_blocks_idx * blockDim.x;
    const uint32_t block_offset = reduction_idx * n_vector_loads;

    static __shared__ T_OUT sdata[32];

    int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

    using T_DECAYED = std::decay_t<T>;

    T_OUT val;
    if (std::is_same<T_DECAYED, float>::value) 
    {
        if(i < n_vector_loads)
        {
            float4 vals = reinterpret_cast<const float4 *>(input)[i + block_offset];
            val = fun((T)vals.x) + fun((T)vals.y) + fun((T)vals.z) + fun((T)vals.w);
        }
        else
        {
            val = 0;
        }
    }

    val = warp_reduce(val);

    if (lane == 0)
    {
        sdata[wid] = val;
    }

    __syncthreads();

    if (wid == 0)
    {
        val = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : 0;
        val = warp_reduce(val);

        if(lane == 0)
        {
            atomicAdd(&output[reduction_idx], val);
        }
    }
}

template <typename T, int BLOCK_SIZE = 1024>
void
cute_reduce_sum(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t threads = BLOCK_SIZE;

    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

    assert(n_elements % N_ELEMS_PER_LOAD == 0);

    n_elements /= N_ELEMS_PER_LOAD;

    const uint32_t blocks = (n_elements + threads - 1) / threads;
    block_reduce<<<blocks, threads, 0, stream>>>(n_elements, [] __device__ (float val) { return val; }, d_input, d_output, blocks);
}