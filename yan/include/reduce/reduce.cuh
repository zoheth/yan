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

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, class T, class ReduceOp>
__global__ void
cute_reduce_kernel(T *input, T *output, int N, ReduceOp op)
{
    auto input_tensor = make_tensor(input, make_layout(N));

    __shared__ T smem[BLOCK_SIZE];
    auto smem_tensor = make_tensor(make_smem_ptr(smem), make_layout(BLOCK_SIZE));

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int items_per_block = BLOCK_SIZE * ITEMS_PER_THREAD;
    int block_offset = bid * items_per_block;
    int block_end = min(block_offset + items_per_block, N);

    T thread_sum = op.identity();

    for (int i = block_offset + tid; i < block_end; i += BLOCK_SIZE)
    {
        thread_sum = op(thread_sum, input_tensor(i));
    }

    smem_tensor(tid) = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            smem_tensor(tid) = op(smem_tensor(tid), smem_tensor(tid + stride));
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        if (BLOCK_SIZE >= 64)
            smem_tensor(tid) = op(smem_tensor(tid), smem_tensor(tid + 32));

        T val = smem_tensor(tid);
        val = warp_reduce(val, op);

        if (tid == 0)
        {
            output[bid] = val;
        }
    }
}

template <int BLOCK_SIZE, class T, class ReduceOp>
__global__ void
cute_final_reduce_kernel(
    T *input,
    T *output,
    int N,
    ReduceOp op)
{
    auto input_tensor = make_tensor(input, make_layout(N));

    __shared__ T smem[BLOCK_SIZE];
    auto smem_tensor = make_tensor(make_smem_ptr(smem), make_layout(BLOCK_SIZE));

    int tid = threadIdx.x;

    T thread_sum = op.identity();
    for (int i = tid; i < N; i += BLOCK_SIZE)
    {
        thread_sum = op(thread_sum, input_tensor(i));
    }

    smem_tensor(tid) = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            smem_tensor(tid) = op(smem_tensor(tid), smem_tensor(tid + stride));
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        if (BLOCK_SIZE >= 64)
            smem_tensor(tid) = op(smem_tensor(tid), smem_tensor(tid + 32));

        T val = smem_tensor(tid);
        val = warp_reduce(val, op);

        if (tid == 0)
        {
            output[0] = val;
        }
    }
}

template <typename T, int BLOCK_SIZE = 256, int ITEMS_PER_THREAD = 4>
void
cute_reduce_sum(T *d_input, T *d_output, int size, cudaStream_t stream = 0)
{
    SumOp<T> sum_op;

    int items_per_block = BLOCK_SIZE * ITEMS_PER_THREAD;
    int num_blocks = (size + items_per_block - 1) / items_per_block;

    T *d_temp = nullptr;
    if (num_blocks > 1)
    {
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
    } else
    {
        d_temp = d_output;
    }

    cute_reduce_kernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_temp, size, sum_op);

    if (num_blocks > 1)
    {
        cute_final_reduce_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
            d_temp, d_output, num_blocks, sum_op);

        cudaFree(d_temp);
    }
}