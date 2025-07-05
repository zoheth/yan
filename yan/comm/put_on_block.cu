#include "nvshmem.h"
#include "nvshmemx.h"
#include <cassert>
#include <iostream>
#include <vector>

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result)                                                \
        {                                                                         \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 1024

__global__ void set_and_shift_kernel(float *send_data, float *recv_data, int n_elems, int mype, int n_pes)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < n_elems)
    {
        send_data[thread_idx] = mype;
    }

    int peer = (mype + 1) % n_pes;

    int block_offset = blockIdx.x * blockDim.x;
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset, min(blockDim.x, num_elems - block_offset),
                             peer);
}

int main(int argc, char *argv[])
{
    int mype, n_pes, mype_node;
    float *send_data, *recv_data;
    int num_elems = 8192;
    int num_blocks;
    cudaStream_t stream;

    nvshmem_init();

    mype = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    assert(send_data != NULL && recv_data != NULL);

    assert(num_elems % THREADS_PER_BLOCK == 0);
    num_blocks = num_elems / THREADS_PER_BLOCK;

    set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems, mype, n_pes);
    nvshmemx_barrier_all_on_stream(stream);

    /* Do data validation */
    std::vector<float> host(num_elems);
    CUDA_CHECK(cudaMemcpy(host.data(), recv_data, num_elems * sizeof(float), cudaMemcpyDefault));
    int ref = (mype - 1 + n_pes) % n_pes;
    bool success = true;

    for (int i = 0; i < num_elems; ++i)
    {
        if (host[i] != ref)
        {
            std::cerr << "Error at index " << i << ": expected " << ref << ", got " << host[i] << std::endl;
            success = false;
        }
    }
    if(success)
    {
        std::cout << "All values are correct!" << std::endl;
    }
    else
    {
        std::cerr << "Some values are incorrect!" << std::endl;
    }

    nvshmem_free(send_data);
    nvshmem_free(recv_data);

    nvshmem_finalize();

    return 0;
}