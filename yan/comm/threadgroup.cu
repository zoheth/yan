#include <cstdlib>
#include <iostream>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <ostream>

#define NTHREADS 128
#define CFACTOR 20

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

#define NVSHMEM_CHECK(stmt)                                                                \
    do                                                                                     \
    {                                                                                      \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result)                                                    \
        {                                                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

__global__ void distributed_vector_sum(int *x, int *y, int *partial_sum, int *sum,
                                       int use_threadgroup, int mype, int n_pes)
{
    int start_index = threadIdx.x;
    // int index          = threadIdx.x;
    int n_elems = blockDim.x * CFACTOR;
    for (int index = start_index; index < n_elems; index += blockDim.x)
    {
        partial_sum[index] = x[index] + y[index];
    }

    nvshmemx_int_fcollect_block(NVSHMEM_TEAM_WORLD, sum, partial_sum, n_elems);
}

int main(int argc, char *argv[])
{
    int  my_pe, n_pes, mype_node;
    int *d_x, *d_y, *partial_sum, *sum;
    int  use_threadgroup = 0;
    int  n_threads       = NTHREADS;
    int  n_elems         = n_threads * CFACTOR;

    nvshmem_init();

    n_pes     = nvshmem_n_pes();
    my_pe     = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaStream_t stream;

    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaMalloc(&d_x, n_elems * sizeof(int));
    cudaMalloc(&d_y, n_elems * sizeof(int));
    cudaMalloc(&partial_sum, n_elems * sizeof(int));

    // partial_sum = (int *)nvshmem_malloc(n_threads * sizeof(int));
    sum = (int *)nvshmem_malloc(n_elems * sizeof(int) * n_pes);

    int *h_x   = (int *)malloc(sizeof(int) * n_elems);
    int *h_y   = (int *)malloc(sizeof(int) * n_elems);
    int *h_sum = (int *)malloc(sizeof(int) * n_elems * n_pes);

    for (int i = 0; i < n_elems; ++i)
    {
        h_x[i] = my_pe * n_elems;
        h_y[i] = i;
    }
    cudaMemcpyAsync(d_x, h_x, n_elems * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y, h_y, n_elems * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    void *args[] = {&d_x, &d_y, &partial_sum, &sum, &use_threadgroup, &my_pe, &n_pes};
    dim3  dimBlock(n_threads);
    dim3  dimGrid(1);
    // cudaFuncSetAttribute(distributed_vector_sum, cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     128*1024);
    NVSHMEM_CHECK(
        nvshmemx_collective_launch((const void *)distributed_vector_sum, dimGrid, dimBlock, args, 0, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpyAsync(h_sum, sum, n_elems * sizeof(int) * n_pes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::cout << "PE " << my_pe << " received sum: ";
    for (int i = 10; i > 0; --i)
    {
        std::cout << h_sum[n_elems * n_pes - i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(partial_sum);
    // nvshmem_free(partial_sum);
    nvshmem_free(sum);
    free(h_x);
    free(h_y);
    free(h_sum);
    nvshmem_finalize();
    return 0;
}