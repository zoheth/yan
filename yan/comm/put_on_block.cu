#include <cassert>
#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
#include <iostream>
#include <thread>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"
#include <nccl.h>

#include "helper.h"

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

#define THREADS_PER_BLOCK 128

int num_elems  = 16384;
int num_blocks = 128;

__global__ void set_kernel(float *data, int n_elems_per_block, int rank)
{
    // Data is guaranteed to be aligned
    int start_idx = blockIdx.x * n_elems_per_block + threadIdx.x;

    for (int i = start_idx; i < start_idx + n_elems_per_block; i += blockDim.x)
    {
        data[i] = rank;
    }
}

__global__ void send_kernel(float *send_data, float *recv_data, int n_elems_per_block, int mype, uint64_t *signal)
{
    signal = signal + blockIdx.x;
    // Data is guaranteed to be aligned
    int peer         = 0;
    int block_offset = blockIdx.x * n_elems_per_block;
    recv_data        = recv_data + (mype - 1) * n_elems_per_block * gridDim.x;
    nvshmemx_float_put_signal_nbi_block(recv_data + n_elems_per_block, send_data + n_elems_per_block, n_elems_per_block,
                                        signal, 1, NVSHMEM_SIGNAL_ADD, peer);
}

__global__ void set_and_send_kernel(float *send_data, float *recv_data, int n_elems_per_block, int mype, uint64_t *signal)
{
    // Data is guaranteed to be aligned
    int start_idx = blockIdx.x * n_elems_per_block + threadIdx.x;

    for (int i = start_idx; i < start_idx + n_elems_per_block; i += blockDim.x)
    {
        send_data[i] = mype;
    }

    signal = signal + blockIdx.x;

    int peer         = 0;
    int block_offset = blockIdx.x * n_elems_per_block;
    recv_data        = recv_data + (mype - 1) * n_elems_per_block * gridDim.x;
    nvshmemx_float_put_signal_nbi_block(recv_data + block_offset, send_data + block_offset, n_elems_per_block,
                                        signal, 1, NVSHMEM_SIGNAL_ADD, peer);
}

__global__ void wait_and_get_kernel(float *output, float *recv_data, int n_elems_per_block, int mype, int n_pes, uint64_t *signal)
{
    signal = signal + blockIdx.x;
    nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, n_pes - 1);

    // Data is guaranteed to be aligned
    int block_offset = blockIdx.x * n_elems_per_block * (n_pes - 1);
    int start_idx    = block_offset + threadIdx.x;

    for (int i = start_idx; i < block_offset + n_elems_per_block * (n_pes - 1); i += blockDim.x)
    {
        output[i] = recv_data[i];
    }
    *signal = 0;
}

__global__ void get_kernel(float *output, float *recv_data, int n_elems_per_block, int mype, int n_pes)
{
    // Data is guaranteed to be aligned
    int block_offset = blockIdx.x * n_elems_per_block * (n_pes - 1);
    int start_idx    = block_offset + threadIdx.x;

    for (int i = start_idx; i < block_offset + n_elems_per_block * (n_pes - 1); i += blockDim.x)
    {
        output[i] = recv_data[i];
    }
}

struct NcclExecutionPolicy
{
    static void run(int rank, int world_size, cudaStream_t stream,
                    float *send_data, float *recv_data, int num_elems_per_block, ncclComm_t comm)
    {
        if (rank != 0)
        {
            set_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, num_elems_per_block, rank);
        }
        NCCL_CHECK(ncclGroupStart());
        if (rank == 0)
        {
            for (int r = 1; r < world_size; ++r)
            {
                NCCL_CHECK(ncclRecv(recv_data, num_elems, ncclFloat, r, comm, stream));
                recv_data += num_elems;
            }
            recv_data -= num_elems * (world_size - 1);
        } else
        {
            NCCL_CHECK(ncclSend(send_data, num_elems, ncclFloat, 0, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());

        if(rank==0)
        {
            get_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems_per_block, rank, world_size);
        }
    }
};

struct NvshmemExecutionPolicy
{
    static void run(int rank, int world_size, cudaStream_t stream,
                    float *send_data, float *recv_data, int num_elems_per_block, ncclComm_t comm /* unused */)
    {
        static uint64_t *signal = (uint64_t *)nvshmem_calloc(num_blocks, sizeof(uint64_t));
        if (rank != 0)
        {
            void  *args[]   = {&send_data, &recv_data, &num_elems_per_block, &rank, &signal};
            // set_and_send_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems_per_block, rank, signal);
            nvshmemx_collective_launch((const void *)set_and_send_kernel, num_blocks, THREADS_PER_BLOCK, args, 0, stream);

        } else
        {
            void  *args[]   = {&send_data, &recv_data, &num_elems_per_block, &rank, &world_size, &signal};
            // wait_and_get_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(recv_data, recv_data, num_elems_per_block, rank, world_size, signal);
            nvshmemx_collective_launch((const void *)wait_and_get_kernel, num_blocks, THREADS_PER_BLOCK, args, 0, stream);
        }
        nvshmemx_barrier_all_on_stream(stream);
    }
};

void check(float *send_data, int rank, int world_size = 4)
{
    if (rank != 0)
    {
        return;
    }
    std::vector<float> host_data(num_elems * 3, 0.0f);
    CUDA_CHECK(cudaMemcpy(host_data.data(), send_data, sizeof(float) * num_elems * 3, cudaMemcpyDeviceToHost));
    bool valid = true;
    for (int i = 0; i < num_elems * 3; ++i)
    {
        if (host_data[i] != float(i / num_elems) + 1)
        {
            std::cerr << "Data validation failed at index " << i << ": expected " << float(i / num_elems) + 1
                      << ", got " << host_data[i] << std::endl;
            valid = false;
            break;
        }
    }
    if (!valid)
    {
        std::cerr << "Data validation failed for rank " << rank << std::endl;
        exit(EXIT_FAILURE);
    } else
    {
        std::cout << "Data validation passed for rank " << rank << std::endl;
    }
}

template <typename ExecutionPolicy, bool PerfTest = true, bool CheckTest = true>
void n2one_comm(int rank, int world_size, ncclComm_t comm = nullptr)
{
    bool use_nvshmem_malloc = std::is_same<ExecutionPolicy, NvshmemExecutionPolicy>::value;

    cudaStream_t stream;
    float       *send_data, *recv_data;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (use_nvshmem_malloc)
    {
        send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems * 3); // 65536 B * 3
        recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems * 3);
    } else
    {
        CUDA_CHECK(cudaMalloc((void **)&send_data, sizeof(float) * num_elems * 3));
        CUDA_CHECK(cudaMalloc((void **)&recv_data, sizeof(float) * num_elems * 3));
    }

    assert(send_data != NULL && recv_data != NULL);

    assert(num_elems % (THREADS_PER_BLOCK * num_blocks) == 0);
    int num_elems_per_block = num_elems / num_blocks;

    if (CheckTest)
    {
        ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        check(send_data, rank, world_size);
    }

    if (PerfTest)
    {
        for (int i = 0; i < 10; ++i)
        {
            ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        }
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < 100; ++i)
        {
            ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        }
        timer.stop();
        float  elapsed_ms     = timer.elapsed_millis();
        double avg_runtime_us = double(elapsed_ms) / double(100) * 1000.0;

        std::cout << rank << "  Avg runtime: " << avg_runtime_us << " us" << std::endl;
    }

    if (use_nvshmem_malloc)
    {
        nvshmem_free(send_data);
        nvshmem_free(recv_data);
    } else
    {
        CUDA_CHECK(cudaFree(send_data));
        CUDA_CHECK(cudaFree(recv_data));
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <nccl|nvshmem>" << std::endl;
        return 1;
    }
    std::string mode = argv[1];

    if (mode == "nccl")
    {
        std::cout << "Running in NCCL mode" << std::endl;
        int          world_size = 4;
        ncclUniqueId id;
        ncclComm_t   comms[world_size];
        NCCL_CHECK(ncclGetUniqueId(&id));

        // Initialize all communicators
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < world_size; ++i)
        {
            CUDA_CHECK(cudaSetDevice(i));
            NCCL_CHECK(ncclCommInitRank(&comms[i], world_size, id, i));
        }
        NCCL_CHECK(ncclGroupEnd());

        std::vector<std::thread> threads;
        for (int i = 0; i < world_size; ++i)
        {
            threads.emplace_back(n2one_comm<NcclExecutionPolicy, true>, i, world_size, comms[i]);
        }
        for (auto &t : threads)
        {
            t.join();
        }
        for (int i = 0; i < world_size; ++i)
        {
            ncclCommDestroy(comms[i]);
        }
    } else if (mode == "nvshmem")
    {
        int mype, n_pes, mype_node;
        nvshmem_init();

        mype      = nvshmem_my_pe();
        n_pes     = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        n2one_comm<NvshmemExecutionPolicy>(mype, n_pes);

        nvshmem_finalize();
    }

    return 0;
}