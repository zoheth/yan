#include <cassert>
#include <cstdlib>
#include <device_host_transport/nvshmem_common_transport.h>
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

__global__ void produce_kernel(float *data, int n_elems_per_block, int rank)
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
    nvshmemx_float_put_signal_nbi_block(recv_data + block_offset, send_data + block_offset, n_elems_per_block,
                                        signal, 1, NVSHMEM_SIGNAL_ADD, peer);
}

__global__ void produce_and_send_kernel(float *send_data, float *recv_data, int n_elems_per_block, int mype, uint64_t *signal)
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

__global__ void wait_and_consume_kernel(float *output, float *recv_data, int n_elems_per_block, int mype, int n_pes, uint64_t *signal)
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

__global__ void consume_kernel(float *output, float *recv_data, int n_elems_per_block, int mype, int n_pes)
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
                    float *send_data, float *recv_data, int num_elems_per_block,
                    ncclComm_t comm)
    {
        if (rank != 0)
        {
            produce_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, num_elems_per_block, rank);
        }

        if (rank == 0)
        {
            NCCL_CHECK(ncclGroupStart());
            for (int r = 1; r < world_size; ++r)
            {
                NCCL_CHECK(ncclRecv(recv_data, num_elems, ncclFloat, r, comm, stream));
                recv_data += num_elems;
            }
            NCCL_CHECK(ncclGroupEnd());

            recv_data -= num_elems * (world_size - 1);
        } else
        {
            NCCL_CHECK(ncclSend(send_data, num_elems, ncclFloat, 0, comm, stream));
        }

        if (rank == 0)
        {

            consume_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems_per_block, rank, world_size);
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            // comm_us += comm_timer->elapsed_millis() * 1000.0;
        }
    }
};

struct NvshmemExecutionPolicy
{
    static void run(int rank, int world_size, cudaStream_t stream,
                    float *send_data, float *recv_data, int num_elems_per_block,
                    ncclComm_t comm /* unused */)
    {
        static uint64_t *signal = (uint64_t *)nvshmem_calloc(num_blocks, sizeof(uint64_t));
        if (rank != 0)
        {
            void *args[] = {&send_data, &recv_data, &num_elems_per_block, &rank, &signal};
            nvshmemx_collective_launch((const void *)produce_and_send_kernel, num_blocks, THREADS_PER_BLOCK, args, 0, stream);
            // void *args[] = {&send_data, &num_elems_per_block, &rank};
            // nvshmemx_collective_launch((const void *)produce_kernel, num_blocks, THREADS_PER_BLOCK, args, 0, stream);
            // void *args1[] = {&send_data, &recv_data, &num_elems_per_block, &rank, &signal};
            // nvshmemx_collective_launch((const void *)send_kernel, num_blocks, THREADS_PER_BLOCK, args1, 0, stream);

            // nvshmemx_float_put_signal_on_stream(recv_data, send_data, 16384, signal, 1, NVSHMEM_SIGNAL_ADD, 0, stream);

        } else
        {
            // nvshmemx_signal_wait_until_on_stream(signal, NVSHMEM_CMP_GE, 3, stream);
            void *args[] = {&send_data, &recv_data, &num_elems_per_block, &rank, &world_size, &signal};
            nvshmemx_collective_launch((const void *)wait_and_consume_kernel, num_blocks, THREADS_PER_BLOCK, args, 0, stream);
            // nvshmemx_signal_op_on_stream(signal, 0, NVSHMEM_SIGNAL_SET, 0, stream);
        }
        nvshmemx_barrier_all_on_stream(stream);
    }
};

void check(float *output, int rank, int world_size = 4)
{
    if (rank != 0)
    {
        return;
    }
    std::vector<float> host_data(num_elems * (world_size - 1), 0.0f);
    CUDA_CHECK(cudaMemcpy(host_data.data(), output, sizeof(float) * num_elems * (world_size - 1), cudaMemcpyDeviceToHost));
    bool valid = true;
    for (int i = 0; i < num_elems * (world_size - 1); ++i)
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
        // exit(EXIT_FAILURE);
    } else
    {
        std::cout << "Data validation passed for rank " << rank << std::endl;
    }
}

template <typename ExecutionPolicy, bool PerfTest = true, bool CheckTest = true>
void n2one_comm(int rank, int world_size, int local_rank,
                ncclComm_t comm = nullptr, void *send_buffer = nullptr, void *recv_buffer = nullptr)
{
    bool use_nvshmem_malloc = std::is_same<ExecutionPolicy, NvshmemExecutionPolicy>::value;

    cudaStream_t stream;
    float       *send_data, *recv_data;
    CUDA_CHECK(cudaSetDevice(local_rank));
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (use_nvshmem_malloc)
    {
        send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems * (world_size - 1)); // 65536 B * (world_size - 1)
        recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems * (world_size - 1));
    } else if (send_buffer && recv_buffer)
    {
        send_data = (float *)send_buffer;
        recv_data = (float *)recv_buffer;
    } else
    {
        CUDA_CHECK(cudaMalloc((void **)&send_data, sizeof(float) * num_elems * (world_size - 1)));
        CUDA_CHECK(cudaMalloc((void **)&recv_data, sizeof(float) * num_elems * (world_size - 1)));
    }

    // std::vector<float> host_data(num_elems, rank);
    // CUDA_CHECK(cudaMemcpyAsync(send_data, host_data.data(), num_elems * sizeof(float), cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    assert(send_data != NULL && recv_data != NULL);

    assert(num_elems % (THREADS_PER_BLOCK * num_blocks) == 0);
    int num_elems_per_block = num_elems / num_blocks;

    double comm_us = 0.0;
    if (CheckTest)
    {
        ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        check(send_data, rank, world_size);
    }

    if (PerfTest)
    {
        double comm_us = 0.0;
        for (int i = 0; i < 10; ++i)
        {
            ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        }
        comm_us = 0.0;
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < 100; ++i)
        {
            ExecutionPolicy::run(rank, world_size, stream, send_data, recv_data, num_elems_per_block, comm);
        }
        timer.stop();
        float  elapsed_ms     = timer.elapsed_millis();
        double avg_runtime_us = double(elapsed_ms) / double(100) * 1000.0;

        std::cout << rank << "Avg runtime: " << avg_runtime_us << " us" << std::endl;
    }

    if (use_nvshmem_malloc)
    {
        nvshmem_free(send_data);
        nvshmem_free(recv_data);
    } else if (send_buffer == nullptr)
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
        int          local_size = 4;
        ncclUniqueId id;
        ncclComm_t   comms[world_size];
        NCCL_CHECK(ncclGetUniqueId(&id));

        void *send_buffers[world_size];
        void *recv_buffers[world_size];

        size_t data_size = sizeof(float) * num_elems * (world_size - 1);

        for (int i = 0; i < world_size; ++i)
        {
            CUDA_CHECK(cudaSetDevice(i));
            ncclMemAlloc((void **)(send_buffers + i), data_size);
            ncclMemAlloc((void **)(recv_buffers + i), data_size);
        }

        // Initialize all communicators
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < world_size; ++i)
        {
            CUDA_CHECK(cudaSetDevice(i));
            NCCL_CHECK(ncclCommInitRank(&comms[i], world_size, id, i));
        }
        NCCL_CHECK(ncclGroupEnd());

        void **send_reg_handles = (void **)malloc(sizeof(*send_reg_handles) * world_size);
        void **recv_reg_handles = (void **)malloc(sizeof(*recv_reg_handles) * world_size);

        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < world_size; ++i)
        {
            NCCL_CHECK(ncclCommWindowRegister(comms[i], send_buffers[i], data_size, (ncclWindow_t *)&send_reg_handles[i], NCCL_WIN_COLL_SYMMETRIC));
            NCCL_CHECK(ncclCommWindowRegister(comms[i], recv_buffers[i], data_size, (ncclWindow_t *)&recv_reg_handles[i], NCCL_WIN_COLL_SYMMETRIC));
        }
        NCCL_CHECK(ncclGroupEnd());

        std::vector<std::thread> threads;
        for (int i = 0; i < world_size; ++i)
        {
            // threads.emplace_back(n2one_comm<NcclExecutionPolicy, true>, i, world_size, comms_timer, comms[i], nullptr, nullptr);
            threads.emplace_back(n2one_comm<NcclExecutionPolicy, true>, i, world_size, i % local_size, comms[i], send_buffers[i], recv_buffers[i]);
        }
        for (auto &t : threads)
        {
            t.join();
        }
        for (int i = 0; i < world_size; ++i)
        {
            NCCL_CHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)send_reg_handles[i]));
            NCCL_CHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)recv_reg_handles[i]));
            ncclCommDestroy(comms[i]);
        }
    } else if (mode == "nvshmem")
    {
        int mype, n_pes, mype_node;
        nvshmem_init();

        mype      = nvshmem_my_pe();
        n_pes     = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        n2one_comm<NvshmemExecutionPolicy>(mype, n_pes, mype_node);

        nvshmem_finalize();
    }

    return 0;
}