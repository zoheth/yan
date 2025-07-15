#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <device_host_transport/nvshmem_common_transport.h>
#include <device_host_transport/nvshmem_constants.h>
#include <host/nvshmemx_api.h>
#include <numeric>
#include <optional>
#include <random>
#include <thread>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <nccl.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "decode_launch.cuh"
#include "helper.h"

#define HEAD_DIM_QKV__ 128

using namespace flashinfer;

using DTypeKV = half;
using DTypeQ  = half;
using DTypeO  = DTypeQ;
using IdType  = uint32_t;

using Params           = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = DefaultAttention<false, false, false, false>;

constexpr int      kHeadDim         = HEAD_DIM_QKV__;
constexpr uint32_t kPaddedBatchSize = 32;

QKVLayout           kv_layout             = flashinfer::QKVLayout::kHND;
int64_t             batch_size            = 4;
int64_t             num_qo_heads          = 8;
int64_t             num_kv_heads          = 8;
int64_t             page_size             = 64;
const int64_t       max_num_pages_per_seq = 320;
const int64_t       total_pages           = batch_size * max_num_pages_per_seq;
std::vector<IdType> h_paged_kv_indptr     = {0, 300, 500, 780, 1100};
size_t              qo_data_size          = batch_size * num_qo_heads * kHeadDim;

struct CudaExecutionPolicy
{
    static void run(
        Params &params, DTypeO *tmp_v, float *tmp_s, cudaStream_t stream,
        int rank, int world_size, ncclComm_t comm = nullptr)
    {
        for (int i = 0; i < 3; ++i)
        {
            checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                            CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
            checkCuda(cudaStreamSynchronize(stream));

            params.q += qo_data_size;
            params.o += qo_data_size;
        }
        params.q -= qo_data_size * 3;
        params.o -= qo_data_size * 3;
        std::swap(params.q, params.o);
        for (int i = 0; i < 3; ++i)
        {
            checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                            CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
            checkCuda(cudaStreamSynchronize(stream));

            params.q += qo_data_size;
            params.o += qo_data_size;
        }
        params.q -= qo_data_size * 3;
        params.o -= qo_data_size * 3;
    }

    static void check(DTypeO *o, int rank, int world_size)
    {
        std::vector<DTypeO> h_o(qo_data_size * 3);
        checkCuda(cudaMemcpy(h_o.data(), o, h_o.size() * sizeof(DTypeO), cudaMemcpyDeviceToHost));
        print_raw_tensor(h_o);
    }
};

struct NcclExecutionPolicy
{
    static void run(
        Params &params, DTypeO *tmp_v, float *tmp_s, cudaStream_t stream,
        int rank, int world_size, ncclComm_t comm)
    {
        if (rank != 0)
        {
            checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                            CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
        }

        NCCL_CHECK(ncclGroupStart());
        if (rank == 0)
        {
            for (int r = 1; r < world_size; ++r)
            {
                NCCL_CHECK(ncclRecv(params.q, qo_data_size, ncclHalf, r, comm, stream));
                params.q += qo_data_size;
            }
            params.q -= qo_data_size * (world_size - 1);
        } else
        {
            NCCL_CHECK(ncclSend(params.o, qo_data_size, ncclHalf, 0, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        // checkCuda(cudaStreamSynchronize(stream));

        if (rank == 0)
        {
            for (int i = 0; i < 3; ++i)
            {
                checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                                CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
                params.q += qo_data_size;
                params.o += qo_data_size;
            }
            params.q -= qo_data_size * 3;
            params.o -= qo_data_size * 3;
        }
    }

    static void check(DTypeO *o, int rank, int world_size)
    {
        if (rank == 0)
        {
            std::vector<DTypeO> h_o(qo_data_size * 3);
            checkCuda(cudaMemcpy(h_o.data(), o, h_o.size() * sizeof(DTypeO), cudaMemcpyDeviceToHost));
            print_raw_tensor(h_o);
        }
    }
};

// Strategy for multi-GPU execution using NVSHMEM
struct NvshmemExecutionPolicy
{
    static void run(
        Params &params, DTypeO *tmp_v, float *tmp_s, cudaStream_t stream,
        int rank, int world_size, ncclComm_t comm /* unused */
    )
    {
        static uint64_t* signals = (uint64_t *)nvshmem_calloc(2, sizeof(uint64_t));
        uint64_t* full = &signals[0];
        uint64_t* dont_produce = &signals[1];

        if (rank != 0)
        {
            // nvshmemx_signal_wait_until_on_stream(dont_produce, NVSHMEM_CMP_EQ, 0, stream);
            // nvshmemx_signal_op_on_stream(dont_produce, 1, NVSHMEM_SIGNAL_SET, rank, stream); 
            checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                            CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
            nvshmemx_half_put_signal_on_stream(params.q + ((rank - 1) * qo_data_size), params.o, qo_data_size, full, 1, NVSHMEM_SIGNAL_ADD, 0, stream);
        }
        if (rank == 0)
        {
            const int num_workers = world_size - 1;
            nvshmemx_signal_wait_until_on_stream(full, NVSHMEM_CMP_GE, num_workers, stream);

            for (int i = 0; i < num_workers; ++i)
            {
                checkCuda(BatchDecodeWithPagedKVCacheDispatched<kHeadDim, PosEncodingMode::kNone, AttentionVariant,
                                                                CudaLaunchPolicy>(params, tmp_v, tmp_s, false, stream));
                params.q += qo_data_size;
                params.o += qo_data_size;
            }
            // nvshmemx_signal_op_on_stream(full, 0, NVSHMEM_SIGNAL_SET, 0, stream);
            // for (int r = 1; r < world_size; ++r)
            // {
            //     nvshmemx_signal_op_on_stream(dont_produce, 0, NVSHMEM_SIGNAL_SET, r, stream);
            // }
            params.q -= qo_data_size * num_workers;
            params.o -= qo_data_size * num_workers;
        }
        nvshmemx_barrier_all_on_stream(stream);
        if(rank == 0)
        {
            nvshmemx_signal_op_on_stream(full, 0, NVSHMEM_SIGNAL_SET, 0, stream);
        }

    }

    static void check(DTypeO *o, int rank, int world_size)
    {
        if (rank == 0)
        {
            std::vector<DTypeO> h_o(qo_data_size * 3);
            checkCuda(cudaMemcpy(h_o.data(), o, h_o.size() * sizeof(DTypeO), cudaMemcpyDeviceToHost));
            print_raw_tensor(h_o);
        }
    }
};

DecodePlanInfo plan(const cudaStream_t stream, void *float_workspace_buffer, void *int_workspace_buffer, void *page_locked_int_workspace_buffer,
                    size_t float_workspace_size_in_bytes, size_t int_workspace_size_in_bytes, IdType *indptr)
{
    DecodePlanInfo plan_info;

    auto work_estimation_func = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
        1, kHeadDim, PosEncodingMode::kNone, AttentionVariant, Params>;
    cudaError_t status = DecodePlan<kHeadDim, PosEncodingMode::kNone, AttentionVariant, Params>(
        float_workspace_buffer, float_workspace_size_in_bytes,
        int_workspace_buffer, page_locked_int_workspace_buffer, int_workspace_size_in_bytes,
        plan_info, indptr, batch_size, num_qo_heads, page_size, false, stream,
        work_estimation_func);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "DecodePlan failed with error %s\n", cudaGetErrorString(status));
        exit(-1);
    }
    return plan_info;
}

void setup_workspace(
    cudaStream_t stream, void **float_workspace_buffer, void **int_workspace_buffer,
    void **page_locked_int_workspace_buffer, size_t &float_workspace_size_in_bytes,
    size_t &int_workspace_size_in_bytes)
{
    {
        size_t tmp_v_size             = total_pages * num_qo_heads * kHeadDim * sizeof(float);
        size_t tmp_s_size             = total_pages * num_qo_heads * sizeof(float);
        float_workspace_size_in_bytes = tmp_v_size + tmp_s_size + 256;
    }
    CUDA_CHECK(cudaMalloc(float_workspace_buffer, float_workspace_size_in_bytes));

    {
        size_t request_indices_size   = kPaddedBatchSize * sizeof(IdType);
        size_t kv_tile_indices_size   = total_pages * sizeof(IdType);
        size_t o_indptr_size          = (batch_size + 1) * sizeof(IdType);
        size_t kv_chunk_size_ptr_size = sizeof(IdType);
        size_t block_valid_mask_size  = kPaddedBatchSize * sizeof(bool);

        int_workspace_size_in_bytes = request_indices_size + kv_tile_indices_size + o_indptr_size +
                                      kv_chunk_size_ptr_size + block_valid_mask_size + 256;
    }

    CUDA_CHECK(cudaMalloc(int_workspace_buffer, int_workspace_size_in_bytes));
    CUDA_CHECK(cudaMallocHost(page_locked_int_workspace_buffer, int_workspace_size_in_bytes));
}

void setup_test_data(DTypeQ **q, DTypeO **o, paged_kv_t<DTypeKV, IdType> &paged_kv, int rank, bool use_nvshmem_malloc)
{

    // DTypeKV *paged_k_cache; // [X, H, page_size, D]
    // DTypeKV *paged_v_cache; // [X, H, page_size, D]
    paged_kv.num_heads              = num_kv_heads;
    paged_kv.page_size              = page_size;
    paged_kv.head_dim               = kHeadDim;
    paged_kv.batch_size             = batch_size;
    std::vector<int64_t> kv_strides = {num_kv_heads * page_size * kHeadDim,
                                       page_size * kHeadDim, kHeadDim};

    paged_kv.stride_page = kv_strides[0];
    paged_kv.stride_n    = kv_layout == QKVLayout::kHND ? kv_strides[2] : kv_strides[1];
    paged_kv.stride_h    = kv_layout == QKVLayout::kHND ? kv_strides[1] : kv_strides[2];

    std::vector<DTypeQ> h_q(qo_data_size);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<DTypeQ>(i % 11 * 0.1);

    // h_paged_k/v_cache: [total_pages, num_kv_heads, page_size, HEAD_DIM_QK]
    std::vector<DTypeKV> h_paged_k_cache(total_pages * num_kv_heads * page_size * kHeadDim);
    std::vector<DTypeKV> h_paged_v_cache(total_pages * num_kv_heads * page_size * kHeadDim);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);
    for (size_t i = 0; i < h_paged_k_cache.size(); ++i)
    {
        h_paged_k_cache[i] = static_cast<DTypeKV>(distrib(gen));
        h_paged_v_cache[i] = static_cast<DTypeKV>(distrib(gen));
    }

    std::vector<IdType> h_paged_kv_indices(h_paged_kv_indptr.back());
    std::iota(h_paged_kv_indices.begin(), h_paged_kv_indices.end(), 0);
    std::vector<IdType> h_paged_kv_last_page_len = {40, 58, 64, 2};

    size_t qo_buffer_size = h_q.size() * sizeof(DTypeQ);
    if (rank == 0)
    {
        qo_buffer_size *= 3;
    }

    if (use_nvshmem_malloc)
    {
        *q = (DTypeQ *)nvshmem_malloc(h_q.size() * sizeof(DTypeQ) * 3);
    } else
    {
        CUDA_CHECK(cudaMalloc(q, qo_buffer_size));
    }
    CUDA_CHECK(cudaMalloc(o, qo_buffer_size));
    CUDA_CHECK(cudaMalloc(&paged_kv.k_data, h_paged_k_cache.size() * sizeof(DTypeKV)));
    CUDA_CHECK(cudaMalloc(&paged_kv.v_data, h_paged_v_cache.size() * sizeof(DTypeKV)));
    CUDA_CHECK(cudaMalloc(&paged_kv.indptr, h_paged_kv_indptr.size() * sizeof(IdType)));
    CUDA_CHECK(cudaMalloc(&paged_kv.indices, h_paged_kv_indices.size() * sizeof(IdType)));
    CUDA_CHECK(cudaMalloc(&paged_kv.last_page_len, h_paged_kv_last_page_len.size() * sizeof(IdType)));

    CUDA_CHECK(cudaMemcpy(*q, h_q.data(), h_q.size() * sizeof(DTypeQ), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv.k_data, h_paged_k_cache.data(), h_paged_k_cache.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv.v_data, h_paged_v_cache.data(), h_paged_v_cache.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv.indptr, h_paged_kv_indptr.data(), h_paged_kv_indptr.size() * sizeof(IdType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv.indices, h_paged_kv_indices.data(), h_paged_kv_indices.size() * sizeof(IdType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv.last_page_len, h_paged_kv_last_page_len.data(), h_paged_kv_last_page_len.size() * sizeof(IdType), cudaMemcpyHostToDevice));
}

template <typename ExecutionPolicy, bool PerfTest = false, bool CheckTest = true>
void decode_with_kvcache(int rank, int world_size, ncclComm_t comm = nullptr)
{
    CUDA_CHECK(cudaSetDevice(rank));

    bool use_nvshmem_malloc = std::is_same<ExecutionPolicy, NvshmemExecutionPolicy>::value;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void  *float_workspace_buffer           = nullptr;
    void  *int_workspace_buffer             = nullptr;
    void  *page_locked_int_workspace_buffer = nullptr;
    size_t float_workspace_size_in_bytes, int_workspace_size_in_bytes;
    setup_workspace(
        stream, &float_workspace_buffer, &int_workspace_buffer,
        &page_locked_int_workspace_buffer, float_workspace_size_in_bytes,
        int_workspace_size_in_bytes);

    DecodePlanInfo plan_info = plan(
        stream, float_workspace_buffer, int_workspace_buffer,
        page_locked_int_workspace_buffer, float_workspace_size_in_bytes,
        int_workspace_size_in_bytes, h_paged_kv_indptr.data());

    DTypeQ *q; // [B, H, D]
    DTypeO *o;

    paged_kv_t<DTypeKV, IdType> paged_kv;
    setup_test_data(&q, &o, paged_kv, rank, use_nvshmem_malloc);

    const int64_t q_stride_n = kHeadDim;
    const int64_t q_stride_h = kHeadDim;
    using Params             = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
    Params params;
    params.q            = q;
    params.paged_kv     = paged_kv;
    params.o            = o;
    params.num_qo_heads = num_qo_heads;
    params.q_stride_n   = q_stride_n;
    params.q_stride_h   = q_stride_h;

    DTypeO *tmp_v = nullptr;
    float  *tmp_s = nullptr;
    params.request_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.request_indices_offset);
    params.kv_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.kv_chunk_size_ptr_offset);
    if (plan_info.split_kv)
    {
        tmp_v = GetPtrFromBaseOffset<DTypeO>(float_workspace_buffer, plan_info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_workspace_buffer, plan_info.s_offset);
        if (plan_info.enable_cuda_graph)
        {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_workspace_buffer, plan_info.block_valid_mask_offset);
        }
    }
    params.padded_batch_size = plan_info.padded_batch_size;

    if (CheckTest)
    {
        ExecutionPolicy::run(params, tmp_v, tmp_s, stream, rank, world_size, comm);
        ExecutionPolicy::check(params.o, rank, world_size);
    }

    if (PerfTest)
    {
        for (int32_t i = 0; i < 10; i++)
        {
            ExecutionPolicy::run(params, tmp_v, tmp_s, stream, rank, world_size, comm);
        }

        {
            int      iterations = 20;
            GpuTimer timer;
            timer.start(stream);
            for (int iter = 0; iter < 20; ++iter)
            {
                ExecutionPolicy::run(params, tmp_v, tmp_s, stream, rank, world_size, comm);
            }
            timer.stop();

            float  elapsed_ms     = timer.elapsed_millis();
            double avg_runtime_ms = double(elapsed_ms) / double(iterations);

            std::cout << rank << "  Avg runtime: " << avg_runtime_ms << " ms" << std::endl;
        }
    }

    // Cleanup
    if (use_nvshmem_malloc)
    {
        nvshmem_free(q);
    } else
    {
        CUDA_CHECK(cudaFree(q));
    }
    CUDA_CHECK(cudaFree(o));
    CUDA_CHECK(cudaFree(paged_kv.k_data));
    CUDA_CHECK(cudaFree(paged_kv.v_data));
    CUDA_CHECK(cudaFree(paged_kv.indptr));
    CUDA_CHECK(cudaFree(paged_kv.indices));
    CUDA_CHECK(cudaFree(paged_kv.last_page_len));
    CUDA_CHECK(cudaFree(float_workspace_buffer));
    CUDA_CHECK(cudaFree(int_workspace_buffer));
    CUDA_CHECK(cudaFreeHost(page_locked_int_workspace_buffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <cuda|nccl|nvshmem>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "cuda")
    {
        std::cout << "Running in CUDA mode (single GPU)" << std::endl;
        // For CUDA, world_size is 1 and rank is 0. No comms needed.
        decode_with_kvcache<CudaExecutionPolicy, true>(0, 1);
    } else if (mode == "nccl")
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
            threads.emplace_back(decode_with_kvcache<NcclExecutionPolicy, true>, i, world_size, comms[i]);
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

        decode_with_kvcache<NvshmemExecutionPolicy, true>(mype, n_pes);

        nvshmem_finalize();
    }
    return 0;
}