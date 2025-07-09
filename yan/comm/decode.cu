#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <numeric>
#include <optional>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "helper.h"

#define HEAD_DIM_QKV__ 128

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

using namespace flashinfer;

using DTypeKV = half;
using DTypeQ  = half;
using DTypeO  = DTypeQ;
using IdType  = uint32_t;

using Params           = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = DefaultAttention<false, false, false, false>;

constexpr int      kHeadDim         = HEAD_DIM_QKV__;
constexpr uint32_t kPaddedBatchSize = 32;

QKVLayout     kv_layout             = flashinfer::QKVLayout::kHND;
int64_t       batch_size            = 4;
int64_t       num_qo_heads          = 8;
int64_t       num_kv_heads          = 8;
int64_t       page_size             = 16;
const int64_t max_num_pages_per_seq = 4;
const int64_t total_pages           = batch_size * max_num_pages_per_seq;

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
        size_t tmp_v_size             = kPaddedBatchSize * num_qo_heads * kHeadDim * sizeof(float);
        size_t tmp_s_size             = kPaddedBatchSize * num_qo_heads * sizeof(float);
        float_workspace_size_in_bytes = tmp_v_size + tmp_s_size + 256;
    }
    CUDA_CHECK(cudaMalloc(float_workspace_buffer, float_workspace_size_in_bytes));

    {
        size_t request_indices_size   = kPaddedBatchSize * sizeof(IdType);
        size_t kv_tile_indices_size   = kPaddedBatchSize * sizeof(IdType);
        size_t o_indptr_size          = (batch_size + 1) * sizeof(IdType);
        size_t kv_chunk_size_ptr_size = sizeof(IdType);
        size_t block_valid_mask_size  = kPaddedBatchSize * sizeof(bool);

        int_workspace_size_in_bytes = request_indices_size + kv_tile_indices_size + o_indptr_size +
                                      kv_chunk_size_ptr_size + block_valid_mask_size + 256;
    }

    CUDA_CHECK(cudaMalloc(int_workspace_buffer, int_workspace_size_in_bytes));
    CUDA_CHECK(cudaMallocHost(page_locked_int_workspace_buffer, int_workspace_size_in_bytes));
}

int main()
{
    int          mype, n_pes, mype_node;
    int          num_blocks;
    cudaStream_t stream;

    nvshmem_init();

    mype      = nvshmem_my_pe();
    n_pes     = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    void  *float_workspace_buffer           = nullptr;
    void  *int_workspace_buffer             = nullptr;
    void  *page_locked_int_workspace_buffer = nullptr;
    size_t float_workspace_size_in_bytes, int_workspace_size_in_bytes;
    setup_workspace(
        stream, &float_workspace_buffer, &int_workspace_buffer,
        &page_locked_int_workspace_buffer, float_workspace_size_in_bytes,
        int_workspace_size_in_bytes);

    DTypeQ *q; // [B, H, D]
    DTypeO *o;

    DTypeKV *paged_k_cache; // [X, H, page_size, D]
    DTypeKV *paged_v_cache; // [X, H, page_size, D]
    IdType  *paged_kv_indptr;
    IdType  *paged_kv_indices;
    IdType  *paged_kv_last_page_len;

    std::vector<DTypeQ> h_q(batch_size * num_qo_heads * kHeadDim);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<DTypeQ>(i % 11 * 0.1);

    // h_paged_k/v_cache: [total_pages, num_kv_heads, page_size, HEAD_DIM_QK]
    std::vector<DTypeKV> h_paged_k_cache(total_pages * num_kv_heads * page_size * kHeadDim);
    std::vector<DTypeKV> h_paged_v_cache(total_pages * num_kv_heads * page_size * kHeadDim);
    for (size_t i = 0; i < h_paged_k_cache.size(); ++i)
    {
        h_paged_k_cache[i] = static_cast<DTypeKV>(i % 13 * 0.1);
        h_paged_v_cache[i] = static_cast<DTypeKV>(i % 17 * 0.1);
    }

    std::vector<IdType> h_paged_kv_indptr = {0, 2, 5, 7, 9};
    std::vector<IdType> h_paged_kv_indices(h_paged_kv_indptr.back());
    std::iota(h_paged_kv_indices.begin(), h_paged_kv_indices.end(), 0);
    std::vector<IdType> h_paged_kv_last_page_len = {4, 8, 16, 9};

    DecodePlanInfo plan_info = plan(
        stream, float_workspace_buffer, int_workspace_buffer,
        page_locked_int_workspace_buffer, float_workspace_size_in_bytes,
        int_workspace_size_in_bytes, h_paged_kv_indptr.data());

    // CUDA_CHECK(cudaMalloc(&q, h_q.size() * sizeof(DTypeQ)));
    q = (DTypeQ *)nvshmem_malloc(h_q.size() * sizeof(DTypeQ));
    CUDA_CHECK(cudaMalloc(&o, h_q.size() * sizeof(DTypeO)));
    CUDA_CHECK(cudaMalloc(&paged_k_cache, h_paged_k_cache.size() * sizeof(DTypeKV)));
    CUDA_CHECK(cudaMalloc(&paged_v_cache, h_paged_v_cache.size() * sizeof(DTypeKV)));
    CUDA_CHECK(cudaMalloc(&paged_kv_indptr, h_paged_kv_indptr.size() * sizeof(IdType)));
    CUDA_CHECK(cudaMalloc(&paged_kv_indices, h_paged_kv_indices.size() * sizeof(IdType)));
    CUDA_CHECK(cudaMalloc(&paged_kv_last_page_len, h_paged_kv_last_page_len.size() * sizeof(IdType)));

    CUDA_CHECK(cudaMemcpy(q, h_q.data(), h_q.size() * sizeof(DTypeQ), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_k_cache, h_paged_k_cache.data(), h_paged_k_cache.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_v_cache, h_paged_v_cache.data(), h_paged_v_cache.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv_indptr, h_paged_kv_indptr.data(), h_paged_kv_indptr.size() * sizeof(IdType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv_indices, h_paged_kv_indices.data(), h_paged_kv_indices.size() * sizeof(IdType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(paged_kv_last_page_len, h_paged_kv_last_page_len.data(), h_paged_kv_last_page_len.size() * sizeof(IdType), cudaMemcpyHostToDevice));

    std::vector<int64_t> kv_strides = {num_kv_heads * page_size * kHeadDim,
                                       page_size * kHeadDim, kHeadDim};

    paged_kv_t<DTypeKV, IdType> paged_kv(
        num_kv_heads, page_size, kHeadDim, batch_size, kv_layout,
        paged_k_cache, paged_v_cache, kv_strides.data(),
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len);

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

    auto run_kernel = [&]() {
        cudaError_t status = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
            kHeadDim, PosEncodingMode::kNone, AttentionVariant>(params, tmp_v, tmp_s, true, stream);

        if (status != cudaSuccess)
        {
            fprintf(stderr, "BatchDecodeWithPagedKVCache failed with error %s\n",
                    cudaGetErrorString(status));
            exit(-1);
        }
    };

    run_kernel();
    std::vector<DTypeO> h_o(batch_size * num_qo_heads * kHeadDim);
    CUDA_CHECK(cudaMemcpy(h_o.data(), o, h_o.size() * sizeof(DTypeO), cudaMemcpyDeviceToHost));
    print_raw_tensor(h_o);

    for (int32_t i = 0; i < 10; i++)
    {
        run_kernel();
    }

    {
        int      iterations = 20;
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < 20; ++iter)
        {
            run_kernel();
        }
        timer.stop();

        float  elapsed_ms     = timer.elapsed_millis();
        double avg_runtime_ms = double(elapsed_ms) / double(iterations);

        std::cout << "  Avg runtime: " << avg_runtime_ms << " ms" << std::endl;
    }

    // Cleanup
    nvshmem_free(q);
    CUDA_CHECK(cudaFree(o));
    CUDA_CHECK(cudaFree(paged_k_cache));
    CUDA_CHECK(cudaFree(paged_v_cache));
    CUDA_CHECK(cudaFree(paged_kv_indptr));
    CUDA_CHECK(cudaFree(paged_kv_indices));
    CUDA_CHECK(cudaFree(paged_kv_last_page_len));
    CUDA_CHECK(cudaFree(float_workspace_buffer));
    CUDA_CHECK(cudaFree(int_workspace_buffer));
    CUDA_CHECK(cudaFreeHost(page_locked_int_workspace_buffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
    return 0;
}