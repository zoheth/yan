#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <flashinfer/page.cuh>

struct ProblemConfig
{
    uint32_t batch_size = 4;
    uint32_t num_qo_heads = 64;
    uint32_t num_kv_heads = 8;
    uint32_t seq_len = 50000; // 50k tokens
    uint32_t head_dim = 128;

    uint32_t              page_size;
    uint32_t              max_seq_len;
    uint32_t              max_num_pages_per_seq;
    uint32_t              total_pages;

    flashinfer::QKVLayout kv_layout;
};

template <typename DTypeQ, typename DTypeO, typename DTypeKV, typename IdType, bool kKVConcat = false>
void setup_test_data(ProblemConfig config, DTypeQ **q, DTypeO **o, flashinfer::paged_kv_t<DTypeKV, IdType> &paged_kv, int rank, bool use_nvshmem_malloc)
{
    using namespace flashinfer;

    // DTypeKV *paged_k_cache; // [X, H, page_size, D]
    // DTypeKV *paged_v_cache; // [X, H, page_size, D]
    paged_kv.num_heads              = config.num_kv_heads;
    paged_kv.page_size              = config.page_size;
    paged_kv.head_dim               = config.head_dim;
    paged_kv.batch_size             = config.batch_size;
    std::vector<int64_t> kv_strides = {config.num_kv_heads * config.page_size * config.head_dim,
                                       config.page_size * config.head_dim, config.head_dim};

    paged_kv.stride_page = kv_strides[0];
    paged_kv.stride_n    = config.kv_layout == QKVLayout::kHND ? kv_strides[2] : kv_strides[1];
    paged_kv.stride_h    = config.kv_layout == QKVLayout::kHND ? kv_strides[1] : kv_strides[2];

    std::vector<DTypeQ> h_q(config.batch_size * config.num_qo_heads * config.head_dim);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<DTypeQ>(i % 11 * 0.1);

    // h_paged_k/v_cache: [total_pages, num_kv_heads, page_size, HEAD_DIM_QK]
    std::vector<DTypeKV> h_paged_k_cache(config.total_pages * config.num_kv_heads * config.page_size * config.head_dim);
    std::vector<DTypeKV> h_paged_v_cache(h_paged_k_cache.size());

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);
    for (size_t i = 0; i < h_paged_k_cache.size(); ++i)
    {
        h_paged_k_cache[i] = static_cast<DTypeKV>(distrib(gen));
        h_paged_v_cache[i] = static_cast<DTypeKV>(distrib(gen));
    }

    std::vector<IdType> h_paged_kv_indptr(config.batch_size + 1);
    for(size_t i = 0; i < h_paged_kv_indptr.size(); ++i)
    {
        h_paged_kv_indptr[i] = i * config.max_num_pages_per_seq;
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
    if constexpr (kKVConcat)
    {
        CUDA_CHECK(cudaMalloc(&paged_kv.k_data, 2 * h_paged_k_cache.size() * sizeof(DTypeKV)));
    } else
    {
        CUDA_CHECK(cudaMalloc(&paged_kv.k_data, h_paged_k_cache.size() * sizeof(DTypeKV)));
        CUDA_CHECK(cudaMalloc(&paged_kv.v_data, h_paged_v_cache.size() * sizeof(DTypeKV)));
    }
    
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