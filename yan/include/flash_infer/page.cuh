#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <torch/library.h>

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

enum class QKVLayout
{
    // [seq_len, num_heads, head_dim]
    kNHD = 0U,
    // [num_heads, seq_len, head_dim]
    kHND = 1U,
};

template <typename DType, typename IdType>
struct paged_kv_t
{
    // todo
    uint32_t page_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t batch_size;
    uint32_t stride_page;
    uint32_t stride_n;
    uint32_t stride_h;

    // Internal layout:
    // [max_num_pages, num_heads, page_size, head_dim] if layout == HND
    // [max_num_pages, page_size, num_heads, head_dim] if layout == NHD
    DType  *k_data;
    DType  *v_data;
    IdType *indices;

    // todo
    // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
    IdType *indptr;
    // [batch_size] The offset of the last page for each request in the batch
    IdType *last_page_len;
    // [batch_size] The start position of each request in the batch.
    IdType *rope_pos_offset;
};

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices, at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len, int64_t layout)
{
    unsigned int nnz        = append_value.size(0);
    unsigned int batch_size = kv_last_page_len.size(0);
    CHECK_EQ(kv_indptr.size(0), batch_size + 1);
    CHECK_EQ(batch_indices.size(0), nnz);
    CHECK_EQ(positions.size(0), nnz);

    auto device = append_key.device();

    QKVLayout kv_layout = QKVLayout(layout);

    unsigned int num_heads, page_size, head_dim;
    head_dim = paged_k_cache.size(3);
    if (kv_layout == QKVLayout::kHND)
    {
        num_heads = paged_k_cache.size(1);
        page_size = paged_k_cache.size(2);
    } else
    {
        page_size = paged_k_cache.size(1);
        num_heads = paged_k_cache.size(2);
    }

    const int64_t* kv_cache_strides = nullptr;
    auto k_strides = paged_k_cache.strides();
    auto v_strides = paged_v_cache.strides();
    TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
    kv_cache_strides = k_strides.data();

    auto append_k_strides = append_key.strides();
}