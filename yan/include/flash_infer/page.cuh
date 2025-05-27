#pragma once

template <typename DType, typename IdType>
struct paged_kv_t {
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
    DType* k_data;
    DType* v_data;
    IdType* indices;

    // todo
    // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages 
    IdType* indptr;
    // [batch_size] The offset of the last page for each request in the batch
    IdType* last_page_len;
    // [batch_size] The start position of each request in the batch.
    IdType* rope_pos_offset;
};

void append_paged_kv_cache()
{}