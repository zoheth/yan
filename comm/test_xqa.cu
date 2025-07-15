// #include "xqa_sources.h"
#include <cstdint>
#include <mha.h>
#include <random>
#include <numeric> 
#include <ranges>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "helper.h"

using DTypeKV = __nv_fp8_e4m3;
using DTypeQ  = half;
using DTypeO  = DTypeQ;
using IdType  = KVCachePageIndex;

#define HEAD_DIM_QKV__ 128

constexpr int      kHeadDim         = HEAD_DIM_QKV__;
constexpr uint32_t kPaddedBatchSize = 32;

flashinfer::QKVLayout kv_layout             = flashinfer::QKVLayout::kHND;
int64_t               batch_size            = 4;
int64_t               num_qo_heads          = 8;
int64_t               num_kv_heads          = 8;
int64_t               page_size             = 64;
const int64_t         seq_len               = 50000; // 50k tokens
const int64_t         max_seq_len          = round_up(seq_len, page_size);
const uint32_t         max_num_pages_per_seq = div_up(max_seq_len, page_size);
const int64_t         total_pages           = batch_size * max_num_pages_per_seq;

size_t                qo_data_size          = batch_size * num_qo_heads * kHeadDim;

void setup_test_data(int rank, bool use_nvshmem_malloc)
{
    float const k_scale = 1 / 4.f;
    float const v_scale = k_scale;
    float const q_scale = 1.0f;

    // uint32_t const total_kv_heads = num_kv_heads * batch_size;

    size_t const num_semaphores = round_up<size_t>(num_kv_heads * batch_size, 2) + 2 +
                                  num_kv_heads * batch_size + 2;
    uint32_t *semaphores;
    checkCuda(cudaMalloc(&semaphores, num_semaphores * sizeof(uint32_t)));
    checkCuda(cudaMemset(semaphores, 0, num_semaphores * sizeof(uint32_t)));

    size_t const scratch_size = (256u << 20);
    void        *scratch_buffer;
    checkCuda(cudaMalloc(&scratch_buffer, scratch_size));
    checkCuda(cudaMemset(scratch_buffer, 0, scratch_size));

    float *kv_scale;
    checkCuda(cudaMalloc(&kv_scale, sizeof(float)));
    checkCuda(cudaMemset(kv_scale, k_scale, sizeof(float)));

    auto h_rope_cos_sin = std::vector<Vec<float, validElemsPerKHead>>(seq_len);

    auto h_cache_heads = std::vector<GMemCacheHead>(2* num_kv_heads * batch_size * max_seq_len);
    auto h_q_heads = std::vector<InputHead>(batch_size * num_qo_heads);
    auto h_o_heads = std::vector<OutputHead>(batch_size * num_qo_heads);
    auto h_seq_len_list = std::vector<uint32_t>(batch_size, seq_len);
    auto h_ctx_len_list = std::vector<uint32_t>(batch_size, 0);

    auto h_page_list = std::vector<KVCachePageIndex>(max_num_pages_per_seq);

    flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv;
}

int main()
{
    int rank   = 0;
    int device = rank;
    CUDA_CHECK(cudaSetDevice(rank));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DTypeQ *q; // [B, H, D]
    DTypeO *o;

    flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv;
    setup_test_data(&q, &o, paged_kv, rank, false);

    launchHopperF8MHA(
        prop,
        num_kv_heads,
        1.0f, // q scale
        o,
        q

    );
}