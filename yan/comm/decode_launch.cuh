#pragma once

#include <flashinfer/attention/decode.cuh>
#include <nvshmemx.h>

#include "helper.h"

struct CudaLaunchPolicy
{
    template <typename... Args>
    static void launchEx(Args &&...args)
    {
        CUDA_CHECK(cudaLaunchKernelEx(std::forward<Args>(args)...));
    }

    template <typename... Args>
    static void launch(Args &&...args)
    {
        CUDA_CHECK(cudaLaunchKernel(std::forward<Args>(args)...));
    }
};

struct NvshmemLaunchPolicy
{
    template <typename... Args>
    static void launchEx(Args &&...args)
    {
        throw std::runtime_error("nvshmemx_collective_launch_ex is not implemented");
    }

    template <typename... Args>
    static void launch(Args &&...args)
    {
        nvshmemx_collective_launch(std::forward<Args>(args)...);
    }
};

template <uint32_t HEAD_DIM, flashinfer::PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params, typename LaunchPolicy>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO *tmp_v,
                                                  float *tmp_s, bool enable_pdl,
                                                  cudaStream_t stream)
{
    using namespace flashinfer;
    using DTypeQ                     = typename Params::DTypeQ;
    using DTypeKV                    = typename Params::DTypeKV;
    using DTypeO                     = typename Params::DTypeO;
    using IdType                     = typename Params::IdType;
    const uint32_t num_qo_heads      = params.num_qo_heads;
    const uint32_t num_kv_heads      = params.paged_kv.num_heads;
    const uint32_t padded_batch_size = params.padded_batch_size;

    constexpr uint32_t vec_size         = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
    auto               compute_capacity = flashinfer::GetCudaComputeCapability();
    constexpr uint32_t bdx              = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
        constexpr uint32_t bdy               = GROUP_SIZE;
        constexpr uint32_t num_threads       = std::max(128U, bdx * bdy);
        constexpr uint32_t bdz               = num_threads / (bdx * bdy);
        constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
        DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
            const uint32_t smem_size =
                2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
                std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV *),
                         2 * bdy * bdz * sizeof(float));
            auto kernel =
                flashinfer::BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                                              vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
            FLASHINFER_CUDA_CALL(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            dim3 nblks(padded_batch_size, num_kv_heads);
            dim3 nthrs(bdx, bdy, bdz);

            // PDL launch config
            cudaLaunchAttribute attribute[1];
            cudaLaunchConfig_t  config;
            if (enable_pdl)
            {
                attribute[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
                attribute[0].val.programmaticStreamSerializationAllowed = 1;
                config.attrs                                            = attribute;
                config.numAttrs                                         = 1;
                config.gridDim                                          = nblks;
                config.blockDim                                         = nthrs;
                config.dynamicSmemBytes                                 = smem_size;
                config.stream                                           = stream;
            }
            if (tmp_v == nullptr)
            {
                // do not use partition-kv kernel
                params.partition_kv = false;

                if (enable_pdl)
                {
                    LaunchPolicy::launchEx(&config, kernel, params);
                } else
                {
                    void *args[] = {(void *)&params};
                    LaunchPolicy::launch((void *)kernel, nblks, nthrs, args, smem_size, stream);
                }
            } else
            {
                // use partition-kv kernel
                params.partition_kv = true;
                auto o              = params.o;
                auto lse            = params.lse;
                params.o            = tmp_v;
                params.lse          = tmp_s;
                if (enable_pdl)
                {
                    LaunchPolicy::launchEx(&config, kernel, params);
                } else
                {
                    void *args[] = {(void *)&params};

                    LaunchPolicy::launch((void *)kernel, nblks, nthrs, args, smem_size, stream);
                }
                if constexpr (AttentionVariant::use_softmax)
                {
                    FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                        tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
                        num_qo_heads, HEAD_DIM, enable_pdl, stream));
                } else
                {
                    FLASHINFER_CUDA_CALL(
                        VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                                   nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
                }
            }
        });
    });
    return cudaSuccess;
}
