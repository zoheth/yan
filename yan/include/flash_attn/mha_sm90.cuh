#pragma once
#include <__clang_cuda_builtin_vars.h>
#include <cassert>
#include <cstdint>
#include <cute/tensor.hpp>
#include <sys/types.h>

#define Q_HEADS_PER_CTA 64
#define HEAD_GRP_SIZE 16

struct SpecDecParams
{
    uint32_t qSeqLen;
    uint32_t const* qCuSeqLens; // Cumulative 
    uint32_t const* mask;
};

template <bool usePagedKVCache>
struct KVCacheList;



template <typename T>
__host__ __device__ constexpr inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

__host__ __device__ constexpr inline uint32_t exactDiv(uint32_t a, uint32_t b)
{
    assert(a % b == 0);
    return a / b;
}

constexpr uint32_t ctaNbQHead = Q_HEADS_PER_CTA;
constexpr uint32_t headGrpSize = HEAD_GRP_SIZE;
inline constexpr uint32_t inputTokenPerCta = exactDiv(ctaNbQHead, headGrpSize);



__device__ inline uint32_t getInputTokOffset(SpecDecParams const& params, uint32_t idxReq)
{
    return (params.qCuSeqLens == nullptr) ? params.qSeqLen * idxReq : params.qCuSeqLens[idxReq];
}

__global__ void xqa_kernel_sm90(uint32_t const nbKHeads, 
    uint32_t const batchSize,
    SpecDecParams const specDecParams)
{
    uint32_t const idxReq = blockIdx.z / nbKHeads;

    uint32_t const reqInputTokBeg = getInputTokOffset(specDecParams, idxReq);
    uint32_t const reqInputTokEnd = getInputTokOffset(specDecParams, idxReq+1);
    uint32_t const nbInputSeqSplit = gridDim.x;
    assert(nbInputSeqSplit == divUp(specDecParams.qSeqLen, inputTokenPerCta));

    uint32_t const idxHeadGrp = blockIdx.z % nbKHeads;
    assert(gridDim.z == nbKHeads * batchSize);
    uint32_t const cacheSeqLen_past = 0;

    
}

