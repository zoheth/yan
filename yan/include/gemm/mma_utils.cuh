#pragma once

#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

#include "utils.cuh"
#include <cstdint>
#include <utility>

__forceinline__ __device__ uint32_t get_lane_id()
{
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

template <int N_, typename MMA>
struct FP8MMA
{
    template <size_t... Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const &desc_a, uint64_t const &desc_b, float *d, bool scale_d,
                                                         std::index_sequence<Idx...>)
    {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const &desc_a, uint64_t const &desc_b,
                                                 float *d, bool scale_d)
    {
        call_fma_impl(desc_a, desc_b, d, scale_d, std::make_index_sequence<N_ / 2>{});
    }

	static constexpr int M = 64;
	static constexpr int N = N_;
	static constexpr int K = 32;
	static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {
	static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
}

__device__ __host__ constexpr int
get_num_math_warpgroups(int block_m)
{
    return block_m == 64 ? 1 : 2;
}

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m)
{
    static_assert(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return get_num_math_warpgroups(block_m) * kNumMathThreadsPerGroup + kNumTMAThreads;
}