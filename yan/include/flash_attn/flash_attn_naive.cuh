#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

// #define CUTE_DEBUG

template <class Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout layout)
{
    using namespace cute;
    auto l = logical_divide(layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

template <int d, class CtaTiler,
          class SmemLayoutQ, class SmemLayoutKV, class SmemLayoutVtransposed,
          class TiledMma,
          class TiledCopy_g2s, class TiledCopyQ_s2r, class TiledCopyK_s2r, class TiledCopyVT_s2r>
__global__ void flash_attn_cute(CtaTiler cta_tiler, half *Q, half *K, half *V, half *O,
                                int seq_len, TiledCopy_g2s copy_g2s, TiledCopyQ_s2r copyQ_s2r,
                                TiledCopyK_s2r copyK_s2r, TiledCopyVT_s2r copyVT_s2r,
                                TiledMma tiled_mma)
{
    using namespace cute;
    int offset = blockIdx.x * seq_len * d;

    Tensor mQ = make_tensor(make_gmem_ptr(Q + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(K + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(V + offset), make_shape(seq_len, d), make_stride(d, _1{}));
    Tensor mO = make_tensor(make_gmem_ptr(O + offset), make_shape(seq_len, d), make_stride(d, _1{}));

    Tensor gQ = local_tile(mQ, select<0, 2>(cta_tiler), make_coord(_, 0)); // (Br, d, j)
    Tensor gK = local_tile(mK, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, i)
    Tensor gV = local_tile(mV, select<1, 2>(cta_tiler), make_coord(_, 0)); // (Bc, d, i)

#ifdef CUTE_DEBUG
    if (thread0())
    {
        print("\n\ngQ: ");
        print(gQ);
        print("\ngK: ");
        print(gK);
    }
#endif
    static_assert(is_static<SmemLayoutQ>::value);
    static_assert(is_static<SmemLayoutKV>::value);

    __shared__ half smemQ[cosize_v<SmemLayoutQ>];
    __shared__ half smemK[cosize_v<SmemLayoutKV>];
    __shared__ half smemV[cosize_v<SmemLayoutKV>];
    Tensor          sQ  = make_tensor(make_smem_ptr(smemQ), SmemLayoutQ{});
    Tensor          sK  = make_tensor(make_smem_ptr(smemK), SmemLayoutKV{});
    Tensor          sV  = make_tensor(make_smem_ptr(smemV), SmemLayoutKV{});
    Tensor          sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});

    ThrCopy thr_copy_q = copy_g2s.get_slice(threadIdx.x);
    Tensor  tQgQ       = thr_copy_q.partition_S(gQ); //
    Tensor  tQsQ       = thr_copy_q.partition_D(sQ);

#ifdef CUTE_DEBUG
    if (thread0())
    {
        print("\n\ntQgQ: ");
        print(tQgQ);
        print("\ntQsQ: ");
        print(tQsQ);
    }
#endif

    ThrCopy thr_copy_k = copy_g2s.get_slice(threadIdx.x);
    Tensor  tKgK       = thr_copy_k.partition_S(gK);
    Tensor  tKsK       = thr_copy_k.partition_D(sK);

    ThrCopy thr_copy_v = copy_g2s.get_slice(threadIdx.x);
    Tensor  tVgV       = thr_copy_v.partition_S(gV);
    Tensor  tVsV       = thr_copy_v.partition_D(sV);

#ifdef CUTE_DEBUG
    if (thread0())
    {
        print("\n\ntKgK: ");
        print(tKgK);
        print("\ntKsK: ");
        print(tKsK);
    }
#endif
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor tPrQ  = thr_mma.partition_fragment_A(gQ(_, _, 0));
    Tensor tPrK  = thr_mma.partition_fragment_B(gK(_, _, 0));
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt(_, _, 0));

    ThrCopy thr_copy_q_s2r = copyQ_s2r.get_slice(threadIdx.x);
    Tensor  tPsQ           = thr_copy_q_s2r.partition_S(sQ);
    Tensor  tPrQ_view      = thr_copy_q_s2r.retile_D(tPrQ);

    ThrCopy thr_copy_k_s2r = copyK_s2r.get_slice(threadIdx.x);
    Tensor  tPsK           = thr_copy_k_s2r.partition_S(sK);
    Tensor  tPrK_view      = thr_copy_k_s2r.retile_D(tPrK);

    ThrCopy thr_copy_vt_s2r = copyVT_s2r.get_thread_slice(threadIdx.x);
    Tensor  tOsVt           = thr_copy_vt_s2r.partition_S(sVt);
    Tensor  tOrVt_view      = thr_copy_vt_s2r.retile_D(tOrVt);

    auto K_BLOCK_MAX = size<2>(tPrQ);

#ifdef CUTE_DEBUG
    if (thread0())
    {
        print("\n\ntCsQ: ");
        print(tCsQ);
        print("\ntCsK: ");
        print(tCsK);

        print("\n\ntCrQ: ");
        print(tCrQ);
        print("\ntCrK: ");
        print(tCrK);
    }
#endif

    for (int i = 0; i < size<2>(gK); i++)
    {
        copy(copy_g2s, tKgK(_, _, _, i), tKsK(_, _, _, 0));
        copy(copy_g2s, tVgV(_, _, _, i), tVsV(_, _, _, 0));
        cp_async_fence();
        cp_async_wait<0>();

        Tensor acc_o = partition_fragment_C(tiled_mma, select<0, 2>(cta_tiler));
        clear(acc_o);
        for (int j = 0; j < size<2>(gQ); j++)
        {
            // Tensor rP = make_tensor<half>(select<0, 1>(cta_tiler));

            Tensor tPrP = partition_fragment_C(tiled_mma, select<0, 1>(cta_tiler));
            // Tensor tOrP = thr_mma.partition_fragment_C(rP);

            clear(tPrP);

            copy(copy_g2s, tQgQ(_, _, _, j), tQsQ(_, _, _, 0));
            cp_async_fence();
            cp_async_wait<0>();

            __syncthreads();

            if (i == 0 && j == 0 && thread0())
            {
                print("\n\ntPrQ: ");
                print(tPrQ);

                print("\n\ntPrK: ");
                print(tPrK);

                print("\n\tPrP: ");
                print(tPrP);
            }

            for (int k = 0; k < K_BLOCK_MAX; k++)
            {
                copy(copyQ_s2r, tPsQ(_, _, k, 0), tPrQ_view(_, _, k));
                copy(copyK_s2r, tPsK(_, _, k, 0), tPrK_view(_, _, k));

                gemm(tiled_mma, tPrP, tPrQ(_, _, k), tPrK(_, _, k), tPrP);
            }

            __syncthreads();

            Tensor tOrP = make_tensor(tPrP.data(), convert_layout_acc_Aregs(tPrP.layout()));
            // Tensor tPrP = make_tensor<half>(partition_shape_A(tiled_mma, select<0, 1>(cta_tiler)));
            // Tensor tPrP = thr_mma.partition_fragment_A(rP);
            // copy(tOrP, tPrP);
            // clear(tOrO);

            if (i == 0 && j == 0 && thread0())
            {
                print("\n\ntOrP: ");
                print(tOrP);

                print("\n\ntOrVt: ");
                print(tOrVt);

                print("\n\nacc_o: ");
                print(acc_o);

                print("\n\ntOsVt(_, _, k, 0): ");
                print(tOsVt(_, _, 0, 0));
                print("\n\ntOrVt_view(_, _, k): ");
                print(tOrVt_view(_, _, 0));
                print("\n\n");
            }

            auto K_BLOCK_PV = size<2>(tOrP);
            for (int k = 0; k < K_BLOCK_PV; k++)
            {
                copy(copyVT_s2r, tOsVt(_, _, k, 0), tOrVt_view(_, _, k));
                gemm(tiled_mma, acc_o, tOrP(_, _, k), tOrVt(_, _, k), acc_o);
            }

            __syncthreads();
        }
        Tensor gO   = local_tile(mO, select<0, 2>(cta_tiler), make_coord(i, 0));
        Tensor tOgO = thr_mma.partition_C(gO);

        if(thread0()&&i==0)
        {
            print("\n\nacc_o: ");
            print(acc_o);
            print("\n\ntOgO: ");
            print(tOgO);
            print("\n\n");
            tOgO(0) = 106;
        }

        for(int idx = 0; idx < size(acc_o); idx++)
        {
            //tOgO(idx) =  __hadd(tOgO(idx), tOrO(idx));
            // tOgO(idx) = acc_o(idx);
        }
    }
}

template <int d>
void flash_attn_func(half *query, half *key, half *value, half *output, int batch_size, int num_heads, int seq_len, cudaStream_t stream)
{
    // key: [batch_size, num_heads, seq_len, d]
    // value: [batch_size, num_heads, seq_len, d]
    // query: [batch_size, num_heads, seq_len, d]
    // output: [batch_size, num_heads, seq_len, d]
    // temp: [batch_size, num_heads, seq_len, seq_len]
    using namespace cute;

    auto prob_shape = make_shape(batch_size, num_heads, seq_len, d);

    auto Br       = Int<64>{};
    auto Bc       = Int<64>{};
    auto kHeadDim = Int<d>{};

    constexpr int kBlockHeadDim = 64;
    static_assert(d % kBlockHeadDim == 0);

    auto kInnerStage = Int<1>{};
    auto kOuterStage = Int<1>{};

    auto cta_tiler = make_shape(Br, Bc, kHeadDim);

    using SmemLayoutAtom = decltype(make_layout(make_shape(Int<8>{}, Int<kBlockHeadDim>{}),
                                                make_stride(Int<kBlockHeadDim>{}, Int<1>{})));

    auto sQ           = tile_to_shape(SmemLayoutAtom{}, make_shape(Br, kHeadDim, kInnerStage));
    auto sK           = tile_to_shape(SmemLayoutAtom{}, make_shape(Bc, kHeadDim, kOuterStage));
    auto sV           = tile_to_shape(SmemLayoutAtom{}, make_shape(Bc, kHeadDim, kOuterStage));
    auto sVtransposed = tile_to_shape(SmemLayoutAtom{}, make_shape(kHeadDim, Bc, kOuterStage));

    TiledMMA  tiled_mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                         Layout<Shape<_4, _1, _1>>{},
                                         Tile<_64, _16, _16>{});
    TiledCopy copyQ_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout
                                          Layout<Shape<_1, _8>>{});                 // Val layout
    TiledCopy copyK_g2s = copyQ_g2s;
    TiledCopy copyV_g2s = copyQ_g2s;

    TiledCopy copyQ_s2r  = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    TiledCopy copyK_s2r  = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    TiledCopy copyVT_s2r = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);

    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(batch_size * num_heads);

    flash_attn_cute<d, decltype(cta_tiler),
                    decltype(sQ), decltype(sK), decltype(sVtransposed),
                    decltype(tiled_mma),
                    decltype(copyQ_g2s), decltype(copyQ_s2r),
                    decltype(copyK_s2r)>
        <<<dimGrid, dimBlock, 0, stream>>>(cta_tiler, query, key, value, output, seq_len, copyQ_g2s, copyQ_s2r, copyK_s2r, copyVT_s2r, tiled_mma);
}