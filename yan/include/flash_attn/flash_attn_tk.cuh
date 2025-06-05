#include "tk/kittens.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>

using namespace kittens;

constexpr int NUM_WORKERS = 4; // This kernel uses 4 worker warps per block, and 2 blocks per SM.

template <int D>
constexpr size_t ROWS = 16 * (128 / D);

template <int D, typename T = bf16, typename L = row_l>
using qkvo_tile = rt<T, ROWS<D>, D, L>;

template <int D, typename T = float>
using attn_tile = rt<T, ROWS<D>, ROWS<D>>;

template <int D>
using shared_tile = st_bf<ROWS<D>, D>;

template <int D>
using global_layout = gl<bf16, -1, -1, -1, D>;

template <int D>
struct Globals
{
    global_layout<D> Qg, Kg, Vg, Og;
};

template <int D>
__launch_bounds__(NUM_WORKERS *WARP_THREADS, 1)
    __global__ void attend_ker(const __grid_constant__ Globals<D> g)
{
    using load_group          = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int           load_id     = load_group::groupid();
    int           worker_id   = kittens::warpid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;

    const int batch = blockIdx.z;
    const int head  = blockIdx.y;
    const int q_seq = blockIdx.x * NUM_WORKERS + worker_id;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator                  al((int *)&__shm[0]);

    shared_tile<D>(&k_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    shared_tile<D>(&v_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();

    shared_tile<D>(&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);

    qkvo_tile<D, bf16>        q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile<D, float>       o_reg;
    attn_tile<D, float>       att_block;
    attn_tile<D, bf16>        att_block_mma;

    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    // each warp loads its own Q tile of 16x64
    if (q_seq * ROWS<D> < g.Qg.rows)
    {
        load(qo_smem[worker_id], g.Qg, {batch, head, q_seq, 0});
        __syncwarp();
        load(q_reg, qo_smem[worker_id]);
    }
    __syncthreads();

    // temperature adjustment. Pre-multiplying by lg2(e), too, so we can use exp2 later.
    if constexpr (D == 64)
        mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089));
    else if constexpr (D == 128)
        mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089));

    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);

    // launch the load of the first k, v tiles
    int kv_blocks = g.Qg.rows / (LOAD_BLOCKS*ROWS<D>);
    int tic = 0;

    load_group::load_async(k_smem[load_id][0], g.Kg, {batch, head, load_id, 0});
    load_group::load_async(v_smem[load_id][0], g.Vg, {batch, head, load_id, 0});

    
}
