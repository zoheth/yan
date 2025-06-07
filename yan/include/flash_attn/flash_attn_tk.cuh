#include "tk/kittens.cuh"
#include <cmath>
#include <cstdio>

#define KITTENS_4090
#define KITTENS_TIMINGS

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3;

template <int D>
constexpr size_t ROWS = 16 * (128 / D); // height of each worker tile (rows)

template <int D, typename T = bf16, typename L = row_l>
using qkvo_tile = rt<T, ROWS<D>, D, L>;

template <int D, typename T = float>
using attn_tile = rt<T, ROWS<D>, ROWS<D>>;

template <int D>
using shared_tile = st_bf<ROWS<D>, D>;

template <int D>
using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel

template <int D>
struct Globals
{
    global_layout<D> Qg, Kg, Vg, Og;
#ifdef KITTENS_TIMINGS
    gl<int, 1, -1, -1, 64> timings;
#endif
};

__device__ int get_smid(void)
{
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

template <int D>
__launch_bounds__(NUM_WORKERS *WARP_THREADS, 1)
    __global__ void attend_ker(const __grid_constant__ Globals<D> g)
{

    using load_group     = kittens::group<2>;                                   // pairs of workers collaboratively load k, v tiles
    int           loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int     batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;

#ifdef KITTENS_TIMINGS
    int smid = get_smid();
    if (group<4>::laneid() == 0)
    {
        g.timings[coord<>{smid, 0, 0}] = clock64();
    }
#endif

    extern __shared__ alignment_dummy __shm[];
    shared_allocator                  al((int *)&__shm[0]);

    shared_tile<D>(&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D>(&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();

    shared_tile<D>(&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16>                    q_reg, k_reg;                    // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l>             v_reg;                           // V is column layout, as we use mma_AB.
    qkvo_tile<D, float>                   o_reg;                           // Output tile.
    attn_tile<D, float>                   att_block;                       // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16>                    att_block_mma;                   // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq * ROWS<D> < g.Qg.depth())
    {
        load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0}); // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

#ifdef KITTENS_TIMINGS
    if (group<4>::laneid() == 0)
    {
        g.timings[coord<>{smid, 0, 1}] = clock64();
    }
#endif

    if constexpr (D == 64)
        q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr (D == 128)
        q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec  = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg    = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS * ROWS<D> - 1) / (LOAD_BLOCKS * ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});

#ifdef KITTENS_TIMINGS
    if (group<4>::laneid() == 0)
    {
        g.timings[coord<>{smid, 0, 2}] = clock64();
    }
#endif

    // iterate over k, v for these q's that have been loaded
    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic = (tic + 1) % 3)
    {
        int next_load_idx = (kv_idx + 1) * LOAD_BLOCKS + loadid;
        if (next_load_idx * ROWS<D> < g.Kg.depth())
        {
            int next_tic = (tic + 1) % 3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        } else
            load_async_wait();
        __syncthreads();

#pragma unroll LOAD_BLOCKS
        for (int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx * LOAD_BLOCKS + subtile) * ROWS<D> < g.Kg.depth(); subtile++)
        {
            load(k_reg, k_smem[subtile][tic]);                                   // load k from shared into registers
            att_block = 0.f;                                                     // zero 16x16 attention tile
            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
            int first_index = (kv_idx * LOAD_BLOCKS + subtile) * ROWS<D>;        // one past the last KV index of this tile
            int start_fill  = g.Kg.depth() - first_index < ROWS<D> ? g.Kg.depth() - first_index : ROWS<D>;
            right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());
            max_vec_last = max_vec;
            max_vec      = max<axis::COL>(att_block, max_vec);
            att_block    = exp2(att_block - max_vec);
            max_vec_last = exp2(max_vec_last - max_vec);
            norm_vec *= max_vec_last;
            norm_vec      = sum<axis::COL>(att_block, norm_vec);
            att_block_mma = att_block; // copy to bf16 tile
            load(v_reg, v_smem[subtile][tic]);
            o_reg *= max_vec_last;
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

#ifdef KITTENS_TIMINGS
    if (group<4>::laneid() == 0)
    {
        g.timings[coord<>{smid, 0, 3}] = clock64();
    }
#endif

    o_reg /= norm_vec;
    __syncthreads();
    if (q_seq * ROWS<D> < g.Og.depth())
    {                                    // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }

#ifdef KITTENS_TIMINGS
    if (group<4>::laneid() == 0)
    {
        g.timings[coord<>{smid, 0, 4}] = clock64();
    }
#endif
}

template <int HEAD_DIM>
void flash_attn_func(bf16 *query, bf16 *key, bf16 *value, bf16 *output, int batch_size, int num_heads, int seq_len, cudaStream_t stream, int* timings)
{
    global_layout<HEAD_DIM> Qg(query, batch_size, seq_len, num_heads, nullptr);
    global_layout<HEAD_DIM> Kg(key, batch_size, seq_len, num_heads, nullptr);
    global_layout<HEAD_DIM> Vg(value, batch_size, seq_len, num_heads, nullptr);
    global_layout<HEAD_DIM> Og(output, batch_size, seq_len, num_heads, nullptr);
    gl<int, 1, -1, -1, 64> timings_gl(timings, 1, 128, 1, nullptr);
    Globals<HEAD_DIM> globals{Qg, Kg, Vg, Og};

    dim3 grid(seq_len / (qkvo_tile<HEAD_DIM>::rows * NUM_WORKERS), num_heads, batch_size);

    cudaFuncSetAttribute(attend_ker<HEAD_DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000 / 2);
    attend_ker<HEAD_DIM><<<grid, 32 * NUM_WORKERS, 100000 / 2, stream>>>(globals);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaPeekAtLastError())));
    }
    cudaDeviceSynchronize();
}
