#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA           = half_t;                    // Element type for A matrix operand
using LayoutA            = cutlass::layout::RowMajor; // Layout type for A matrix operand
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix configuration
using ElementB           = half_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// C/D matrix configuration
using ElementC           = float;
using LayoutC            = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Kernel functional config
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256, _128, _64>;
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_2, _2, _1>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
    ElementAccumulator, ElementC, LayoutC, AlignmentC, ElementC,
    LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
    LayoutB, AlignmentB, ElementAccumulator, MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using CollectiveAllReduce = cutlass::comm::collective::CollectiveAllReduceMulticastWarpSpecialized<
    ElementC, MmaTileShape_MNK, typename CollectiveEpilogue::StrideD>;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::Sm100GemmARUniversal<
    Shape<int, int, int, int>, // Indicates ProblemShape
    CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler, CollectiveAllReduce>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference =
    cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                     ElementC, LayoutC,
                                     ElementAccumulator, ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA  stride_A;
StrideB  stride_B;
StrideC  stride_C;
StrideD  stride_D;
uint64_t seed = 1;

cutlass::DeviceAllocation<typename Gemm::ElementA>                block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB>                block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC>                block_C;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput>
                                                                          block_D_red;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput>         block_ref_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D_red;

__global__ void ref_reduce_kernel(ElementC *out, ElementC **ref_D_ptr,
                                  ElementC *arrD_red,
                                  ElementC *arrD, size_t npes, size_t nelem)
{
    int                tid    = threadIdx.x + blockIdx.x * blockDim.x;
    volatile ElementC *output = out;
    volatile ElementC *val_ptr;
    for (int i = tid; i < nelem; i += gridDim.x * blockDim.x)
    {
        val_ptr   = ref_D_ptr[0] + i;
        output[i] = *(val_ptr);
        for (int n = 1; n < npes; ++n)
        {
            val_ptr = ref_D_ptr[n] + i;
            output[i] += *(val_ptr);
        }
    }
}

__global__ void compare_kernel(ElementC *expected_out, ElementC *actual_out,
                               ElementC **ref_D_ptr,
                               ElementC *arrD_red, ElementC *arrD, int mype,
                               size_t npes,
                               size_t nelem)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < nelem; i += gridDim.x * blockDim.x)
    {
        if (actual_out[i] != expected_out[i])
        {
            printf("%d elem: %d, mismatch expected_out: %f, actual: %f
                   computed
                   : % f
                   : % f \n ", mype, i, expected_out[i],
                         actual_out[i],
                     *(ref_D_ptr[0] + i), *(ref_D_ptr[1] + i));
        }
    }
}

//////  nvshmem variables //////
nvshmem_team_t *teams_dev, *teams;
int             num_teams;
int             mype, npes;

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
    initialize(options); // Initializes the inputs

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    // Create a structure of gemm kernel arguments
    auto arguments = args_from_options(options);

    auto grid       = gemm.get_grid_shape(arguments);
    dim3 blockShape = GemmKernel::get_block_shape();

    int sm_count;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                                      cudaDevAttrMultiProcessorCount, 0));

    int max_active_blocks = gemm.maximum_active_blocks();
    printf("%d Grid dimension: (%d, %d, %d), block: (%d, %d, %d),
           occupancy
           : % d\n ", mype, grid.x, grid.y, grid.z,
                 blockShape.x,
             blockShape.y, blockShape.z, sm_count);

    int max_concurrent_blocks = sm_count * max_active_blocks;
    if (max_concurrent_blocks < (grid.x * grid.y * grid.z))
    {
        fprintf(stderr,
                "Grid size exceeds maximum concurrent blocks. Using Tile-granular "
                "APIs requires all thread blocks to be concurrent across PEs\n");
        exit(1);
    }

    // create teams
    // each block has 1 warpgroup acting as epilogue, so num_teams = #blocks
    num_teams = grid.x * grid.y * grid.z;
    teams     = (nvshmem_team_t *)malloc(num_teams * sizeof(nvshmem_team_t));

    for (int i = 0; i < num_teams; ++i)
    {
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes,
                                   nullptr, 0, &teams[i]);
    }
    CUDA_CHECK(cudaMalloc((void **)&teams_dev,
                          num_teams * sizeof(nvshmem_team_t)));
    CUDA_CHECK(cudaMemcpy(teams_dev, teams,
                          num_teams * sizeof(nvshmem_team_t), cudaMemcpyHostToDevice));

    // populate AR arguments
    arguments.allReduceArgs = {block_D.get(), block_D_red.get(), stride_D,
                               nvshmem_my_pe(), nvshmem_n_pes(), teams_dev};

    // Using the arguments, query for extra workspace required
    // for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());

    // Check if output result and reference kernel are equal or not
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    Result result;
    result.passed = verify(options, mype, npes);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed")
              << std::endl;

    if (!result.passed)
    {
        exit(-1);
    }

    // Run profiling loop
    if (options.iterations > 0)
    {
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < options.iterations; ++iter)
        {
            CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
            CUTLASS_CHECK(gemm.run());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();

        // Compute average runtime and GFLOPs.
        float elapsed_ms      = timer.elapsed_millis();
        result.avg_runtime_ms = double(elapsed_ms) /
                                double(options.iterations);
        result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

        std::cout << "  Problem Size: " << options.m << 'x'
                  << options.n << 'x' << options.k << std::endl;
        std::cout << "  Avg runtime: " << result.avg_runtime_ms
                  << " ms" << std::endl;
        std::cout << "  GFLOPS: " << result.gflops << std::endl;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////

template <class ElementT_, class TileShape_, class StrideMNL_>
class CollectiveAllReduceMulticastWarpSpecialized
{
  public:
    using ElementT  = ElementT_;
    using TileShape = TileShape_;
    using StrideMNL = StrideMNL_;

    struct Arguments
    {
        ElementT       *ptr_aux = nullptr; // start pointer of matrix
        ElementT       *out_ptr = nullptr; // start pointer of matrix
        StrideMNL       stride;
        int             rank;
        int             world_size;
        nvshmem_team_t *teams = nullptr;
    };

    struct Params
    {
        ElementT               *ptr_aux = nullptr;
        ElementT               *out_ptr = nullptr;
        StrideMNL               stride;
        int                     rank;
        int                     world_size;
        Layout<Shape<int, int>> tile_layout;
        nvshmem_team_t         *teams = nullptr;
    };

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const &problem_shape,
                            Arguments const    &args)
    {
        // Append 1s until problem shape is rank-4
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L]       = problem_shape_mnkl;

        int m_tiles = ceil_div(M, size<0>(TileShape{}));
        int n_tiles = ceil_div(N, size<1>(TileShape{}));
        //  number of tiles in each dimension
        auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

        return {
            args.ptr_aux,
            args.out_ptr,
            args.stride,
            args.rank,
            args.world_size,
            tile_layout,
            args.teams,
        };
    }

    const Params *params_ptr;

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized() {}

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized(Params const &params) : params_ptr(&params) {}

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void do_allreduce(ProblemShapeMNKL const &problem_shape,
                                     TileCoordMNKL const    &tile_coord)
    {
        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (m >= size<0>(params_ptr->tile_layout.shape()) ||
            n >= size<1>(params_ptr->tile_layout.shape()))
        {
            // early exit if out of bound
            return;
        }

        int tile_index = params_ptr->tile_layout(m, n);
        int tiles_per_rank =
            cute::ceil_div(cute::product(params_ptr->tile_layout.shape()),
                           params_ptr->world_size);

        // only root PE will do reduction for this tile
        // only needed if using two-shot algorithm
        int root = tile_index / tiles_per_rank;

        Tensor mAux     = make_tensor(params_ptr->ptr_aux,
                                      make_layout(make_shape(M, N, L),
                                                  params_ptr->stride)); // (M,N,L)
        Tensor mAux_out = make_tensor(
            params_ptr->out_ptr, make_layout(make_shape(M, N, L),
                                             params_ptr->stride)); // (M,N,L)

        Tensor gAux =
            local_tile(mAux, take<0, 2>(TileShape{}), make_coord(m, n, l));
        Tensor gAux_out =
            local_tile(mAux_out, take<0, 2>(TileShape{}),
                       make_coord(m, n, l));

        // predication tensor
        Tensor coordAux = make_identity_tensor(shape(mAux));
        Tensor pAux     = local_tile(coordAux, take<0, 2>(TileShape{}),
                                     make_coord(m, n, l));

        auto boundary    = nvshmemx::make_shape<int, int>(M, N);
        auto start_coord = nvshmemx::make_shape<int, int>(
            size<0>(pAux(0, 0)), size<1>(pAux(0, 0)));

        // Call AR
        auto tensor_shape  = nvshmemx::make_shape(M, N);
        auto tensor_stride = nvshmemx::make_stride(
            size<0>(params_ptr->stride),
            size<1>(params_ptr->stride));
        nvshmemx::Tensor srcTensor = nvshmemx::Tensor(gAux.data(),
                                                      nvshmemx::make_layout(tensor_shape, tensor_stride));
        nvshmemx::Tensor dstTensor = nvshmemx::Tensor(gAux_out.data(),
                                                      nvshmemx::make_layout(tensor_shape, tensor_stride));

        int blkId = blockIdx.x + gridDim.x * blockIdx.y;

        nvshmemx::tile_sum_allreduce_warpgroup<decltype(srcTensor),
                                               decltype(dstTensor),
                                               decltype(boundary),
                                               nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
            params_ptr->teams[blkId], srcTensor, dstTensor, start_coord,
            boundary, root, 0);
    }

    CUTLASS_DEVICE
    void tile_collective_wait()
    {
        int blkId = blockIdx.x + gridDim.x * blockIdx.y;
        nvshmemx::tile_collective_wait_warpgroup<
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
            params_ptr->teams[blkId], 0);
    }
};

///////////////////////////////////////////////////////////////////

int main(int argc, char const **args)
{
    // initialize nvshmem
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(mype));
    printf(" Executing PE: %d out of %d\n", mype, npes);

    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
    // and must have compute capability at least 100a.

    if (__CUDACC_VER_MAJOR__ < 12 ||
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8))
    {
        std::cerr << "This example requires CUDA 12.8 or newer."
                  << std::endl;
        // Returning zero so this test passes on older Toolkits.
        // Its actions are no-op.
        return 0;
    }

    cudaDeviceProp props;
    int            current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (props.major != 10 || props.minor != 0)
    {
        std::cerr
            << "This example requires a GPU with compute capability 100a)."
            << std::endl;
        return 0;
    }

    //
    // Parse options
    //

    Options options;

    options.parse(argc, args);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    //
    // Evaluate CUTLASS kernels
    //

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

    nvshmem_barrier_all();

    for (int i = 0; i < num_teams; ++i)
    {
        nvshmem_team_destroy(teams[i]);
    }
    nvshmem_barrier_all();

    block_D.free();
    block_D_red.free();
    block_ref_D.free();
    free(teams);
    CUDA_CHECK(cudaFree(teams_dev));
    nvshmem_finalize();
    return 0;
}
