#include <device_host/nvshmem_common.cuh>
#include <host/nvshmem_api.h>
#include <host/nvshmemx_coll_api.h>
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void simple_shift(int *destination)
{
    int mype = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int peer  = (mype + 1) % n_pes;

    nvshmem_int_p(destination, mype, peer);
}

int main(void)
{
    int mype_node, msg;
    cudaStream_t stream;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    int *destination = (int *)nvshmem_malloc(sizeof(int));
    
    simple_shift<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    return 0;

}