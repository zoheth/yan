cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler")

project(test LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(NVSHMEM_PREFIX /usr/lib64/nvshmem/12)
set(BUILD_SHARED_LIBS ON)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/cmake/nvshmem)

# add_executable(nvl_p2p_test nvl_p2p_test.cu)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_FAST_DEBUG -O3 --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

include_directories(${CMAKE_SOURCE_DIR}/../include
${CMAKE_SOURCE_DIR}/../../third-party/cutlass/include
${CMAKE_SOURCE_DIR}/../../third-party/cutlass/tools/util/include)

add_executable(test gemm_allreduce.cu)
set_property(TARGET test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test PRIVATE nvshmem::nvshmem_host)
target_link_libraries(test PRIVATE nvshmem::nvshmem_device)