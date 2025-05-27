cmake_minimum_required(VERSION 3.23.1)
project(yan-c LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(yan yan/csrc/main.cu)

target_include_directories(yan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/yan/include
    /home/x/miniconda3/envs/dl/lib/python3.11/site-packages/torch/include
)