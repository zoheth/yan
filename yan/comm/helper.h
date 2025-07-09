#pragma once

#include "cuda_runtime.h"
#include <iostream>
#include <vector>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                          \
    {                                                                                                  \
        cutlass::Status error = status;                                                                \
        if (error != cutlass::Status::kSuccess)                                                        \
        {                                                                                              \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                                                    \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t  _start;
    cudaEvent_t  _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

template <typename T>
void print_raw_tensor(const T *data, size_t size, size_t width = 8)
{
    const size_t head_rows = 3;
    const size_t tail_rows = 3;

    const size_t head_elements = head_rows * width;
    const size_t tail_elements = tail_rows * width;

    if (size <= head_elements + tail_elements)
    {
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << data[i] << " ";
            if ((i + 1) % width == 0)
            {
                std::cout << std::endl;
            }
        }
        if (size % width != 0)
        {
            std::cout << std::endl;
        }
        return;
    }

    for (size_t i = 0; i < head_elements; ++i)
    {
        std::cout << data[i] << " ";
        if ((i + 1) % width == 0)
        {
            std::cout << std::endl;
        }
    }

    std::cout << "..." << std::endl;

    const size_t middle_rows        = 2;
    const size_t middle_elements    = middle_rows * width;
    const size_t middle_start_index = (size - middle_elements) / 2;

    for (size_t i = 0; i < middle_elements; ++i)
    {
        std::cout << data[middle_start_index + i] << " ";
        if ((i + 1) % width == 0)
        {
            std::cout << std::endl;
        }
    }

    std::cout << "..." << std::endl;

    const size_t tail_start_index = size - tail_elements;
    for (size_t i = 0; i < tail_elements; ++i)
    {
        std::cout << data[tail_start_index + i] << " ";
        if ((i + 1) % width == 0)
        {
            std::cout << std::endl;
        }
    }
}

template <typename T>
void print_raw_tensor(const std::vector<T> &data, size_t width = 8)
{
    print_raw_tensor(data.data(), data.size(), width);
}
