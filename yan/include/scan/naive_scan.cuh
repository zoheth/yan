#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#define SECTION_SIZE 1024
#define CFACTOR 4

__device__ void brent_kung_scan_block(float *data, unsigned int tx, unsigned raw_stride)
{
	for (unsigned int stride = 1; stride <= blockDim.x; stride <<= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index < 2 * blockDim.x)
		{
			data[index * raw_stride + raw_stride -1] += data[(index - stride) * raw_stride + raw_stride - 1];
		}
	}

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index + stride < 2 * blockDim.x)
		{
			data[(index + stride) * raw_stride + raw_stride -1] += data[index * raw_stride + raw_stride - 1];
		}
	}
	__syncthreads();
}

__device__ void scan_block(float *data, unsigned int c_factor)
{
	unsigned int tx = threadIdx.x;
	for (unsigned int i = 1; i < c_factor; ++i)
	{
		data[tx * c_factor + i] += data[tx * c_factor + i - 1];
		__syncthreads();
		data[(tx+blockDim.x) * c_factor + i] += data[(tx+blockDim.x) * c_factor + i - 1];
	}
	__syncthreads();

	brent_kung_scan_block(data, tx, c_factor);

	__syncthreads();

	if (tx > 0)
	{
		for (unsigned int i = 0; i < c_factor - 1; ++i)
		{
			data[tx * c_factor + i] += data[(tx - 1) * c_factor + c_factor - 1];
		}
	}
	__syncthreads();
	for (unsigned int i = 0; i < c_factor - 1; ++i)
	{
		data[(tx+blockDim.x) * c_factor + i] += data[((tx+blockDim.x) - 1) * c_factor + c_factor - 1];
	}
}

__global__ void all_scan(float *X, float *Y, unsigned int N)
{
	scan_block(X, N/SECTION_SIZE);
	__syncthreads();
	for (unsigned int j = 0; j < N/SECTION_SIZE*2; ++j)
	{
		if (threadIdx.x + j * blockDim.x < N)
		{
			Y[threadIdx.x + j * blockDim.x] = X[threadIdx.x + j * blockDim.x];
		}
	}
}

__global__ void scan_phase1(float *X, float *Y, unsigned int N, float *block_sums)
{
	__shared__ float XY[SECTION_SIZE * CFACTOR];
	unsigned int    tx = threadIdx.x;
	unsigned int     start_i = blockIdx.x * blockDim.x * CFACTOR * 2 + tx;

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			XY[tx + j * blockDim.x] = X[start_i + j * blockDim.x];
		}
		else
		{
			XY[tx + j * blockDim.x] = 0;
		}
	}

	__syncthreads();

	scan_block(XY, CFACTOR);

	__syncthreads();

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			Y[start_i + j * blockDim.x] = XY[tx + j * blockDim.x];
		}
	}

	__syncthreads();
	if (tx == blockDim.x - 1)
	{
		block_sums[blockIdx.x] = XY[2 * blockDim.x * CFACTOR - 1];
	}

}

__global__ void scan_phase2(unsigned int N, float *block_sums)
{
	scan_block(block_sums, N / SECTION_SIZE);
}

__global__ void scan_phase3(float *Y, unsigned int N, float *block_sums)
{
	unsigned int i = blockIdx.x * blockDim.x * CFACTOR + threadIdx.x;
    
	if (blockIdx.x > 0)
	{
		for (int j = 0; j < CFACTOR; ++j)
		{
			if (i + j * blockDim.x < N)
			{
				Y[i + j * blockDim.x] += block_sums[blockIdx.x - 1];
			}
		}
	}
}


template <int BLOCK_SIZE=1024>
void naive_scan_c(float *X, float *Y, unsigned int N)
{
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = (num_blocks + CFACTOR - 1) / CFACTOR;
    
    float *block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(float));

    scan_phase1<<<num_blocks, BLOCK_SIZE / 2>>>(X, Y, N, block_sums);
    scan_phase2<<<1, BLOCK_SIZE / 2>>>(num_blocks, block_sums);
    scan_phase3<<<num_blocks, BLOCK_SIZE>>>(Y, N, block_sums);
    cudaFree(block_sums);
}