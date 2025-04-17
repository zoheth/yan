# yan (炎)

## Overview

yan (炎) is a high-performance CUDA operator library designed for learning purposes while emphasizing clean code and maximum performance. Built on DeepGEMM's JIT framework and CuTe implementation, yan delivers efficient operators optimized primarily for RTX 4090 GPUs.

## Features

- **Diverse Operator Suite**: Implementation of multiple high-performance operators including:
  - Reduction
  - Scan (prefix sum)
  - General Matrix Multiplication (GEMM)
  - Online Softmax
  - Flash Attention
  - Custom Triplane Sampling Operator

- **Performance Highlights**:
  - GEMM and Softmax operators achieve 1.5x throughput compared to PyTorch implementations
  - Flash Attention performance reaches 98% of Dao AI Lab's Flash Attention 2 implementation
  - Fused Triplane Sampling operator delivers 3x speed improvement

## Technical Foundation

- **JIT Framework**: Based on [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- **Tensor Engine**: Powered by NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass)
- **Optimization Target**: Primarily optimized for NVIDIA RTX 4090 GPUs

## Benchmarks

| Operator | Performance vs. Baseline |
|----------|--------------------------|
| GEMM | 1.5x vs. PyTorch |
| Softmax | 1.5x vs. PyTorch |
| Flash Attention | 98% of [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) |
| Triplane Sampling | 3x improvement with fusion |

## Project Goals

This project aims to serve as both a learning resource and a high-performance library, demonstrating how clean, well-structured code can achieve exceptional performance for critical deep learning operations.

## Acknowledgments

This project draws inspiration from:
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [cutlass](https://github.com/NVIDIA/cutlass)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)