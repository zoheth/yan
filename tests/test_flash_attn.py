import torch

import yan
from yan import bench_kineto

if __name__ == "__main__":
    yan.jit_kernels.accuracy_test()
    