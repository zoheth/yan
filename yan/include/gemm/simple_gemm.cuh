__global__ void scaleKernel(const float* input, float* output, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * scale;
    }
}