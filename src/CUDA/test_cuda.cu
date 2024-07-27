#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() {
        printf("Hello World from GPU!\n");
}

void run_cuda_test() {
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
}
