// Run this with nvcc to confirm that everything is working right

#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Function to test CUDA
bool testCUDA() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // Allocate GPU buffers for three arrays (two input, one output)
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_a!" << std::endl;
        return false;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_b!" << std::endl;
        cudaFree(dev_a); // Free dev_a before returning
        return false;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_c!" << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
        return false;
    }

    // Copy input arrays from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for dev_a!" << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for dev_b!" << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    // Launch the CUDA kernel on the GPU
    addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    // Copy output array from GPU buffer back to host memory
    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for dev_c!" << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    // Verify the results
    std::cout << "Result of addition on GPU: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return true;
}

