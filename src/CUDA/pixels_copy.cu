#include <cuda_runtime.h>
#include <stdint.h>

extern "C" uint32_t* cuda_alloc_pixels_on_device(int size)
{
    uint32_t* d_pixels = nullptr;
    cudaMalloc((void**)&d_pixels, size * sizeof(uint32_t));
    return d_pixels;
}

/*
extern "C" uint32_t* cuda_copy_pixels_to_device(uint32_t* h_pixels, int width, int height)
{
...
}
// This could exist, but I want to avoid ever copying frames to the device to avoid
// promoting rendering on CPU. If you want to use this, think: can I do this all on the GPU without a copy?
*/

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels)
{
    cudaMemcpy(h_pixels, d_pixels, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels)
{
    cudaFree(d_pixels);
}
