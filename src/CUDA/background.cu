#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"



__global__ void background_kernel(
    const Cuda::ivec2 wh,
    const Cuda::ivec3 bg_0,
    const Cuda::ivec3 bg_1,
    const float slider,
    uint32_t* colors
) {

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);

    const float colorLerp = min(max((float(pixel_x)/float(wh.x)-slider)*40.0,-1.0)*0.5+0.5,1.0);

    colors[pixel_y * wh.x + pixel_x] = 255 << 24
        | ((uint32_t) Cuda::lerp(bg_1.x,bg_0.x,colorLerp)) << 16
        | ((uint32_t) Cuda::lerp(bg_1.y,bg_0.y,colorLerp)) << 8
        | ((uint32_t) Cuda::lerp(bg_1.z,bg_0.z,colorLerp))
    ; 

}

// Host function to launch the kernel
extern "C" void background_render(
    const Cuda::ivec2& wh,
    const Cuda::ivec3 bg_0,
    const Cuda::ivec3 bg_1,
    const float slider,
    uint32_t* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    background_kernel<<<numBlocks, threadsPerBlock>>>(
        wh,
        bg_0, bg_1,
        slider,
        d_colors
    );
    cudaDeviceSynchronize();
}
