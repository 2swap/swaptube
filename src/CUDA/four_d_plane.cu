
#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"
#include "four_d_shared.cuh"

__device__ unsigned int four_d_color(Cuda::vec4 a, float brightness, Cuda::vec3 channels) {

    float ax = min(1.0,smallishness(a.y,brightness))*channels.x;
    float ay = min(1.0,smallishness(a.z,brightness))*channels.y;
    float az = min(1.0,smallishness(a.w,brightness))*channels.z;

    return Cuda::OKLABtoRGB(
        min(1.0,(ax+ay+az)*0.5)*255,
        1.0,
        (ax-ay)*0.866,
        (ax+ay)*0.5-az
    );
}


__global__ void four_d_plane_kernel(
    const Cuda::ivec2 wh,
    const Cuda::vec2 lx_ty,
    const Cuda::vec2 rx_by,
    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,

    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,

    const float brightness,
    const Cuda::vec3 channels,

    const float slider,
    const int equation,

    unsigned int internal_color,
    unsigned int* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);
    Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);
    

    const float commute = min(max((float(pixel_x)/float(wh.x)-slider)*40.0,-1.0),1.0);

    Cuda::vec4 four_d_ouput = four_d_function((point_vec.x*x_unit+point_vec.y*y_unit),equation,commute,jj,ijj); 

    colors[pixel_y * wh.x + pixel_x] = four_d_color(four_d_ouput,brightness,channels); 

}

// Host function to launch the kernel
extern "C" void four_d_plane_render(
    const Cuda::ivec2& wh,
    const Cuda::vec2& lx_ty,
    const Cuda::vec2& rx_by,
    Cuda::vec4 x_unit,
    Cuda::vec4 y_unit,

    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,

    const float brightness,
    const Cuda::vec3 channels,

    const float slider,
    const int equation,

    unsigned int internal_color,
    unsigned int* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    four_d_plane_kernel<<<numBlocks, threadsPerBlock>>>(
        wh, lx_ty, rx_by,
        x_unit, y_unit,
        jj,ijj,
        brightness, channels,
        slider, equation,
        internal_color, d_colors
    );
    cudaDeviceSynchronize();
}
