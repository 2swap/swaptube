#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"



__global__ void three_d_raymarch_kernel(
    const Cuda::ivec2 wh,

    Cuda::quat camera_orientation,
    Cuda::vec3 camera_position,
    float fov, float max_dist,

    const float brightness,
    uint32_t* colors
) {

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);


    Cuda::vec3 out(0.0,0.0,0.0);
    float dist_traveled = 0.0f;
    float dt = 0.01f;
    
    Cuda::vec3 dir_world = normalize(Cuda::get_raymarch_vector(pixel, wh, fov, camera_orientation))*dt;
    Cuda::vec3 current_position = camera_position + dir_world;

    while (dist_traveled < max_dist) {
        dist_traveled += dt;
        current_position += dir_world;

        float max_position = max(abs(current_position.x),max(abs(current_position.y),abs(current_position.z)));

        if (max_position > 10){
            continue;
        }
        int x_small = (abs(current_position.x - round(current_position.x)) < 0.01) ? 1 : 0;
        int y_small = (abs(current_position.y - round(current_position.y)) < 0.01) ? 1 : 0;
        int z_small = (abs(current_position.z - round(current_position.z)) < 0.01) ? 1 : 0;
        
        if ((x_small + y_small + z_small) > 1){
            colors[pixel_y * wh.x + pixel_x] = 0xffffffff; 
            break;
        }
    }

}

// Host function to launch the kernel
extern "C" void three_d_render(
    const Cuda::ivec2& wh,

    const Cuda::quat& camera_orientation, 
    const Cuda::vec3& camera_position,
    float fov_rad, 
    float max_dist,

    const float brightness,
    uint32_t* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    three_d_raymarch_kernel<<<numBlocks, threadsPerBlock>>>(
        wh,
        camera_orientation, camera_position,
        fov_rad, max_dist,
 
        brightness, 
        d_colors
    );
    cudaDeviceSynchronize();
}
