#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"
#include "four_d_shared.cuh"




__device__ Cuda::vec3 four_d_accum(Cuda::vec4 a, float brightness) {

    Cuda::vec3 accum(
        //  b, b, b
        smallness(a.y,brightness),
        smallness(a.z,brightness),
        smallness(a.w,brightness)
    );
    return accum;
}

__device__ uint32_t accum_to_color(Cuda::vec3 a, float fade) {

    // return 255 << 24 |
    // (uint32_t) min(a.x,255.0) << 16 |
    // (uint32_t) min(a.y,255.0) << 8 |
    // (uint32_t) min(a.z,255.0);

    float ax = min(1.0,fade*a.x);
    float ay = min(1.0,fade*a.y);
    float az = min(1.0,fade*a.z);

    return Cuda::OKLABtoRGB(
        min(1.0,(ax+ay+az)*0.5)*255,
        1.0,
        (ax-ay)*0.866,
        (ax+ay)*0.5-az
    );
}

__global__ void four_d_raymarch_kernel(
    const Cuda::ivec2 wh,

    Cuda::quat camera_orientation,
    Cuda::vec3 camera_position,
    float fov, float max_dist,

    const Cuda::vec4 x_unit, const Cuda::vec4 y_unit, const Cuda::vec4 z_unit,
    const Cuda::vec4 jj, const Cuda::vec4 ijj,
    const float brightness, const Cuda::vec3 channels,  const float fade,
    const float slider, const float slider_width,
    const int equation, const float equation_lerp,

    uint32_t* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);
    // Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    Cuda::vec3 out(0.0,0.0,0.0);
    float dist_traveled = 0.0f;
    float dt = 0.005f;
    
    Cuda::vec3 dir_world = normalize(Cuda::get_raymarch_vector(pixel, wh, fov, camera_orientation))*dt;

    Cuda::vec3 current_position = camera_position + dir_world;
    // Cuda::vec3 current_position(0,0,-10);

    const float x_dir = float(pixel_x)/float(wh.x);
    // 2*sigmoid(z)-1 == tanh(z/2); tanh avoids exp() overflowing to inf (inf/inf = NaN)
    const float commute = tanhf((x_dir-slider) * slider_width * 0.5f);
    // Cuda::vec4 trans(0.0,1.0,-1.0,2.0);

    while (dist_traveled < max_dist) {
        dist_traveled += dt;
        current_position += dir_world;


        Cuda::vec4 point = x_unit*current_position.x+y_unit*current_position.y + z_unit*current_position.z;
        Cuda::vec4 four_d_output = four_d_function(point, equation, commute, jj, ijj);
        if (equation_lerp > 0){
            four_d_output = veclerp(four_d_output, four_d_function(point, equation+1, commute, jj, ijj), equation_lerp);
        }

        out += four_d_accum(four_d_output,brightness); 
    }

    colors[pixel_y * wh.x + pixel_x] = accum_to_color(out*channels,fade); 

}

// Host function to launch the kernel
extern "C" void four_d_render(
    const Cuda::ivec2& wh,

    const Cuda::quat& camera_orientation, 
    const Cuda::vec3& camera_position,
    float fov_rad, 
    float max_dist,

    const Cuda::vec4 x_unit, const Cuda::vec4 y_unit, const Cuda::vec4 z_unit,
    const Cuda::vec4 jj, const Cuda::vec4 ijj,

    const float brightness, const Cuda::vec3 channels, const float fade,
    const float slider, const float slider_width,
    const int equation, const float equation_lerp,
    uint32_t* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    four_d_raymarch_kernel<<<numBlocks, threadsPerBlock>>>(
        wh,
        camera_orientation, camera_position,
        fov_rad, max_dist,
        x_unit, y_unit, z_unit,
        jj,ijj,
        brightness, channels,fade,
        slider, slider_width,
        equation, equation_lerp,

        d_colors
    );
    cudaDeviceSynchronize();
}
