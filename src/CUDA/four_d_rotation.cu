#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"
#include "four_d_shared.cuh"
#include "color.cuh"


__global__ void four_d_rotation_raymarch(
    const Cuda::ivec2 wh,

    Cuda::quat camera_orientation,
    Cuda::vec3 camera_position,
    float fov, float max_dist,

    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,
    const Cuda::vec4 z_unit,
    const Cuda::vec4 rotater,
    const Cuda::vec4 rotaterInv,
    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,
    const float slider,
    const float conjugate,
    const float plane,
    uint32_t* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);

    float dist_traveled = 0.0f;
    // float dt = 0.02f;
    float dt = 0.002f;
    
    Cuda::vec3 dir_world = normalize(Cuda::get_raymarch_vector(pixel, wh, fov, camera_orientation))*dt;
    Cuda::vec3 current_position = camera_position + dir_world;

    const float commute = min(max((float(pixel_x)/float(wh.x)-slider)*40.0,-1.0),1.0);

    Cuda::vec4 last_position(0,0,0,0);
    float last_dot = -current_position.x*0.66666 - current_position.y*0.6666 + current_position.z*0.33333;
    bool hit_plane = false;

    int pixel_index = pixel_y * wh.x + pixel_x;
    colors[pixel_index] = 0x00000000; 
    while (dist_traveled < max_dist) {
        dist_traveled += dt;
        current_position += dir_world;

        Cuda::vec4 pos_rotated = four_d_mult(
            x_unit*current_position.x+y_unit*current_position.y + z_unit*current_position.z,
            rotater,commute,jj,ijj
        );

        if (conjugate != 0){
            pos_rotated =  four_d_mult(rotaterInv,pos_rotated,commute,jj,ijj);
        }


        if (abs(pos_rotated.y) < 1 && abs(pos_rotated.z) < 1 && abs(pos_rotated.w) < 1 && abs(pos_rotated.x) < 1){
            if (abs(last_position.y) > 1){
                if (last_position.y > 0){
                    colors[pixel_index] = 0xffaa0000; 
                    break;
                }
                colors[pixel_index] = 0xffaa7700; 
                break;
            } else if (abs(last_position.z) > 1){
                if (last_position.z > 0){
                    colors[pixel_index] = 0xffcccc00; 
                    break;
                }
                colors[pixel_index] = 0xffcccccc; 
                break;
            } else if (abs(last_position.w) > 1){
                if (last_position.w > 0){
                    colors[pixel_index] = 0xff00cc55; 
                    break;
                }
                colors[pixel_index] = 0xff5533dd; 
                break;
            }
            colors[pixel_index] = 0xffee55cc; 
            break;
        }
        
        last_position = pos_rotated;

        if (hit_plane || plane==0) continue;

        float this_dot = -current_position.x*0.66666 - current_position.y*0.6666 + current_position.z*0.33333;
        if (this_dot == 0 || last_dot / this_dot < 0){
            float a_dot = -current_position.x*0.7071 + current_position.y*0.7071;
            float b_dot = -current_position.x*0.2357 - current_position.y*0.2357 - current_position.z*0.9428;

            if (abs(a_dot) < 0.35 && abs(b_dot) < 0.35){
                hit_plane = true;
            }
        }
        last_dot = this_dot;
    }

    if (hit_plane){
        colors[pixel_index] = Cuda::colorlerp(colors[pixel_index],0xffbbddff,0.5*plane);
    }

}

// Host function to launch the kernel
extern "C" void four_d_rotation_render(
    const Cuda::ivec2& wh,

    const Cuda::quat& camera_orientation, 
    const Cuda::vec3& camera_position,
    float fov_rad, 
    float max_dist,

    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,
    const Cuda::vec4 z_unit,
    const Cuda::vec4 rotater,
    const Cuda::vec4 rotaterInv,

    const Cuda::vec4 jj,
    const Cuda::vec4 ijj,

    const float slider,
    const float conjugate,
    const float plane,

    uint32_t* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    four_d_rotation_raymarch<<<numBlocks, threadsPerBlock>>>(
        wh,
        camera_orientation, camera_position,
        fov_rad, max_dist,
        x_unit, y_unit,z_unit,

        rotater,
        rotaterInv,
        jj,
        ijj,
        slider,
        conjugate,
        plane,
        d_colors
    );
    cudaDeviceSynchronize();
}
