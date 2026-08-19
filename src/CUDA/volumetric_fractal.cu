#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "common_graphics.cuh"
#include "complex_functions.cuh"
#include <stdio.h>

const float EPSILON = 2e-4f;
const float MAXITERS = 50;

__device__ uint32_t getColor(){
    return 0xffffffff;
}

__device__ Cuda::vec3 argb_to_vec3(uint32_t argb){
    return Cuda::vec3(((argb & 0x00ff0000) >> 16) / 255.0, ((argb & 0x0000ff00) >> 8) / 255.0, ((argb & 0x000000ff)) / 255.0);
}

__device__ __forceinline__ uint32_t vec3_to_argb(float alpha, const Cuda::vec3& rgb){
    return ((int) (alpha * 255.0) << 24) | ((int) (rgb.x * 255.0) << 16) | ((int) (rgb.y * 255.0) << 8) | ((int) (rgb.z * 255.0));
}

__device__ float iteration_band_normalize(float iters, float power, float bailout_radius_sq, float focus_n, float focus_spread){
    float x = powf(bailout_radius_sq, powf(power, -iters - 1));
    return /*exp2f(-((iters - focus_n) * (iters - focus_n)) / focus_spread)*/ 1 / (x * x - x);
}

// Implements basic raymarch algorithm
__device__ uint32_t sumRay(const Cuda::vec3& ro, const Cuda::vec3& rd, float minDist, float maxDist, float divergence, float stepMult, float dropoff, float opacityMult, float minOpacity, float p){
    Cuda::vec3 r = ro;

    // Between 0 and 1
    Cuda::vec3 color_accum(0, 0, 0);
    float sq_radius = 0;

    float opacity = 0;
    Cuda::vec3 z(0, 0, 0);

    float iters = 0;
    float weight = 0;

    //divergence = 0.01;

    float step = divergence * minDist * stepMult;

    for(float t = minDist; t <= maxDist; t += step){
        step = divergence * t * stepMult;
        
        r = ro + t * rd;

        //cuComplex c = make_cuComplex(r.x, r.y);
        //cuComplex z = make_cuComplex(r.z, 0);
        
        //iters = mandelbrot_iterations(0, 0, r.z + 2, r.x, r.y, 50, 65536, sq_radius, 0b11111111);
        //iters = mandelbrot_iterations_2(r.x, r.y, 0.75 * cosf(M_PI * r.z) - 0.25, 0.5 * sinf(M_PI * r.z), MAXITERS, 4, sq_radius, 0, 0);
        //iters = mandelbrot_iterations_2(r.z, 0, r.x, r.y, MAXITERS, 4, sq_radius, 0, 0);
        iters = mandelbulb_iterations(z, p, r, MAXITERS, 256, sq_radius);
        //iters = jacobibrot2_iterations(Cuda::vec3(0, 0, 0), r, MAXITERS, 4, sq_radius);

        if(iters == MAXITERS || opacity >= 1.0){
            opacity = 1.0;
            break;
        }/*else{
            float log_zn = log(sq_radius)/2;
            float nu = log(log_zn / log(2)) / log(2);
            iters += (1-nu); // Do not use gradient for exponential parameterization
        }*/

        float c = iters / MAXITERS;

        weight = fmaxf( /*(1 - opacity) */ (fminf(iteration_band_normalize(iters, p, 256, 20, 5), 20 / step) * step * opacityMult * 0.002), minOpacity);
        
        color_accum += weight * argb_to_vec3(Cuda::rainbow(c));//argb_to_vec3(0xffffffff);

        opacity += weight;        
    }

    opacity = fmaxf(fminf(opacity, 1.0), 0.0);
    color_accum = clamp(color_accum, Cuda::vec3(0, 0, 0), Cuda::vec3(1, 1, 1));

    return vec3_to_argb(1.0, color_accum);
}

__global__ void volumeRay(
    const Cuda::ivec2 wh, 
    const Cuda::vec3 pos, const Cuda::quat camera_orientation, float fov, 
    const Cuda::vec3 min_corner, const Cuda::vec3 max_corner,
    const float min_dist, const float max_dist,
    const Cuda::vec3 lightPos, int max_raymarch_iters, int max_mandelbulb_iters, 
    float p, 
    uint32_t* colors
){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    Cuda::ivec2 pixel(pixel_x, pixel_y);

    if (pixel_x >= wh.x || pixel_y >= wh.y) {return;}

    Cuda::vec3 rd = (Cuda::get_raymarch_vector(pixel, wh, fov, camera_orientation));

    Cuda::vec3 low_intersections = (min_corner - pos) / rd;
    Cuda::vec3 high_intersections = (max_corner - pos) / rd;

    Cuda::vec3 close_intersection = Cuda::vec3(fminf(low_intersections.x, high_intersections.x), fminf(low_intersections.y, high_intersections.y), fminf(low_intersections.z, high_intersections.z));
    Cuda::vec3 far_intersection = Cuda::vec3(fmaxf(low_intersections.x, high_intersections.x), fmaxf(low_intersections.y, high_intersections.y), fmaxf(low_intersections.z, high_intersections.z));

    float close_dist = fmaxf(fmaxf(close_intersection.x, close_intersection.y), close_intersection.z);
    float far_dist = fminf(fminf(far_intersection.x, far_intersection.y), far_intersection.z);

    close_dist = fmaxf(close_dist, min_dist);
    far_dist = fminf(far_dist, max_dist);

    // Raymarches each point
    const uint32_t color = sumRay(pos, rd, close_dist, far_dist, 1 / (sqrtf(wh.x * wh.y) * fov), 0.5, 0.92, 1.0, 0, p);

    // Writes color to image buffer array
    colors[pixel_y * wh.x + pixel_x] = color;
}

extern "C" void render_volume(
    const Cuda::ivec2& wh,
    const Cuda::vec3& pos, const Cuda::quat& camera, float fov,
    const Cuda::vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const float p,
    uint32_t* colors
){
    // Defines thread and block sizes for kernel launch
    dim3 threads(16, 16);
    dim3 block((wh.x + threads.x - 1) / threads.x, (wh.y + threads.y - 1) / threads.y);

    volumeRay<<<block, threads>>>(wh, pos, normalize(camera), fov, Cuda::vec3(-2, -2, -2), Cuda::vec3(2, 2, 2), 0.002, 8, lightPos, max_raymarch_iters, max_mandelbulb_iters, p, colors);
}