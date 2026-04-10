#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "common_graphics.cuh"
#include "fractal_sdf.cuh"

const float EPSILON = 2e-4f;

__device__ Color getLighting(const Cuda::vec3& pos, const Cuda::vec3& lightPos, const Cuda::vec3& normal, float shadow, float iters, float max_raymarch_iters){
    float light = max(dot(normal, normalize(lightPos - pos)), 0.0);
    light *= max(shadow, 0.1);
    float glow = min(iters / max_raymarch_iters, 1.0);
    return d_OKLABtoRGB(255, min(light + shadow + glow, 1.0), glow * -0.4, light * -0.4);
}

// Gets the distance to the surface based on Signed Distance Function
__device__ float distMap(const Cuda::vec3& pos, int max_mandelbulb_iters){
    float dist = sdf::mandelbulb(pos, max_mandelbulb_iters);
    return dist;
}

// Implements basic raymarch algorithm
__device__ Cuda::vec4 raymarch(const Cuda::vec3& ro, const Cuda::vec3& rd, float maxDist, int max_raymarch_iters, int max_mandelbulb_iters){
    Cuda::vec3 r = ro;
    float d = distMap(r, max_mandelbulb_iters);
    float t = d;
    for(int i = 0; i < max_raymarch_iters; i++){
        r = ro + t * rd;
        d = distMap(r, max_mandelbulb_iters);
        if(d < EPSILON){
            return Cuda::vec4(r.x, r.y, r.z, (float) i);
        }
        t += d;
        if(t >= maxDist){
            return Cuda::vec4(r.x, r.y, r.z, -1.0);
        }
    }
    return Cuda::vec4(r.x, r.y, r.z, max_raymarch_iters);
}

// Raymarches point for purposes of lighting in direction of light, determining if a direct path exists
__device__ float marchLight(const Cuda::vec3& pos, const Cuda::vec3& lightPos, float minStep, int max_raymarch_iters, int max_mandelbulb_iters){
    Cuda::vec3 r = pos;
    const Cuda::vec3 rd = normalize(lightPos - pos);
    float lightDist = length(lightPos - pos);
    float d = minStep;
    float t = d;
    float shadow = 1.0;
    for(int i = 0; i < max_raymarch_iters; i++){
        r = pos + t * rd;
        d = distMap(r, max_mandelbulb_iters);
        if(d < EPSILON){
            return 0.0;
        }
        shadow = min(shadow, d / t);
        t += d;
        if(t >= lightDist){
            return max(shadow, 0.0);
        }
    }
    return 0.0;
}

// Approximates gradient of SDF at point to be normal of surface
__device__ Cuda::vec3 getNormal(const Cuda::vec3& pos, int max_mandelbulb_iters){
    return normalize(Cuda::vec3(
        distMap(pos + Cuda::vec3(EPSILON, 0, 0), max_mandelbulb_iters) - distMap(pos - Cuda::vec3(EPSILON, 0, 0), max_mandelbulb_iters),
        distMap(pos + Cuda::vec3(0, EPSILON, 0), max_mandelbulb_iters) - distMap(pos - Cuda::vec3(0, EPSILON, 0), max_mandelbulb_iters),
        distMap(pos + Cuda::vec3(0, 0, EPSILON), max_mandelbulb_iters) - distMap(pos - Cuda::vec3(0, 0, EPSILON), max_mandelbulb_iters)
    ));
}

__global__ void runRaymarch(const int width, const int height, const Cuda::vec3 pos, const Cuda::quat camera_orientation, float fov, const Cuda::vec3 lightPos, int max_raymarch_iters, int max_mandelbulb_iters, Color* colors){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= width || pixel_y >= height) {return;}

    Cuda::vec3 rd = Cuda::get_raymarch_vector(pixel_x, pixel_y, width, height, fov, camera_orientation);

    // Raymarches each point
    const Cuda::vec4 rayEnd = raymarch(pos, rd, 5.0, max_raymarch_iters, max_mandelbulb_iters);

    const Cuda::vec3 end_pos = rayEnd;
    float iters = rayEnd.w;
    
    Color color;
    
    // Determines whether to color pixel
    if(iters >= 0.0){
        float shadow = marchLight(end_pos, lightPos, 8.0 * EPSILON, max_raymarch_iters, max_mandelbulb_iters);
        color = getLighting(end_pos, lightPos, getNormal(end_pos, max_mandelbulb_iters), shadow, iters, max_raymarch_iters);
    }else{
        color = 0xff000000;
    }

    // Writes color to image buffer array
    colors[pixel_y * width + pixel_x] = color;
}

extern "C" void render_raymarch(
    const int width, const int height,
    const Cuda::vec3& pos, const Cuda::quat& camera, float fov,
    const Cuda::vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    Color* colors
){
    Color* d_colors;

    // Allocates device memory for color array (image buffer)
    cudaMalloc(&d_colors, width * height * sizeof(Color)); 

    // Defines thread and block sizes for kernel launch
    dim3 threads(16, 16);
    dim3 block((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    runRaymarch<<<block, threads>>>(width, height, pos, normalize(camera), fov, lightPos, max_raymarch_iters, max_mandelbulb_iters, d_colors);

    cudaMemcpy(colors, d_colors, width * height * sizeof(Color), cudaMemcpyDeviceToHost);

    cudaFree(d_colors);
}
