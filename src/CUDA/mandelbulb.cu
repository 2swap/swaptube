#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "common_graphics.cuh"
#include "fractal_sdf.cuh"

const float EPSILON = 1e-4f;

__device__ unsigned int getLighting(const Cuda::vec3& pos, const Cuda::vec3& lightPos, const Cuda::vec3& normal, float shadow, float iters, float max_raymarch_iters){
    float light = fmaxf(dot(normal, normalize(lightPos - pos)), 0.25);
    light *= fmaxf(shadow, 0.25);
    float glow = fminf(iters * 2 / max_raymarch_iters, 1.0);
    return Cuda::OKLABtoRGB(255, fminf(light + shadow + glow, 1.0), glow * -0.4, light * -0.4);
}

// Gets the distance to the surface based on Signed Distance Function
__device__ float distMap(const Cuda::vec3& pos, int max_sdf_iters, const int sdfID, const float sdflerp){
    float dist;
    switch(sdfID){
        case 1:
            //dist = sdf::quatJulia2(Cuda::quat(pos.x, pos.y, pos.z, 0.1), Cuda::quat(-8, 16, 2, 0.81)/22.0, max_sdf_iters);
            //dist = sdf::quatJulia2(Cuda::quat(0, 0, 0, 0), Cuda::quat(0, pos.x, pos.y, pos.z), max_sdf_iters);
            //dist = (sdf::quatJulia2(Cuda::quat(pos.x, pos.y, pos.z, 0.1), Cuda::quat(-2,6,15,-6)/22.0, max_sdf_iters) * (1.0 - sdflerp) + sdf::mandelbulb8(pos, max_sdf_iters) * sdflerp);
            //dist = sdf::mandelbulb(pos, 2.0, max_sdf_iters);
            dist = sdf::mandelbulb8(pos, max_sdf_iters);
            //dist = sdf::burningbulb2(pos, max_sdf_iters);
            //dist = sdf::juliabulb(pos, Cuda::vec3(0.6, 0.5, 0.8), 8.0, max_sdf_iters);
            //dist = sdf::mandeljulia(pos.z, 0, pos.x, pos.y, max_sdf_iters);
            break;
        default:
            dist = 0;
            break;
    }
    return fminf(dist, 0.5f);
}

// Implements basic raymarch algorithm
__device__ Cuda::vec4 raymarch(const Cuda::vec3& ro, const Cuda::vec3& rd, const float maxDist, const int max_raymarch_iters, const int max_mandelbulb_iters, const int sdfID, const float sdflerp){
    Cuda::vec3 r = ro;
    float d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
    float t = d;
    for(int i = 0; i < max_raymarch_iters; i++){
        r = ro + t * rd;
        d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
        if(d < EPSILON){
            return Cuda::vec4(r.x, r.y, r.z, (float) i);
        }
        t += d;
        if(t >= maxDist){
            return Cuda::vec4(r.x, r.y, r.z, -1.0f);
        }
    }
    return Cuda::vec4(r.x, r.y, r.z, max_raymarch_iters);
}

// Raymarches point for purposes of lighting in direction of light, determining if a direct path exists
__device__ float marchLight(const Cuda::vec3& pos, const Cuda::vec3& lightPos, const float minStep, const int max_raymarch_iters, const int max_mandelbulb_iters, const int sdfID, const float sdflerp){
    Cuda::vec3 r = pos;
    const Cuda::vec3 rd = normalize(lightPos - pos);
    float lightDist = length(lightPos - pos);
    float d = minStep;
    float t = d;
    float shadow = 1.0f;
    for(int i = 0; i < max_raymarch_iters; i++){
        r = pos + t * rd;
        d = distMap(r, max_mandelbulb_iters, sdfID, sdflerp);
        if(d < EPSILON){
            return 0.0f;
        }
        shadow = fminf(shadow, d / t);
        t += d;
        if(t >= lightDist){
            return fmaxf(shadow, 0.0f);
        }
    }
    return 0.0f;
}

// Approximates gradient of SDF at point to be normal of surface
__device__ Cuda::vec3 getNormal(const Cuda::vec3& pos, const int max_mandelbulb_iters, const int sdfID, const float sdflerp){
    return normalize(Cuda::vec3(
        distMap(pos + Cuda::vec3(EPSILON, 0, 0), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - Cuda::vec3(EPSILON, 0, 0), max_mandelbulb_iters, sdfID, sdflerp),
        distMap(pos + Cuda::vec3(0, EPSILON, 0), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - Cuda::vec3(0, EPSILON, 0), max_mandelbulb_iters, sdfID, sdflerp),
        distMap(pos + Cuda::vec3(0, 0, EPSILON), max_mandelbulb_iters, sdfID, sdflerp) - distMap(pos - Cuda::vec3(0, 0, EPSILON), max_mandelbulb_iters, sdfID, sdflerp)
    ));
}

__global__ void runRaymarch(
    const Cuda::ivec2 wh,
    const Cuda::vec3 pos, const Cuda::quat camera_orientation, float fov, 
    const Cuda::vec3 lightPos, 
    const int max_raymarch_iters, const int max_mandelbulb_iters, 
    const int sdfID, const float sdflerp,
    uint32_t* colors){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= wh.x || pixel_y >= wh.y) {return;}

    Cuda::vec3 rd = Cuda::get_raymarch_vector(Cuda::ivec2(pixel_x, pixel_y), wh, fov, camera_orientation);

    // Raymarches each point
    const Cuda::vec4 rayEnd = raymarch(pos, rd, 10.0f, max_raymarch_iters, max_mandelbulb_iters, sdfID, sdflerp);

    const Cuda::vec3 end_pos = rayEnd;
    float iters = rayEnd.w;
    
    uint32_t color;
    
    // Determines whether to color pixel
    if(iters >= 0.0f){
        float shadow = marchLight(end_pos, lightPos, 8.0f * EPSILON, max_raymarch_iters / 4.0f, max_mandelbulb_iters, sdfID, sdflerp);
        color = getLighting(end_pos, lightPos, getNormal(end_pos, max_mandelbulb_iters, sdfID, sdflerp), shadow, iters, max_raymarch_iters);
    }else{
        color = 0xff000000;
    }

    // Writes color to image buffer array
    colors[pixel_y * wh.x + pixel_x] = color;
}

extern "C" void render_raymarch(
    const Cuda::ivec2& wh,
    const Cuda::vec3& pos, const Cuda::quat& camera, float fov,
    const Cuda::vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    const int sdfID, const float sdflerp,
    uint32_t* d_colors
){
    // Defines thread and block sizes for kernel launch
    dim3 threads(16, 16);
    dim3 block((wh.x + threads.x - 1) / threads.x, (wh.y + threads.y - 1) / threads.y);

    runRaymarch<<<block, threads>>>(wh, pos, normalize(camera), fov, lightPos, max_raymarch_iters, max_mandelbulb_iters, sdfID, sdflerp, d_colors);
}
