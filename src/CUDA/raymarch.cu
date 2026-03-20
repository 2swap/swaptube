#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "fractal_sdf.cuh"

using namespace Cuda;

// NVCC on Windows does not reliably treat namespace-scope const floats as device constants.
#define RAYMARCH_EPSILON 2e-4f

__device__ unsigned int getLighting(vec3 pos, vec3 lightPos, vec3 normal, float shadow, float iters, float maxIters){
    float light = max(dot(normal, normalize(lightPos - pos)), 0.0);
    light *= max(shadow, 0.1);
    float glow = min(iters / min(maxIters, maxIters), 1.0);
    return d_OKLABtoRGB(255, min(light + shadow + glow, 1.0), glow * -0.4, light * -0.4);
}

// Gets the distance to the surface based on Signed Distance Function
__device__ float distMap(vec3 pos){
    float dist = sdf::mandelbulb(pos, 5);
    return dist;
}

// Implements basic raymarch algorithm
__device__ vec4 raymarch(vec3 ro, vec3 rd, float maxDist, int maxIters){
    vec3 r = ro;
    float d = distMap(r);
    float t = d;
    for(int i = 0; i < maxIters; i++){
        r = ro + t * rd;
        d = distMap(r);
        if(d < RAYMARCH_EPSILON){
            return vec4(r.x, r.y, r.z, (float) i);
        }
        t += d;
        if(t >= maxDist){
            return vec4(r.x, r.y, r.z, -1.0);
        }
    }
    return vec4(r.x, r.y, r.z, (float) maxIters); // Returns final position along with iterations for lighting
}

// Raymarches point for purposes of lighting in direction of light, determining if a direct path exists
__device__ float marchLight(vec3 pos, vec3 lightPos, float minStep, int maxIters){
    vec3 r = pos;
    vec3 rd = normalize(lightPos - pos);
    float lightDist = length(lightPos - pos);
    float d = minStep;
    float t = d;
    float shadow = 1.0;
    for(int i = 0; i < maxIters; i++){
        r = pos + t * rd;
        d = distMap(r);
        if(d < RAYMARCH_EPSILON){
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
__device__ vec3 getNormal(vec3 pos){
    return normalize(vec3(
        distMap(pos + vec3(RAYMARCH_EPSILON, 0, 0)) - distMap(pos - vec3(RAYMARCH_EPSILON, 0, 0)),
        distMap(pos + vec3(0, RAYMARCH_EPSILON, 0)) - distMap(pos - vec3(0, RAYMARCH_EPSILON, 0)),
        distMap(pos + vec3(0, 0, RAYMARCH_EPSILON)) - distMap(pos - vec3(0, 0, RAYMARCH_EPSILON))
    ));
}

__global__ void runRaymarch(const int width, const int height, float ratio, vec3 cameraPos, vec3 cameraDir, vec3 cameraUp, float fov, vec3 lightPos, int maxIters, unsigned int* colors){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= width || pixel_y >= height) {return;}

    // Creates an x and y value between -1.0 and 1.0 for each pixel
    vec2 uv = -vec2(2.0 * ((float) pixel_x) / ((float) width) - 1.0, 2.0 * ((float) pixel_y) / ((float) height) - 1.0);

    // Determines an orthogonal "right" direction based on camera's (unit) pointing and upward direction
    vec3 cameraRight = cross(cameraDir, cameraUp);
    vec3 rd = normalize(cameraDir + (uv.x * cameraRight + (uv.y / ratio) * cameraUp) * tanf(fov / 2.0));

    // Raymarches each point
    vec4 rayEnd = raymarch(cameraPos, rd, 5.0, maxIters);

    vec3 pos = rayEnd;
    float iters = rayEnd.w;
    
    unsigned int color;
    
    // Determines whether to color pixel
    if(iters >= 0.0){
        float shadow = marchLight(pos, lightPos, 8.0 * RAYMARCH_EPSILON, maxIters);
        vec3 normal = getNormal(pos);
        color = getLighting(pos, lightPos, normal, shadow, iters, maxIters);
    }else{
        color = 0xff000000;
    }

    // Writes color to image buffer array
    colors[pixel_y * width + pixel_x] = color;
}

extern "C" void render_raymarch(
    const int width, const int height,
    vec3 cameraPos, vec3 cameraDir, vec3 cameraUp, float fov,
    vec3 lightPos, int maxIters,
    unsigned int* colors
){
    unsigned int* d_colors;
    
    // Allocates device memory for color array (image buffer)
    cudaMalloc(&d_colors, width * height * sizeof(unsigned int)); 

    // Defines thread and block sizes for kernel launch
    dim3 threads(16, 16);
    dim3 block((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    float ratio = (float) width / (float) height;

    runRaymarch<<<block, threads>>>(width, height, ratio, cameraPos, normalize(cameraDir), normalize(cameraUp), fov * (M_PI / 180), lightPos, maxIters, d_colors);

    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_colors);
}
