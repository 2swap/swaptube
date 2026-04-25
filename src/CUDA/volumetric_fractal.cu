#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "color.cuh"
#include "common_graphics.cuh"
#include "complex_functions.cuh"

const float EPSILON = 2e-4f;
const float MAXITERS = 50;

__constant__ int fractalSDF;

// Implements basic raymarch algorithm
__device__ unsigned int sumRay(const Cuda::vec3& ro, const Cuda::vec3& rd, float maxDist, float stepSize, float p){
    Cuda::vec3 r = ro;
    float total = 0; // Between 0 and 1
    float sq_radius = 0;

    float weight = stepSize / maxDist;
    for(float t = 0; t < maxDist; t += stepSize){
        r = ro + t * rd;
        //cuComplex c = make_cuComplex(r.x, r.y);
        //cuComplex z = make_cuComplex(r.z, 0);
        Cuda::vec3 z = Cuda::vec3(0, 0, 0);
        float iters = cuCFunc::mandelbulb_iterations(z, p, r, MAXITERS, 65536, sq_radius);
        if(iters == MAXITERS){
            break;
        }/*else{
            float log_zn = log(sq_radius)/2;
            float nu = log(log_zn / log(2)) / log(2);
            iters += (1-nu); // Do not use gradient for exponential parameterization
        }*/
        total += (iters / MAXITERS) * weight;
    }

    total *= total;

    total *= 64 * p * 256;

    total = fminf(total, 255);

    return 0xff000000 + ((int) fmaxf(total - 80, 0) << 16) + ((int) fmaxf(total - 40, 0) << 8) + ((int) total);
}

__global__ void volumeRay(const int width, const int height, const Cuda::vec3 pos, const Cuda::quat camera_orientation, float fov, const Cuda::vec3 lightPos, int max_raymarch_iters, int max_mandelbulb_iters, float p, unsigned int* colors){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= width || pixel_y >= height) {return;}

    Cuda::vec3 rd = Cuda::get_raymarch_vector(pixel_x, pixel_y, width, height, fov, camera_orientation);

    // Raymarches each point
    const unsigned int color = sumRay(pos, rd, 4.0, 0.01, p);

    // Writes color to image buffer array
    colors[pixel_y * width + pixel_x] = color;
}

extern "C" void render_volume(
    const int width, const int height,
    const Cuda::vec3& pos, const Cuda::quat& camera, float fov,
    const Cuda::vec3& lightPos,
    const int max_raymarch_iters, const int max_mandelbulb_iters,
    float p,
    unsigned int* colors
){
    unsigned int* d_colors;

    // Allocates device memory for color array (image buffer)
    cudaMalloc(&d_colors, width * height * sizeof(unsigned int)); 

    // Defines thread and block sizes for kernel launch
    dim3 threads(8, 8);
    dim3 block((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    volumeRay<<<block, threads>>>(width, height, pos, normalize(camera), fov, lightPos, max_raymarch_iters, max_mandelbulb_iters, p, d_colors);

    cudaMemcpy(colors, d_colors, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_colors);
}