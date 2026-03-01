#include "../Host_Device_Shared/vec.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include "../Host_Device_Shared/ThreeDimensionStructs.h"
#include "color.cuh" // Contains overlay_pixel and set_pixel
#include "common_graphics.cuh" // Contains fill_circle

namespace Cuda {

__global__ void render_points_kernel(
    unsigned int* pixels, int width, int height,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Point* points, int num_points,
    quat camera_direction, vec3 camera_pos, float fov)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) return;
    Point p = points[idx];
    if (p.opacity == 0) return;
    bool behind_camera = false;
    Cuda::vec3 pixel;
    d_coordinate_to_pixel(
        p.center, behind_camera,
        camera_direction, camera_pos, fov,
        geom_mean_size, width, height, pixel);
    if (behind_camera) return;
    float dot_size = p.size * points_radius_multiplier * geom_mean_size / 400.0f;
    d_fill_circle(pixel.x, pixel.y, dot_size, p.color, pixels, width, height, points_opacity * p.opacity);
}

__global__ void render_lines_kernel(
    unsigned int* pixels, int width, int height,
    float geom_mean_size, int thickness, float lines_opacity,
    Line* lines, int num_lines,
    quat camera_direction, vec3 camera_pos, float fov)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_lines) return;
    Line ln = lines[idx];
    if (ln.opacity == 0) return;
    bool behind_camera1 = false, behind_camera2 = false;
    vec3 p1, p2;
    d_coordinate_to_pixel(
        ln.start, behind_camera1,
        camera_direction, camera_pos, fov,
        geom_mean_size, width, height, p1);
    if (behind_camera1) return;
    d_coordinate_to_pixel(
        ln.end,   behind_camera2,
        camera_direction, camera_pos, fov,
        geom_mean_size, width, height, p2);
    if (behind_camera2) return;
    bresenham(
        p1.x, p1.y, p2.x, p2.y,
        ln.color, lines_opacity * ln.opacity, thickness,
        pixels, width, height);
}

extern "C" void render_points_on_gpu(
    unsigned int* h_pixels, int width, int height,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Point* h_points, int num_points,
    quat camera_direction, vec3 camera_pos, float fov)
{
    unsigned int* d_pixels = nullptr;
    Point*        d_points = nullptr;
    size_t pix_sz = width * height * sizeof(unsigned int);
    size_t pt_sz  = num_points * sizeof(Point);

    cudaMalloc((void**)&d_pixels, pix_sz);
    cudaMemcpy(d_pixels, h_pixels, pix_sz, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_points, pt_sz);
    cudaMemcpy(d_points, h_points, pt_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    render_points_kernel<<<numBlocks, blockSize>>>(
        d_pixels, width, height,
        geom_mean_size, points_opacity, points_radius_multiplier,
        d_points, num_points,
        camera_direction, camera_pos, fov);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_points);
}

extern "C" void render_lines_on_gpu(
    unsigned int* h_pixels, int width, int height,
    float geom_mean_size, int thickness, float lines_opacity,
    Line* h_lines, int num_lines,
    quat camera_direction, vec3 camera_pos, float fov)
{
    unsigned int* d_pixels = nullptr;
    Line*         d_lines  = nullptr;
    size_t pix_sz = width * height * sizeof(unsigned int);
    size_t ln_sz  = num_lines * sizeof(Line);

    cudaMalloc((void**)&d_pixels, pix_sz);
    cudaMemcpy(d_pixels, h_pixels, pix_sz, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_lines, ln_sz);
    cudaMemcpy(d_lines, h_lines, ln_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_lines + blockSize - 1) / blockSize;
    render_lines_kernel<<<numBlocks, blockSize>>>(
        d_pixels, width, height,
        geom_mean_size, thickness, lines_opacity,
        d_lines, num_lines,
        camera_direction, camera_pos, fov);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_lines);
}

}
