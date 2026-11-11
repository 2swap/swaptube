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

__global__ void render_points_kernel(
    unsigned int* pixels, const Cuda::ivec2 wh,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Cuda::Point* points, int num_points,
    Cuda::quat camera_direction, Cuda::vec3 camera_pos, float fov,
    bool brightest_wins, // TEMP (FourDAlgebra): revert ASAP
    unsigned long long* depth)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) return;
    Cuda::Point p = points[idx];
    if (p.opacity == 0) return;
    const Cuda::vec3 pixel = Cuda::coordinate_to_pixel(
        p.center, camera_direction, camera_pos, fov, geom_mean_size, wh);
    if (pixel.z <= 0) return; // behind the camera
    float dot_size = p.size * points_radius_multiplier * geom_mean_size / 140.0f;
    const float world_r = dot_size * pixel.z / (geom_mean_size * fov);
    Cuda::d_fill_circle(pixel, dot_size, p.color, pixels, wh, points_opacity * p.opacity, brightest_wins, depth, pixel.z, world_r);
}

__global__ void render_lines_kernel(
    unsigned int* pixels, const Cuda::ivec2 wh,
    float geom_mean_size, int thickness, float lines_opacity,
    Cuda::Line* lines, int num_lines,
    Cuda::quat camera_direction, Cuda::vec3 camera_pos, float fov,
    bool brightest_wins, // TEMP (FourDAlgebra): revert ASAP
    unsigned long long* depth)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_lines) return;
    Cuda::Line ln = lines[idx];
    if (ln.opacity == 0) return;
    const Cuda::vec3 p1 = Cuda::coordinate_to_pixel(
        ln.start, camera_direction, camera_pos, fov, geom_mean_size, wh);
    if (p1.z <= 0) return; // an endpoint behind the camera drops the whole line
    const Cuda::vec3 p2 = Cuda::coordinate_to_pixel(
        ln.end, camera_direction, camera_pos, fov, geom_mean_size, wh);
    if (p2.z <= 0) return;
    Cuda::bresenham(
        p1, p2,
        ln.color, lines_opacity * ln.opacity, thickness,
        pixels, wh, ln.is_dashed, brightest_wins, depth, p1.z, p2.z);
}

__global__ void resolve_depth_kernel(unsigned int* pixels, const Cuda::ivec2 wh, const unsigned long long* depth) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= wh.x * wh.y) return;
    const unsigned long long key = depth[idx];
    if (key == ~0ull) return;
    pixels[idx] = Cuda::color_combine(pixels[idx], (unsigned int)key);
}

extern "C" unsigned long long* create_depth_buffer(const Cuda::ivec2& wh) {
    unsigned long long* d_depth = nullptr;
    const size_t sz = (size_t)wh.x * wh.y * sizeof(unsigned long long);
    cudaMalloc((void**)&d_depth, sz);
    cudaMemset(d_depth, 0xff, sz);
    return d_depth;
}

extern "C" void resolve_depth_buffer(unsigned int* d_pixels, const Cuda::ivec2& wh, unsigned long long* d_depth) {
    int blockSize = 256;
    int numBlocks = (wh.x * wh.y + blockSize - 1) / blockSize;
    resolve_depth_kernel<<<numBlocks, blockSize>>>(d_pixels, wh, d_depth);
    cudaDeviceSynchronize();
    cudaFree(d_depth);
}

extern "C" void render_points_on_gpu(
    unsigned int* d_pixels, const Cuda::ivec2& wh,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Cuda::Point* h_points, int num_points,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov,
    bool brightest_wins, // TEMP (FourDAlgebra): revert after this video
    unsigned long long* d_depth)
{
    Cuda::Point*  d_points = nullptr;
    size_t pt_sz  = num_points * sizeof(Cuda::Point);

    cudaMalloc((void**)&d_points, pt_sz);
    cudaMemcpy(d_points, h_points, pt_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    render_points_kernel<<<numBlocks, blockSize>>>(
        d_pixels, wh,
        geom_mean_size, points_opacity, points_radius_multiplier,
        d_points, num_points,
        camera_direction, camera_pos, fov,
        brightest_wins, d_depth);
    cudaDeviceSynchronize();

    // TODO we don't really need to copy these back and forth every frame...
    cudaFree(d_points);
}

extern "C" void render_lines_on_gpu(
    unsigned int* d_pixels, const Cuda::ivec2& wh,
    float geom_mean_size, int thickness, float lines_opacity,
    Cuda::Line* h_lines, int num_lines,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov,
    bool brightest_wins, // TEMP (FourDAlgebra): revert after this video
    unsigned long long* d_depth)
{
    Cuda::Line* d_lines  = nullptr;
    size_t ln_sz  = num_lines * sizeof(Cuda::Line);

    cudaMalloc((void**)&d_lines, ln_sz);
    cudaMemcpy(d_lines, h_lines, ln_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_lines + blockSize - 1) / blockSize;
    render_lines_kernel<<<numBlocks, blockSize>>>(
        d_pixels, wh,
        geom_mean_size, thickness, lines_opacity,
        d_lines, num_lines,
        camera_direction, camera_pos, fov,
        brightest_wins, d_depth);
    cudaDeviceSynchronize();

    cudaFree(d_lines);
}
