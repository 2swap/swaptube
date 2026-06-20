#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include <vector>
#include <iostream>
#include "color.cuh"

__global__ void render_surface_kernel(
    // d_pixels is the array which we will plop a surface on. The surface is bounded by (x1,y1) on the top left and (x2,y2) on the top right.
    uint32_t* d_pixels,
    int x1,
    int y1,
    const Cuda::ivec2 plot_wh,
    int pixels_w,
    uint32_t* d_surface,
    const Cuda::ivec2 surface_wh,
    float opacity,
    Cuda::vec3 camera_pos,
    Cuda::quat camera_direction,
    float dotnormcam,
    const Cuda::vec3 surface_normal,
    const Cuda::vec3 surface_center,
    const Cuda::vec3 surface_pos_x_dir,
    const Cuda::vec3 surface_pos_y_dir,
    const float surface_ilr2,
    const float surface_iur2,
    float halfwidth,
    float halfheight,
    float over_w_fov) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= plot_wh.x || py >= plot_wh.y) return;

    // Compute the ray direction from the camera through the screen point
    Cuda::vec3 ray_dir((px - halfwidth) * over_w_fov, (py - halfheight) * over_w_fov, 1.0f);
    ray_dir = rotate_vector(ray_dir, camera_direction);

    // Compute the intersection point in 3D space
    const float t = dotnormcam / Cuda::dot(surface_normal, ray_dir);

    // Convert 3D intersection point to surface's local 2D coordinates
    Cuda::vec3 centered = camera_pos + t * ray_dir - surface_center;

    Cuda::vec2 surface_coords(
        Cuda::dot(centered, surface_pos_x_dir) * surface_ilr2 + 0.5f,
        Cuda::dot(centered, surface_pos_y_dir) * surface_iur2 + 0.5f
    );

    int pixels_index = px + py * pixels_w;

    // If this pixel does not intersect the surface, return
    if (surface_coords.x >= 1 || surface_coords.x < 0 || surface_coords.y >= 1 || surface_coords.y < 0)  return;

    const Cuda::ivec2 surface_xy = Cuda::ivec2(surface_coords.x * surface_wh.x, surface_coords.y * surface_wh.y);

    uint32_t color = d_surface[surface_xy.x + surface_xy.y * surface_wh.x];

    d_pixels[pixels_index] = Cuda::color_combine(d_pixels[pixels_index], color, opacity);
}

extern "C" void cuda_render_surface(
    uint32_t* d_pixels,
    int x1,
    int y1,
    const Cuda::ivec2& plot_wh,
    int pixels_w,
    uint32_t* h_surface,
    const Cuda::ivec2& surface_wh,
    float opacity,
    Cuda::vec3 camera_pos,
    Cuda::quat camera_direction,
    const Cuda::vec3& surface_normal,
    const Cuda::vec3& surface_center,
    const Cuda::vec3& surface_pos_x_dir,
    const Cuda::vec3& surface_pos_y_dir,
    const float surface_ilr2,
    const float surface_iur2,
    float halfwidth,
    float halfheight,
    float over_w_fov) {
    
    float dotnormcam = Cuda::dot(surface_normal, (surface_center - camera_pos));

    size_t surface_size = surface_wh.x * surface_wh.y * sizeof(uint32_t);
    uint32_t* d_surface;
    cudaMalloc(&d_surface, surface_size);

    // Copy data to device
    cudaMemcpy(d_surface, h_surface, surface_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    int numBlocks = (plot_wh.x * plot_wh.y + blockSize.x - 1) / blockSize.x;

    // Launch the kernel
    render_surface_kernel<<<numBlocks, blockSize>>>(
        d_pixels, x1, y1, plot_wh, pixels_w,
        d_surface, surface_wh, opacity,
        camera_pos, conjugate(camera_direction),
        dotnormcam, surface_normal, surface_center, surface_pos_x_dir, surface_pos_y_dir, surface_ilr2, surface_iur2, halfwidth, halfheight, over_w_fov
    );

    cudaFree(d_surface);
}
