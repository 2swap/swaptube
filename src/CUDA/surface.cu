#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include <vector>
#include <iostream>
#include "color.cuh"

__global__ void render_surface_kernel(
    // d_pixels is the array which we will plop a surface on. The surface is bounded by (x1,y1) on the top left and (x2,y2) on the top right.
    unsigned int* d_pixels_dev,
    const Cuda::vec2 x1y1,
    const Cuda::vec2 plot_size,
    const Cuda::vec2 pixels_size,
    unsigned int* d_surface,
    const Cuda::vec2 surface_size,
    float opacity,
    const Cuda::vec3 camera_pos,
    const Cuda::quat camera_direction,
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels_to_render = plot_size.x * plot_size.y;

    if (idx >= num_pixels_to_render) return;

    // Compute pixel coordinates in plot
    int px = idx % (int)plot_size.x + x1y1.x;
    int py = idx / plot_size.y + x1y1.y;

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

    int pixels_index = px + py * pixels_size.x;

    // If this pixel does not intersect the surface, return
    if (surface_coords.x >= 1 || surface_coords.x < 0 || surface_coords.y >= 1 || surface_coords.y < 0)  return;

    int surface_x = surface_coords.x * surface_size.x;
    int surface_y = surface_coords.y * surface_size.y;

    unsigned int color = d_surface[surface_y * (int)surface_size.x + surface_x];

    d_pixels_dev[pixels_index] = d_color_combine(d_pixels_dev[pixels_index], color, opacity);
}

extern "C" void cuda_render_surface(
    vector<unsigned int>& pix,
    const Cuda::vec2& x1y1,
    const Cuda::vec2& plot_size,
    const Cuda::vec2& pixels_size,
    unsigned int* d_surface,
    const Cuda::vec2& surface_size,
    float opacity,
    const Cuda::vec3& camera_pos,
    const Cuda::quat& camera_direction,
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

    // Allocate memory on the device
    unsigned int* d_pixels_dev;
    size_t pixels_mem = pixels_size.x * pixels_size.y * sizeof(unsigned int);
    cudaMalloc(&d_pixels_dev, pixels_mem);
    cudaMemcpy(d_pixels_dev, pix.data(), pixels_mem, cudaMemcpyHostToDevice);

    size_t surface_mem = surface_size.x * surface_size.y * sizeof(unsigned int);
    unsigned int* d_surface_dev;
    cudaMalloc(&d_surface_dev, surface_mem);

    // Copy data to device
    cudaMemcpy(d_surface_dev, d_surface, surface_mem, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (plot_size.x * plot_size.y + blockSize - 1) / blockSize;

    // Launch the kernel
    render_surface_kernel<<<numBlocks, blockSize>>>(
        d_pixels_dev, x1y1, plot_size, pixels_size,
        d_surface_dev, surface_size, opacity,
        camera_pos, camera_direction,
        dotnormcam, surface_normal, surface_center, surface_pos_x_dir, surface_pos_y_dir, surface_ilr2, surface_iur2, halfwidth, halfheight, over_w_fov
    );

    // Copy results back to host
    cudaMemcpy(pix.data(), d_pixels_dev, pixels_mem, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels_dev);
    cudaFree(d_surface_dev);
}
