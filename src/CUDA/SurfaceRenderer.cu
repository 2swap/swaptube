#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <glm/gtx/quaternion.hpp>
#include <iostream>

using namespace std;

__device__ inline float d_lerp(float a, float b, float w) { return a * (1 - w) + b * w; }
__device__ inline int d_argb(int a, int r, int g, int b) {
    return (a << 24) +
           (r << 16) +
           (g << 8) +
           (b);
}
__device__ inline int d_geta(int col) { return (col & 0xff000000) >> 24; }
__device__ inline int d_getr(int col) { return (col & 0x00ff0000) >> 16; }
__device__ inline int d_getg(int col) { return (col & 0x0000ff00) >> 8; }
__device__ inline int d_getb(int col) { return (col & 0x000000ff); }
__device__ inline int d_colorlerp(int col1, int col2, float w) {
    return d_argb(round(d_lerp(d_geta(col1), d_geta(col2), w)),
                         round(d_lerp(d_getr(col1), d_getr(col2), w)),
                         round(d_lerp(d_getg(col1), d_getg(col2), w)),
                         round(d_lerp(d_getb(col1), d_getb(col2), w)));
}
__device__ int color_combine_device(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = d_geta(base_color) / 255.0f;
    float over_opacity = d_geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = round(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = d_colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

__global__ void render_surface_kernel(
    // d_pixels is the array which we will plop a surface on. The surface is bounded by (x1,y1) on the top left and (x2,y2) on the top right.
    unsigned int* d_pixels_dev,
    int x1,
    int y1,
    int plot_w,
    int plot_h,
    int pixels_w,
    unsigned int* d_surface,
    int surface_w,
    int surface_h,
    float opacity,
    glm::vec3 camera_pos,
    glm::quat camera_direction,
    glm::quat conjugate_camera_direction,
    float dotnormcam,
    const glm::vec3 surface_normal,
    const glm::vec3 surface_center,
    const glm::vec3 surface_pos_x_dir,
    const glm::vec3 surface_pos_y_dir,
    const float surface_ilr2,
    const float surface_iur2,
    float halfwidth,
    float halfheight,
    float over_w_fov) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels_to_render = plot_w * plot_h;

    if (idx >= num_pixels_to_render) return;

    // Compute pixel coordinates in plot
    int px = idx % plot_w + x1;
    int py = idx / plot_w + y1;

    // Compute the ray direction from the camera through the screen point
    glm::vec3 ray_dir((px - halfwidth) * over_w_fov, (py - halfheight) * over_w_fov, 1.0f);
    ray_dir = conjugate_camera_direction * ray_dir * camera_direction;

    // Compute the intersection point in 3D space
    const float t = dotnormcam / glm::dot(surface_normal, ray_dir);

    // Convert 3D intersection point to surface's local 2D coordinates
    glm::vec3 centered = camera_pos + t * ray_dir - surface_center;

    glm::vec2 surface_coords(
        glm::dot(centered, surface_pos_x_dir) * surface_ilr2 + 0.5f,
        glm::dot(centered, surface_pos_y_dir) * surface_iur2 + 0.5f
    );

    // If this pixel does not intersect the surface, return
    if (surface_coords.x >= 1 || surface_coords.x < 0 || surface_coords.y >= 1 || surface_coords.y < 0) return;

    int surface_x = surface_coords.x * surface_w;
    int surface_y = surface_coords.y * surface_h;

    unsigned int color = d_surface[surface_x + surface_y * surface_w];

    int pixels_index = px + py * pixels_w;

    d_pixels_dev[pixels_index] = color_combine_device(d_pixels_dev[pixels_index], color, opacity);
}

extern "C" void cuda_render_surface(
    vector<unsigned int>& pix,
    int x1,
    int y1,
    int plot_w,
    int plot_h,
    int pixels_w,
    unsigned int* d_surface,
    int surface_w,
    int surface_h,
    float opacity,
    glm::vec3 camera_pos,
    glm::quat camera_direction,
    glm::quat conjugate_camera_direction,
    const glm::vec3& surface_normal,
    const glm::vec3& surface_center,
    const glm::vec3& surface_pos_x_dir,
    const glm::vec3& surface_pos_y_dir,
    const float surface_ilr2,
    const float surface_iur2,
    float halfwidth,
    float halfheight,
    float over_w_fov) {
    
    float dotnormcam = glm::dot(surface_normal, (surface_center - camera_pos));

    // Allocate memory on the device
    size_t pixels_size = pix.size() * sizeof(int);
    unsigned int* d_pixels_dev;
    cudaMalloc(&d_pixels_dev, pixels_size);
    cudaMemcpy(d_pixels_dev, pix.data(), pixels_size, cudaMemcpyHostToDevice);

    size_t surface_size = surface_w * surface_h * sizeof(unsigned int);
    unsigned int* d_surface_dev;
    cudaMalloc(&d_surface_dev, surface_size);

    // Copy data to device
    cudaMemcpy(d_surface_dev, d_surface, surface_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (plot_w * plot_h + blockSize - 1) / blockSize;

    // Launch the kernel
    render_surface_kernel<<<numBlocks, blockSize>>>(
        d_pixels_dev, x1, y1, plot_w, plot_h, pixels_w,
        d_surface_dev, surface_w, surface_h, opacity,
        camera_pos, camera_direction, conjugate_camera_direction,
        dotnormcam, surface_normal, surface_center, surface_pos_x_dir, surface_pos_y_dir, surface_ilr2, surface_iur2, halfwidth, halfheight, over_w_fov
    );

    // Copy results back to host
    cudaMemcpy(pix.data(), d_pixels_dev, pixels_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels_dev);
    cudaFree(d_surface_dev);
}
