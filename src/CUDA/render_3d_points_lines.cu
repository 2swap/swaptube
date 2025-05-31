#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "../Scenes/Common/ThreeDimensionStructs.cpp"
#include "../misc/cuda_color.cu" // Contains overlay_pixel and set_pixel

__device__ void device_coordinate_to_pixel(
    const glm::vec3& coordinate,
    bool &behind_camera,
    const glm::quat& camera_direction,
    const glm::vec3& camera_pos,
    const glm::quat& conjugate_camera_direction,
    float fov,
    float geom_mean_size,
    int width,
    int height,
    float& outx,
    float& outy)
{
    glm::vec3 rotated = camera_direction * (coordinate - camera_pos) * conjugate_camera_direction;
    if (rotated.z <= 0) { behind_camera = true; return; }
    float scale = (geom_mean_size * fov) / rotated.z;
    outx = scale * rotated.x + width * 0.5f;
    outy = scale * rotated.y + height * 0.5f;
}

__device__ void device_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    for (float dx = -r; dx < r; dx++) {
        float sdx = square(dx);
        for (float dy = -r; dy < r; dy++) {
            if (sdx + square(dy) < r*r)
                //overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
                    set_pixel(cx + dx, cy + dy, col     , pixels, width, height);
        }
    }
}

__device__ __forceinline__ void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness, unsigned int* pixels, int width, int height) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        set_pixel(x1, y1, col,          pixels, width, height);
        //overlay_pixel(x1, y1, col, opacity, pixels, width, height);
        for (int i = 1; i < thickness; i++) {
            set_pixel(x1 + i, y1, col, pixels, width, height);
            set_pixel(x1 - i, y1, col, pixels, width, height);
            set_pixel(x1, y1 + i, col, pixels, width, height);
            set_pixel(x1, y1 - i, col, pixels, width, height);
            //overlay_pixel(x1 + i, y1, col, opacity, pixels, width, height);
            //overlay_pixel(x1 - i, y1, col, opacity, pixels, width, height);
            //overlay_pixel(x1, y1 + i, col, opacity, pixels, width, height);
            //overlay_pixel(x1, y1 - i, col, opacity, pixels, width, height);
        }
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
    }
}

__global__ void render_points_kernel(
    unsigned int* pixels, int width, int height,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Point* points, int num_points,
    glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) return;
    Point p = points[idx];
    if (p.opacity == 0) return;
    bool behind_camera = false;
    float px, py;
    device_coordinate_to_pixel(
        p.center, behind_camera,
        camera_direction, camera_pos, conjugate_camera_direction, fov,
        geom_mean_size, width, height, px, py);
    if (behind_camera) return;
    float dot_size = p.size * points_radius_multiplier * geom_mean_size / 400.0f;
    device_fill_circle(px, py, dot_size, p.color, pixels, width, height, points_opacity * p.opacity);
}

__global__ void render_lines_kernel(
    unsigned int* pixels, int width, int height,
    float geom_mean_size, int thickness, float lines_opacity,
    Line* lines, int num_lines,
    glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_lines) return;
    Line ln = lines[idx];
    if (ln.opacity == 0) return;
    bool behind_camera1 = false, behind_camera2 = false;
    float p1x, p1y, p2x, p2y;
    device_coordinate_to_pixel(
        ln.start, behind_camera1,
        camera_direction, camera_pos, conjugate_camera_direction, fov,
        geom_mean_size, width, height, p1x, p1y);
    device_coordinate_to_pixel(
        ln.end,   behind_camera2,
        camera_direction, camera_pos, conjugate_camera_direction, fov,
        geom_mean_size, width, height, p2x, p2y);
    if (behind_camera1 || behind_camera2) return;
    bresenham(
        p1x, p1y, p2x, p2y,
        ln.color, lines_opacity * ln.opacity, thickness,
        pixels, width, height);
}

extern "C" void render_points_on_gpu(
    unsigned int* h_pixels, int width, int height,
    float geom_mean_size, float points_opacity, float points_radius_multiplier,
    Point* h_points, int num_points,
    glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov)
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
        camera_direction, camera_pos, conjugate_camera_direction, fov);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_points);
}

extern "C" void render_lines_on_gpu(
    unsigned int* h_pixels, int width, int height,
    float geom_mean_size, int thickness, float lines_opacity,
    Line* h_lines, int num_lines,
    glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov)
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
        camera_direction, camera_pos, conjugate_camera_direction, fov);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_lines);
}
