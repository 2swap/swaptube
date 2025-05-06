#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "../Scenes/Common/ThreeDimensionStructs.cpp"

__device__ inline int geta(int color) { 
    return (color >> 24) & 0xFF; 
}

__device__ inline int colorlerp(int c1, int c2, float t) {
    int a1 = (c1 >> 24) & 0xFF; int r1 = (c1 >> 16) & 0xFF; int g1 = (c1 >> 8) & 0xFF; int b1 = c1 & 0xFF;
    int a2 = (c2 >> 24) & 0xFF; int r2 = (c2 >> 16) & 0xFF; int g2 = (c2 >> 8) & 0xFF; int b2 = c2 & 0xFF;
    int a = roundf((1 - t) * a1 + t * a2);
    int r = roundf((1 - t) * r1 + t * r2);
    int g = roundf((1 - t) * g1 + t * g2);
    int b = roundf((1 - t) * b1 + t * b2);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

__device__ inline double square(double x) {
    return x * x;
}

__device__ int device_color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0f;
    float over_opacity = geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = roundf(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

__device__ void overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int idx = y * width + x;
    int base = pixels[idx];
    int blended = device_color_combine(base, col, opacity);
    pixels[idx] = blended;
}

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
    //printf("circle: (%f, %f) from point (%f, %f, %f)\n", outx, outy, coordinate.x, coordinate.y, coordinate.z);
}

__device__ void device_fill_circle(float cx, float cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    for (float dx = -r; dx < r; dx++) {
        float sdx = square(dx);
        for (float dy = -r; dy < r; dy++) {
            if (sdx + square(dy) < r*r)
                overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
        }
    }
}

__device__ void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness, unsigned int* pixels, int width, int height) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    if (dx > 10000 || dy > 10000) return;
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    while (true) {
        overlay_pixel(x1, y1, col, opacity, pixels, width, height);
        for (int i = 1; i < thickness; i++) {
            overlay_pixel(x1 + i, y1, col, opacity, pixels, width, height);
            overlay_pixel(x1 - i, y1, col, opacity, pixels, width, height);
            overlay_pixel(x1, y1 + i, col, opacity, pixels, width, height);
            overlay_pixel(x1, y1 - i, col, opacity, pixels, width, height);
        }
        if (x1 == x2 && y1 == y2) break;
        int e2 = err * 2;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
    }
}

__global__ void render_points_kernel(
    unsigned int* pixels, int width, int height,
    float geom_mean_size, float points_opacity,
    Point* points, int num_points,
    glm::quat camera_direction, glm::vec3 camera_pos, glm::quat conjugate_camera_direction, float fov)
{
    float size_scale = geom_mean_size / 400.f;
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
    float dot_size = size_scale * p.size;
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
    bool behind_camera = false;
    float p1x, p1y, p2x, p2y;
    device_coordinate_to_pixel(
        ln.start, behind_camera,
        camera_direction, camera_pos, conjugate_camera_direction, fov,
        geom_mean_size, width, height, p1x, p1y);
    device_coordinate_to_pixel(
        ln.end,   behind_camera,
        camera_direction, camera_pos, conjugate_camera_direction, fov,
        geom_mean_size, width, height, p2x, p2y);
    if (behind_camera) return;
    bresenham(
        p1x, p1y, p2x, p2y,
        ln.color, lines_opacity * ln.opacity, thickness,
        pixels, width, height);
}

extern "C" void render_points_on_gpu(
    unsigned int* h_pixels, int width, int height,
    float geom_mean_size, float points_opacity,
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
        geom_mean_size, points_opacity,
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
