// Draws simple geometric shapes
#include <cuda_runtime.h>
#include "color.cuh"

__global__ void circle_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 center, const float radius_squared, const uint32_t color, const Cuda::ivec2 min_pos)
{
    Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x >= wh.x || pos.y >= wh.y) return;

    Cuda::vec2 delta = pos - center;
    if (dot(delta, delta) <= radius_squared) {
        d_overlay_pixel(pos, color, 1.0f, pix, wh);
    }
}

__device__ inline float edge_function(const Cuda::vec2& a, const Cuda::vec2& b, const Cuda::vec2& c)
{
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

__global__ void triangle_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 p0, const Cuda::vec2 p1, const Cuda::vec2 p2, const uint32_t color, const Cuda::ivec2 min_pos, const Cuda::ivec2 max_pos)
{
    Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x >= max_pos.x || pos.y >= max_pos.y || pos.x >= wh.x || pos.y >= wh.y) return;

    Cuda::vec2 p = pos;
    float w0 = edge_function(p1, p2, p);
    float w1 = edge_function(p2, p0, p);
    float w2 = edge_function(p0, p1, p);

    if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) || (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f)) {
        d_overlay_pixel(pos, color, 1.0f, pix, wh);
    }
}

extern "C" void draw_circle(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& center, const float radius, const uint32_t color)
{
    const Cuda::ivec2 min_pos(max(0, (int)floorf(center.x - radius)), max(0, (int)floorf(center.y - radius)));
    const Cuda::ivec2 max_pos(min(wh.x, (int)ceilf(center.x + radius)), min(wh.y, (int)ceilf(center.y + radius)));

    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x, (size.y + blockSize.y - 1) / blockSize.y);
    const float radius_squared = radius * radius;
    circle_kernel<<<gridSize, blockSize>>>(pix, wh, center, radius_squared, color, min_pos);
}

extern "C" void draw_triangle(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& p0, const Cuda::vec2& p1, const Cuda::vec2& p2, const uint32_t color)
{
    const Cuda::ivec2 min_pos(max(0, (int)floorf(min(p0.x, min(p1.x, p2.x)))) , max(0, (int)floorf(min(p0.y, min(p1.y, p2.y)))));
    const Cuda::ivec2 max_pos(min(wh.x, (int)ceilf(max(p0.x, max(p1.x, p2.x)))) , min(wh.y, (int)ceilf(max(p0.y, max(p1.y, p2.y)))));

    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x, (size.y + blockSize.y - 1) / blockSize.y);
    triangle_kernel<<<gridSize, blockSize>>>(pix, wh, p0, p1, p2, color, min_pos, max_pos);
}

__global__ void quad_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 p0, const Cuda::vec2 p1, const Cuda::vec2 p2, const Cuda::vec2 p3, const uint32_t color, const Cuda::ivec2 min_pos, const Cuda::ivec2 max_pos)
{
    Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x >= max_pos.x || pos.y >= max_pos.y || pos.x >= wh.x || pos.y >= wh.y) return;

    Cuda::vec2 p = pos;

    float w0 = edge_function(p0, p1, p);
    float w1 = edge_function(p1, p2, p);
    float w2 = edge_function(p2, p3, p);
    float w3 = edge_function(p3, p0, p);

    if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && w3 >= 0.0f) || (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && w3 <= 0.0f)) {
        d_overlay_pixel(pos, color, 1.0f, pix, wh);
    }
}

extern "C" void draw_quadrilateral(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& p0, const Cuda::vec2& p1, const Cuda::vec2& p2, const Cuda::vec2& p3, const uint32_t color)
{
    const Cuda::ivec2 min_pos(max(0, (int)floorf(min(min(p0.x, p1.x), min(p2.x, p3.x)))) , max(0, (int)floorf(min(min(p0.y, p1.y), min(p2.y, p3.y)))));
    const Cuda::ivec2 max_pos(min(wh.x, (int)ceilf(max(max(p0.x, p1.x), max(p2.x, p3.x)))) , min(wh.y, (int)ceilf(max(max(p0.y, p1.y), max(p2.y, p3.y)))));

    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x, (size.y + blockSize.y - 1) / blockSize.y);
    quad_kernel<<<gridSize, blockSize>>>(pix, wh, p0, p1, p2, p3, color, min_pos, max_pos);
}

__global__ void rectangle_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::ivec2 top_left, const Cuda::ivec2 bottom_right, const uint32_t color, const Cuda::ivec2 min_pos, const Cuda::ivec2 max_pos)
{
    Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x >= max_pos.x || pos.y >= max_pos.y || pos.x >= wh.x || pos.y >= wh.y) return;

    if (pos.x >= top_left.x && pos.x < bottom_right.x && pos.y >= top_left.y && pos.y < bottom_right.y) {
        d_overlay_pixel(pos, color, 1.0f, pix, wh);
    }
}

extern "C" void draw_rectangle(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::ivec2& top_left, const Cuda::ivec2& bottom_right, const uint32_t color)
{
    const Cuda::ivec2 min_pos(max(0, top_left.x), max(0, top_left.y));
    const Cuda::ivec2 max_pos(min(wh.x, bottom_right.x), min(wh.y, bottom_right.y));

    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x, (size.y + blockSize.y - 1) / blockSize.y);
    rectangle_kernel<<<gridSize, blockSize>>>(pix, wh, min_pos, max_pos, color, min_pos, max_pos);
}
