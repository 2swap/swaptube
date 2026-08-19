// Draws simple geometric shapes
#include <cuda_runtime.h>
#include "color.cuh"
#include "common_graphics.cuh"

__global__ void circle_kernel(uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 center, const float radius_squared, const uint32_t color, const float opacity, const Cuda::ivec2 min_pos)
{
    Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x >= wh.x || pos.y >= wh.y) return;

    Cuda::vec2 delta = pos - center;
    if (dot(delta, delta) <= radius_squared) {
        overlay_pixel(pos, color, opacity, pix, wh);
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
        overlay_pixel(pos, color, 1.0f, pix, wh);
    }
}

extern "C" void draw_circle(uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& center, const float radius, const uint32_t color, const float opacity)
{
    if (opacity <= 0.0f) return;
    const Cuda::ivec2 min_pos(max(0, (int)floorf(center.x - radius)), max(0, (int)floorf(center.y - radius)));
    const Cuda::ivec2 max_pos(min(wh.x, (int)ceilf(center.x + radius)), min(wh.y, (int)ceilf(center.y + radius)));

    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x, (size.y + blockSize.y - 1) / blockSize.y);
    const float radius_squared = radius * radius;
    circle_kernel<<<gridSize, blockSize>>>(pix, wh, center, radius_squared, color, opacity, min_pos);
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
        overlay_pixel(pos, color, 1.0f, pix, wh);
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
        overlay_pixel(pos, color, 1.0f, pix, wh);
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

__global__ void convex_polygon_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::vec2* verts, const int n,
    const uint32_t color, const float opacity,
    const Cuda::ivec2 min_pos, const Cuda::ivec2 max_pos)
{
    const Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x,
                                                  blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= max_pos.x || pos.y >= max_pos.y || pos.x >= wh.x || pos.y >= wh.y) return;

    const Cuda::vec2 p(pos.x, pos.y);

    float nearest = 1e30f;
    for (int i = 0; i < n; i++) {
        const Cuda::vec2 a = verts[i];
        const Cuda::vec2 b = verts[(i + 1) % n];
        const Cuda::vec2 ab = b - a;
        const float len = Cuda::length(ab);
        if (len < 1e-6f) continue;   // coincident vertices carry no edge
        const Cuda::vec2 ap = p - a;
        const float signed_distance = (ab.y * ap.x - ab.x * ap.y) / len;   // > 0 to the left, clockwise winding
        if (signed_distance < nearest) nearest = signed_distance;
    }

    const float coverage = Cuda::clamp(0.5f+nearest, 0.0f, 1.0f);
    if (coverage <= 0.0f) return;
    overlay_pixel(pos, color, coverage * opacity, pixels, wh);
}

extern "C" void draw_convex_polygon(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::vec2* h_verts, int n,
    uint32_t color, float opacity)
{
    if (n < 3 || opacity <= 0.0f) return;

    // Only the polygon's own bounding box can be covered.
    float lx = h_verts[0].x, rx = h_verts[0].x, ty = h_verts[0].y, by = h_verts[0].y;
    for (int i = 1; i < n; i++) {
        lx = fminf(lx, h_verts[i].x); rx = fmaxf(rx, h_verts[i].x);
        ty = fminf(ty, h_verts[i].y); by = fmaxf(by, h_verts[i].y);
    }
    const int left = (int)floorf(lx) - 1, top    = (int)floorf(ty) - 1;
    const int right = (int)ceilf(rx) + 1, bottom = (int)ceilf(by)  + 1;
    const Cuda::ivec2 min_pos(left  > 0     ? left  : 0,     top    > 0     ? top    : 0);
    const Cuda::ivec2 max_pos(right < wh.x  ? right : wh.x,  bottom < wh.y  ? bottom : wh.y);
    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    Cuda::vec2* d_verts = nullptr;
    const size_t bytes = (size_t)n * sizeof(Cuda::vec2);
    cudaMalloc(&d_verts, bytes);
    cudaMemcpy(d_verts, h_verts, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);
    convex_polygon_kernel<<<grid, block>>>(d_pixels, wh, d_verts, n, color, opacity, min_pos, max_pos);
    cudaDeviceSynchronize();

    cudaFree(d_verts);
}

__global__ void bezier_kernel(
    uint32_t* pix, const Cuda::ivec2 wh, const Cuda::vec2 p1, const Cuda::vec2 p2, const Cuda::vec2 p3, const Cuda::vec2 p4, 
    Cuda::vec2 lx_ty, Cuda::vec2 rx_by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1000) return;

    Cuda::vec2 point1 = point_to_pixel_in_screen(bezier_2d(
            p1, p2, p3, p4, i/1000.0f), lx_ty, rx_by, wh);
    Cuda::vec2 point2 = point_to_pixel_in_screen(bezier_2d(
            p1, p2, p3, p4, (i+1)/1000.0f), lx_ty, rx_by, wh);

    bresenham(point1.x, point1.y, point2.x, point2.y, 0xFFFFFFFF, 1.0f, 2, pix, wh, false);
}

extern "C" void cuda_draw_bezier(
    uint32_t* pix, const Cuda::ivec2& wh, const Cuda::vec2& p1, const Cuda::vec2& p2, const Cuda::vec2& p3, 
    const Cuda::vec2& p4, const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by)
{
    int blockSize = 256;
    int gridSize = (1000 + blockSize - 1) / blockSize;
    bezier_kernel<<<gridSize, blockSize>>>(pix, wh, p1, p2, p3, p4, lx_ty, rx_by);
    cudaDeviceSynchronize();
}

__global__ void render_path_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh, const Cuda::vec2* rope, const int rope_length, Cuda::vec2 lx_ty, Cuda::vec2 rx_by,
    const uint32_t color, const float opacity, const int thickness, const int closed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int segment_count = closed ? rope_length : rope_length - 1;
    if (i >= segment_count) return;

    Cuda::vec2 pixel_1 = point_to_pixel_in_screen(rope[i], lx_ty, rx_by, wh);
    Cuda::vec2 pixel_2 = point_to_pixel_in_screen(rope[(i+1)%rope_length], lx_ty, rx_by, wh);

    bresenham(pixel_1.x, pixel_1.y, pixel_2.x, pixel_2.y, color, opacity, thickness, pixels, wh, false);
}

// `d_path` is already a device pointer - the caller's, kept alive across frames
// (see Rope, which owns one for its whole lifetime). `closed` decides whether the
// last point connects back to the first, which is what turns a polyline into a
// loop.
extern "C" void cuda_render_path(uint32_t* d_pixels, const Cuda::ivec2& wh, const Cuda::vec2* d_path, const int path_length,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by, const uint32_t color, const float opacity, const float thickness, const bool closed)
{
    const int segment_count = closed ? path_length : path_length - 1;
    if (segment_count <= 0) return;

    int blockSize = 256;
    int gridSize = (segment_count + blockSize - 1) / blockSize;
    render_path_kernel<<<gridSize, blockSize>>>(d_pixels, wh, d_path, path_length, lx_ty, rx_by, color, opacity, (int)thickness, closed ? 1 : 0);
    cudaDeviceSynchronize();
}

// Same, but for a path that only exists on the host: allocates a device buffer
// for the frame, copies it over, draws, and frees it again. For a caller with no
// device buffer of its own to keep around - see OuterBilliardsScene, which
// rebuilds its orbit from scratch every frame anyway.
extern "C" void cuda_render_path_from_host(uint32_t* d_pixels, const Cuda::ivec2& wh, const Cuda::vec2* h_path, const int path_length,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by, const uint32_t color, const float opacity, const float thickness, const bool closed)
{
    if (path_length < 2) return;

    Cuda::vec2* d_path = nullptr;
    const size_t bytes = (size_t)path_length * sizeof(Cuda::vec2);
    cudaMalloc(&d_path, bytes);
    cudaMemcpy(d_path, h_path, bytes, cudaMemcpyHostToDevice);

    cuda_render_path(d_pixels, wh, d_path, path_length, lx_ty, rx_by, color, opacity, thickness, closed);

    cudaFree(d_path);
}
