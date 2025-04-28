#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//////////////////////
// Helper functions //
//////////////////////

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

//////////////////////////////
// Device pixel overlaying  //
//////////////////////////////

__device__ void overlay_pixel(int x, int y, int col, float opacity, unsigned int* pixels, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int index = y * width + x;
    int base = pixels[index];
    int blended = color_combine(base, col, opacity);
    pixels[index] = blended;
}

//////////////////////////////
// Device Drawing Functions //
//////////////////////////////

//TODO this is defined in color.cpp too, deduplicate!
__device__ int color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0f;
    float over_opacity = geta(over_color) / 255.0f * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = roundf(final_opacity * 255.0f);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

// TODO this is also defined in Pixels.cpp, please de-dupe
__device__ void device_fill_circle(int cx, int cy, float r, int col, unsigned int* pixels, int width, int height, float opa=1.0f) {
    int i_r = (int)r;
    for (int dx = -i_r + 1; dx < i_r; dx++) {
        float sdx = (dx / r) * (dx / r);
        for (int dy = -i_r + 1; dy < i_r; dy++) {
            float sdy = (dy / r) * (dy / r);
            if (sdx + sdy < 1.0f)
                overlay_pixel(cx + dx, cy + dy, col, opa, pixels, width, height);
        }
    }
}

// TODO this is defined also in pixels.cpp, pls deduplicate!
__device__ void bresenham(int x1, int y1, int x2, int y2, int col, float opacity, int thickness, unsigned int* pixels, int width, int height) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    if(dx > 10000 || dy > 10000){
        return;
    }

    int sx = (x1 < x2) ? 1 : -1; // Direction on x axis
    int sy = (y1 < y2) ? 1 : -1; // Direction on y axis
    int err = dx - dy;

    while(true) {
        overlay_pixel(x1, y1, col, opacity, pixels, width, height);
        for(int i = 1; i < thickness; i++){
            overlay_pixel(x1 + i, y1, col, opacity, pixels, width, height);
            overlay_pixel(x1 - i, y1, col, opacity, pixels, width, height);
            overlay_pixel(x1, y1 + i, col, opacity, pixels, width, height);
            overlay_pixel(x1, y1 - i, col, opacity, pixels, width, height);
        }

        if (x1 == x2 && y1 == y2) break;

        int e2 = err * 2;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

//////////////////////////////
// Device Structures        //
//////////////////////////////

struct DevicePoint {
    double3 center;  // Projected pixel coordinates (x,y) in double format. z is unused.
    int color;
    float opacity;
    int highlight; // 0 = NORMAL, 1 = RING, 2 = BULLSEYE
    float size;
};

struct DeviceLine {
    double3 start;  // Projected pixel coordinates (x,y) in double format.
    double3 end;
    int color;
    float opacity;
};

//////////////////////////////
// Kernels                  //
//////////////////////////////

__global__ void render_points_kernel(unsigned int* pixels, int width, int height,
                                     DevicePoint* points, int num_points) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        DevicePoint p = points[idx];
        int px = (int)round(p.center.x);
        int py = (int)round(p.center.y);
        float dot_size = p.size;
        if (p.highlight == 1) { // RING
            device_fill_ellipse(px, py, dot_size * 2.0f, dot_size * 2.0f, p.color, pixels, width, height, 1.0f);
            device_fill_ellipse(px, py, dot_size * 1.5f, dot_size * 1.5f, 0xFF000000, pixels, width, height, 1.0f);
        } else if (p.highlight == 2) { // BULLSEYE
            device_fill_ellipse(px, py, dot_size * 3.0f, dot_size * 3.0f, p.color, pixels, width, height, 1.0f);
            device_fill_ellipse(px, py, dot_size * 2.5f, dot_size * 2.5f, 0xFF000000, pixels, width, height, 1.0f);
            device_fill_ellipse(px, py, dot_size * 2.0f, dot_size * 2.0f, p.color, pixels, width, height, 1.0f);
            device_fill_ellipse(px, py, dot_size * 1.5f, dot_size * 1.5f, 0xFF000000, pixels, width, height, 1.0f);
        } else {
            device_fill_ellipse(px, py, dot_size, dot_size, p.color, pixels, width, height, p.opacity);
        }
    }
}

__global__ void render_lines_kernel(unsigned int* pixels, int width, int height,
                                    DeviceLine* lines, int num_lines, int thickness) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_lines) {
        DeviceLine ln = lines[idx];
        int x0 = (int)round(ln.start.x);
        int y0 = (int)round(ln.start.y);
        int x1 = (int)round(ln.end.x);
        int y1 = (int)round(ln.end.y);
        bresenham(x0, y0, x1, y1, ln.color, ln.opacity, thickness, pixels, width, height);
    }
}

//////////////////////////////
// Host Rendering Functions //
//////////////////////////////

void cuda_render_points(unsigned int* d_pixels, int width, int height,
                          DevicePoint* d_points, int num_points) {
    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    render_points_kernel<<<numBlocks, blockSize>>>(d_pixels, width, height, d_points, num_points);
    cudaDeviceSynchronize();
}

void cuda_render_lines(unsigned int* d_pixels, int width, int height, int thickness,
                         DeviceLine* d_lines, int num_lines) {
    int blockSize = 256;
    int numBlocks = (num_lines + blockSize - 1) / blockSize;
    render_lines_kernel<<<numBlocks, blockSize>>>(d_pixels, width, height, d_lines, num_lines, thickness);
    cudaDeviceSynchronize();
}

//////////////////////////////
// Entry Points (externed)  //
//////////////////////////////

extern "C" void render_points_on_gpu(unsigned int** d_pixels, int width, int height,
                                     DevicePoint* h_points, int num_points) {
    size_t buffer_size = width * height * sizeof(unsigned int);
    cudaMalloc((void**)d_pixels, buffer_size);
    cudaMemset(*d_pixels, 0, buffer_size);

    DevicePoint* d_points;
    cudaMalloc((void**)&d_points, num_points * sizeof(DevicePoint));
    cudaMemcpy(d_points, h_points, num_points * sizeof(DevicePoint), cudaMemcpyHostToDevice);

    cuda_render_points(*d_pixels, width, height, d_points, num_points);

    cudaFree(d_points);
}

extern "C" void render_lines_on_gpu(unsigned int** d_pixels, int width, int height, int thickness,
                                    DeviceLine* h_lines, int num_lines) {
    size_t buffer_size = width * height * sizeof(unsigned int);
    cudaMalloc((void**)d_pixels, buffer_size);
    cudaMemset(*d_pixels, 0, buffer_size);

    DeviceLine* d_lines;
    cudaMalloc((void**)&d_lines, num_lines * sizeof(DeviceLine));
    cudaMemcpy(d_lines, h_lines, num_lines * sizeof(DeviceLine), cudaMemcpyHostToDevice);

    cuda_render_lines(*d_pixels, width, height, thickness, d_lines, num_lines);

    cudaFree(d_lines);
}
