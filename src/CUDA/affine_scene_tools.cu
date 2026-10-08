#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/Color.h"
#include "color.cuh"
#include "common_graphics.cuh"

__device__ float vecnorm(Cuda::vec2 v) {
    return v.x * v.x + v.y * v.y;
}



__global__ void draw_point_kernel(uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 point, float opacity, float sqthickness, uint32_t color, Cuda::vec4 matrix, int style) {
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y) {
        return;
    }
    int pixel_index = pos.y * wh.x + pos.x;

    Cuda::vec2 diff = pos - point * wh;
    Cuda::vec2 transformed(matrix.x * diff.x + matrix.y * diff.y, matrix.z * diff.x + matrix.w * diff.y);
    float sqdist = vecnorm(transformed);
    float sqreldist = sqdist / sqthickness;
    float pixel_opacity = opacity * ((style==0)*(1 - sqreldist*sqreldist) + (style==1)*(sqreldist<=1) + (style==2)*fmaxf(0,64*(sqreldist-0.75)*(1-sqreldist)));
    if (pixel_opacity > 0) {
        pixels[pixel_index] = (max(0xff000000 & pixels[pixel_index], (uint32_t)(pixel_opacity * 255) << 24)) | (0x00ffffff & Cuda::color_combine(pixels[pixel_index], color, pixel_opacity));
        //pixels[pixel_index] = Cuda::color_combine(pixels[pixel_index], color, opacity * (1 - sqdist*sqdist / (sqthickness*sqthickness)));
    }
}

extern "C" void cuda_draw_point(uint32_t* d_pixels, Cuda::ivec2 wh, Cuda::vec2 pos, float opacity, float thickness, uint32_t color, Cuda::vec4 matrix, int style) {
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    draw_point_kernel<<<gridSize, blockSize>>>(d_pixels, wh, pos, opacity, thickness*thickness, color, matrix, style);
}



__global__ void draw_line_kernel(uint32_t* pixels, Cuda::ivec2 wh, Cuda::vec2 point1, Cuda::vec2 point2, float opacity, float thickness, uint32_t color1, uint32_t color2, int style) {
    Cuda::ivec2 pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= wh.x || pos.y >= wh.y) {
        return;
    }
    int pixel_index = pos.y * wh.x + pos.x;
    float sqthickness = thickness*thickness;
    float arrowhead_length = 5*thickness;

    Cuda::vec2 diff1 = pos - point1 * wh;
    Cuda::vec2 diff2 = pos - point2 * wh;
    Cuda::vec2 base = (point2 - point1) * wh;
    float det = diff1.x * diff2.y - diff1.y * diff2.x;
    float sqbase_pixels = vecnorm(base);
    float base_pixels = sqrtf(sqbase_pixels);
    float alongpos = (diff1.x * base.x + diff1.y * base.y) / sqbase_pixels;
    float dist1 = vecnorm(diff1);
    float sqdist = fminf(dist1 + (0 <= alongpos && alongpos <= 1) * fminf(0, det*det / sqbase_pixels - dist1), vecnorm(diff2));
    float sqreldist = sqdist / sqthickness;
    float pixel_opacity = opacity * ((style==0)*(1 - sqreldist) + (style==1)*(1 - sqreldist/(1 + fmaxf(0, fminf(
      (1-alongpos)*base_pixels/arrowhead_length*5.0f,
      ((alongpos-1)*base_pixels/arrowhead_length+1)*45+5.0f
    )))));
    if (pixel_opacity > 0) {
        pixels[pixel_index] = (max(0xff000000 & pixels[pixel_index], (uint32_t)(pixel_opacity * 255) << 24)) | (0x00ffffff & Cuda::color_combine(pixels[pixel_index], Cuda::colorlerp(color1, color2, Cuda::clamp(alongpos, 0, 1)), pixel_opacity));
    }
}

extern "C" void cuda_draw_line(uint32_t* d_pixels, Cuda::ivec2 wh, Cuda::vec2 pos1, Cuda::vec2 pos2, float opacity, float thickness, uint32_t color1, uint32_t color2, int style) {
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    draw_line_kernel<<<gridSize, blockSize>>>(d_pixels, wh, pos1, pos2, opacity, thickness, color1, color2, style);
}