#pragma once

#include <glm/glm.hpp>

__device__ __forceinline__ float clamp(float value, float lower, float upper){
    return fminf(fmaxf(value, lower), upper);
}
__device__ __forceinline__ float lerp(float a, float b, float w){
    return w*b+(1-w)*a;
}
__device__ __forceinline__ glm::vec2 pixel_to_point(const glm::vec2& pixel, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip(pixel.x, wh.y-1-pixel.y);
    return flip * (rx_by - lx_ty) / wh + lx_ty;
}

__device__ __forceinline__ glm::vec2 point_to_pixel(const glm::vec2& point, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip = (point - lx_ty) * wh / (rx_by - lx_ty);
    return glm::vec2(flip.x, wh.y-1-flip.y);
}

