#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"
#include "color.cuh"

__device__ __forceinline__ float line_coverage(float distance, float half_width, float world_per_pixel) {
    return Cuda::clamp((half_width + 0.5f * world_per_pixel - distance) / world_per_pixel, 0.0f, 1.0f);
}

__global__ void singularity_graph_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::SingularityGraphParams params,
    const int steps, const int web_steps, const int island_steps)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::vec2 start = Cuda::pixel_to_point_in_screen(
        Cuda::vec2(px, py), params.lx_ty, params.rx_by, Cuda::vec2(wh.x, wh.y));

    const int pivot = Cuda::outer_billiards_pivot(params.verts, params.n, start, params.curvature);
    if (pivot < 0) return;

    const float wpp        = params.world_per_pixel;
    const float half_width = fmaxf(params.line_width, 0.0f) * 0.5f * wpp;
    const float halo       = fmaxf(4.0f * half_width, 1e-20f);
    const bool  want_glow  = params.glow > 0.001f;

    const float to_screen = Cuda::curved_screen_scale(start, params.curvature);

    const bool want_islands = params.island_opacity > 0.01 && params.max_period > 1;
    const float start_norm = Cuda::curved_norm(start, params.curvature);

    float web_intensity = 0.0f;

    float nearest = 1e30f;

    float closest_return = 1e30f;
    int   return_hop     = 0;

    Cuda::vec2 p = start;
    for (int k = 0; k < steps; k++) {
        if (k < web_steps || k < island_steps) {
            const float d = Cuda::outer_billiards_singular_distance(params.rays, params.ray_count, p, params.curvature) * to_screen;
            if (d < nearest) nearest = d;

            if (k < web_steps) {
                const float weight = Cuda::clamp(params.depth - (float)k, 0.0f, 1.0f);

                if (weight > 0.0f) {
                    float intensity = line_coverage(d, half_width, wpp);
                    if (want_glow) {
                        const float soft = params.glow * __expf(-d / halo);
                        intensity = 1.0f - (1.0f - intensity) * (1.0f - soft);
                    }
                    intensity *= weight;
                    if (intensity > web_intensity) web_intensity = intensity;
                }
            }
        }
        p = Cuda::outer_billiards_hop(params.verts, params.n, p, params.curvature);
        if (want_islands && k < params.max_period) {
            const float back = Cuda::curved_closeness(start, p, start_norm, params.curvature);
            if (back < closest_return) { closest_return = back; return_hop = k + 1; }
        }
    }

    const int index = py * wh.x + px;
    uint32_t out = pixels[index];
    if (return_hop > 0) {
        const float fill = 1.0f - line_coverage(nearest, half_width, wpp);
        if (fill > 0.0f) {
            out = Cuda::color_combine(out, Cuda::rainbow(__log2f((float)return_hop) / params.period_octaves),
                                      fill * params.island_opacity);
        }
    }
    if (web_intensity > 0.0f) {
        out = Cuda::color_combine(out, params.line_color, Cuda::clamp(web_intensity * params.web_opacity, 0.0f, 1.0f));
    }
    pixels[index] = out;
}

extern "C" void outer_billiards_singularity_render(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::SingularityGraphParams& params)
{
    if (params.n < 3 || params.ray_count < 3 || wh.x <= 0 || wh.y <= 0) return;

    const bool want_web = params.web_opacity > 0.01 && params.depth > 0.0f;
    const bool want_islands = params.island_opacity > 0.01 && params.max_period > 1 && params.depth > 0.0f;
    if (!want_web && !want_islands) return;

    const int web_steps    = want_web ? (int)ceilf(params.depth) : 0;
    const int island_steps = want_islands ? params.island_depth : 0;
    int steps = web_steps;
    if (island_steps > steps) steps = island_steps;
    if (want_islands && params.max_period > steps) steps = params.max_period;
    if (steps <= 0) return;

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    singularity_graph_kernel<<<grid, block>>>(d_pixels, wh, params, steps, web_steps, island_steps);
    cudaDeviceSynchronize();
}
