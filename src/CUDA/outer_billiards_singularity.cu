#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"
#include "color.cuh"

__device__ __forceinline__ float line_coverage(float distance, float half_width, float world_per_pixel) {
    return Cuda::clamp((half_width + 0.5f * world_per_pixel - distance) / world_per_pixel, 0.0f, 1.0f);
}

__device__ __forceinline__ float curved_arcsinh(float x, float curvature) {
    const float k = -curvature;
    if (k < 1e-9f) return x;
    const float bend = sqrtf(k);
    return asinhf(bend * x) / bend;
}

__device__ __forceinline__ float curved_screen_scale(const Cuda::vec2& q, float curvature) {
    const float n = Cuda::curved_norm(q, curvature);
    if (n <= 0.0f) return 0.0f;
    return powf(n, 0.75f);
}

__device__ __forceinline__ float curved_closeness(const Cuda::vec2& a, const Cuda::vec2& b, float a_norm, float curvature) {
    const Cuda::vec2 delta = a - b;
    const float flat = dot(delta, delta);
    if (curvature == 0.0f) return flat * 0.25f;

    const float nb = Cuda::curved_norm(b, curvature);
    if (a_norm <= 0.0f || nb <= 0.0f) return 1e30f;

    const float cross = a.x * b.y - a.y * b.x;
    const float inner = 1.0f + curvature * dot(a, b);
    const float geo   = sqrtf(a_norm * nb);
    const float denom = 2.0f * geo * (inner + geo);
    if (denom <= 1e-30f) return 1e30f;
    return (flat + curvature * cross * cross) / denom;
}

__device__ __forceinline__ float curved_distance(const Cuda::vec2& a, const Cuda::vec2& b, float curvature) {
    const float half_sq = curved_closeness(a, b, Cuda::curved_norm(a, curvature), curvature);
    if (half_sq >= 1e29f) return 1e30f;
    return 2.0f * curved_arcsinh(sqrtf(half_sq > 0.0f ? half_sq : 0.0f), curvature);
}

struct SingularRay {
    Cuda::vec3 line;
    Cuda::vec3 cap;
    Cuda::vec2 origin;
};

__device__ __forceinline__ SingularRay outer_billiards_build_ray(const Cuda::vec2* verts, int n, int i, float curvature) {
    SingularRay ray;
    const Cuda::vec2 v = verts[i];             // where the ray starts
    const Cuda::vec2 u = verts[(i + 1) % n];   // the far end of the side it extends
    ray.origin = v;

    const Cuda::vec3 m(v.y - u.y, u.x - v.x, v.x * u.y - v.y * u.x);

    float scale = m.x * m.x + m.y * m.y + curvature * m.z * m.z;
    scale = (scale > 1e-20f) ? 1.0f / sqrtf(scale) : 0.0f;
    ray.line = Cuda::vec3(m.x * scale, m.y * scale, m.z * scale);

    const Cuda::vec3 w(m.x, m.y, curvature * m.z);
    ray.cap = Cuda::vec3(v.y * w.z - w.y,
                          w.x - v.x * w.z,
                          v.x * w.y - v.y * w.x);
    return ray;
}

__device__ __forceinline__ float outer_billiards_ray_distance(const SingularRay& ray, const Cuda::vec2& q,
                                                              float norm_q, float curvature) {
    if (ray.cap.x * q.x + ray.cap.y * q.y + ray.cap.z >= 0.0f) {
        const float height = ray.line.x * q.x + ray.line.y * q.y + ray.line.z;
        return curved_arcsinh(fabsf(height) / sqrtf(norm_q), curvature);
    }
    return curved_distance(q, ray.origin, curvature);
}

__device__ __forceinline__ float outer_billiards_pivot_distance(const Cuda::vec2* verts, int n, int pivot,
                                                                const Cuda::vec2& q, float norm_q, float curvature) {
    const int prev = (pivot - 1 + n) % n;
    const SingularRay left  = outer_billiards_build_ray(verts, n, prev,  curvature);
    const SingularRay right = outer_billiards_build_ray(verts, n, pivot, curvature);
    const float dl = outer_billiards_ray_distance(left,  q, norm_q, curvature);
    const float dr = outer_billiards_ray_distance(right, q, norm_q, curvature);
    return fminf(dl, dr);
}

static constexpr float SINGULARITY_GLOW         = 2.f;

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
    const float half_width = 0.6f * wpp;
    const float halo       = fmaxf(4.0f * half_width, 1e-20f);
    const bool  want_glow  = SINGULARITY_GLOW > 0.001f;

    const float to_screen = curved_screen_scale(start, params.curvature);

    const bool want_islands = params.island_opacity > 0.01 && params.max_period > 1;
    const float start_norm = Cuda::curved_norm(start, params.curvature);

    float web_intensity = 0.0f;
    int   web_hop        = 0;

    float nearest = 1e30f;

    float closest_return = 1e30f;
    int   return_hop     = 0;

    Cuda::vec2 p = start;
    for (int k = 0; k < steps; k++) {
        const int pivot = Cuda::outer_billiards_tangent_vertex(params.verts, params.n, p);

        if (k < web_steps || k < island_steps) {
            const float norm_p = Cuda::curved_norm(p, params.curvature);
            const float d = (norm_p > 0.0f
                ? outer_billiards_pivot_distance(params.verts, params.n, pivot, p, norm_p, params.curvature)
                : 1e30f) * to_screen;
            if (d < nearest) nearest = d;

            if (k < web_steps) {
                const float weight = Cuda::clamp(params.depth - (float)k, 0.0f, 1.0f);

                if (weight > 0.0f) {
                    float intensity = line_coverage(d, half_width, wpp);
                    if (want_glow) {
                        const float soft = SINGULARITY_GLOW * __expf(-d / halo);
                        intensity = 1.0f - (1.0f - intensity) * (1.0f - soft);
                    }
                    intensity *= weight;
                    if (intensity > web_intensity) { web_intensity = intensity; web_hop = k; }
                }
            }
        }
        p = Cuda::outer_billiards_reflect(params.verts[pivot], p, params.curvature);
        if (want_islands && k < params.max_period) {
            const float back = curved_closeness(start, p, start_norm, params.curvature);
            if (back < closest_return) { closest_return = back; return_hop = k + 1; }
        }
    }

    const int index = py * wh.x + px;
    uint32_t out = 0x00000000;
    if (return_hop > 0) {
        out = Cuda::rainbow(return_hop*.03, 255 * params.island_opacity);
    }
    if (web_intensity > 0.0f) {
        uint32_t web_color = params.line_color;
        if (params.singularity_rainbow > 0.001f) {
            const uint32_t rainbow_color = Cuda::rainbow(web_hop*.05, 255);
            web_color = Cuda::colorlerp(params.line_color, rainbow_color, params.singularity_rainbow);
        }
        out = Cuda::color_combine(out, web_color, Cuda::clamp(web_intensity * params.web_opacity, 0.0f, 1.0f));
    }
    pixels[index] = out;
}

extern "C" void outer_billiards_singularity_render(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::SingularityGraphParams& params)
{
    if (params.n < 3 || wh.x <= 0 || wh.y <= 0) return;

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
