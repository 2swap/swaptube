#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"
#include "color.cuh"

__global__ void vertex_flow_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::VertexFlowParams params,
    const int steps)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::vec2 free_vertex = Cuda::pixel_to_point_in_screen(
        Cuda::vec2(px, py), params.lx_ty, params.rx_by, Cuda::vec2(wh.x, wh.y));

    Cuda::vec2 verts[Cuda::MAX_BILLIARD_VERTICES];
    for (int i = 0; i < params.n_fixed; i++) verts[i] = params.fixed_verts[i];
    const int n = params.n_fixed + 1;
    verts[params.n_fixed] = free_vertex;

    Cuda::vec2 p = params.ball_start;

    const int flow_floor = (int)floorf(params.flow_depth);
    const int flow_ceil  = (int)ceilf(params.flow_depth);
    const float flow_frac = params.flow_depth - (float)flow_floor;
    Cuda::vec2 flow_p_floor = p;
    Cuda::vec2 flow_p_ceil  = p;

    for (int k = 0; k < steps; k++) {
        const int pivot = Cuda::outer_billiards_pivot(verts, n, p, params.curvature);
        if (pivot < 0) return;
        p = Cuda::outer_billiards_reflect(verts[pivot], p, params.curvature);
        const int n_reflects = k + 1;
        if (n_reflects == flow_floor) flow_p_floor = p;
        if (n_reflects == flow_ceil)  flow_p_ceil  = p;
    }

    const float flow_length_floor = length(flow_p_floor);
    const float flow_length_ceil  = length(flow_p_ceil);
    const float flow_angle_floor  = atan2(flow_p_floor.y, flow_p_floor.x);
          float flow_angle_ceil   = atan2(flow_p_ceil.y, flow_p_ceil.x);
    if (flow_angle_ceil < flow_angle_floor) flow_angle_ceil += 2.0f * 3.1415f;
    const float flow_length_interp = Cuda::lerp(flow_length_floor, flow_length_ceil, flow_frac);
          float flow_angle_interp  = Cuda::lerp(flow_angle_floor, flow_angle_ceil, flow_frac);
    const float flow_angle_mod = fmodf(flow_angle_interp, 2.0f * 3.1415f);
    const Cuda::vec2 flow_p(flow_length_interp * cos(flow_angle_mod), flow_length_interp * sin(flow_angle_mod));

    const float magnitude = length(flow_p);
    const float atan_length = atan(magnitude / 3.0f) / (3.1415f / 2.0f);
    const float angle = atan2(flow_p.y, flow_p.x);
    const float flow_l = 0.8f / (1.0f + .01f * magnitude * magnitude);
    const float bounds_mult = 0.3f;
    const float flow_a = atan_length * sin(angle) * bounds_mult;
    const float flow_b = atan_length * cos(angle) * bounds_mult;

    const int index = py * wh.x + px;
    pixels[index] = Cuda::OKLABtoRGB(255 * params.flow_opacity, flow_l, flow_a, flow_b);
}

extern "C" void outer_billiards_vertex_flow_render(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::VertexFlowParams& params)
{
    if (params.n_fixed < 2 || wh.x <= 0 || wh.y <= 0) return;
    if (params.flow_opacity <= 0.01f || params.flow_depth <= 0.0f) return;

    const int steps = (int)ceilf(params.flow_depth);
    if (steps <= 0) return;

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    vertex_flow_kernel<<<grid, block>>>(d_pixels, wh, params, steps);
    cudaDeviceSynchronize();
}
