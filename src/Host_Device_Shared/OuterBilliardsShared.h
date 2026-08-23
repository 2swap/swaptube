#pragma once

#include <cstdint>
#include "vec.h"
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

const int MAX_BILLIARD_VERTICES = 10;

HOST_DEVICE inline float curved_norm(const vec2& q, float curvature) {
    return 1.0f + curvature * dot(q, q);
}

HOST_DEVICE inline bool curved_in_plane(const vec2& q, float curvature) {
    return curved_norm(q, curvature) > 1e-7f;
}

// Positive when b is counterclockwise of a.
HOST_DEVICE inline float billiards_cross(const vec2& a, const vec2& b) { return a.x * b.y - a.y * b.x; }

// Caller guarantees p is outside the (convex, counterclockwise) table
HOST_DEVICE inline int outer_billiards_tangent_vertex(const vec2* verts, int n, const vec2& p) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        const vec2 a = verts[best] - p;
        const vec2 b = verts[i] - p;
        const float turn = billiards_cross(a, b);
        if (turn < 0.0f) best = i;
        else if (turn == 0.0f && dot(b, b) > dot(a, a)) best = i;
    }
    return best;
}

HOST_DEVICE inline int outer_billiards_pivot(const vec2* verts, int n, const vec2& p, float curvature) {
    if (n < 2 || !curved_in_plane(p, curvature)) return -1;
    return outer_billiards_tangent_vertex(verts, n, p);
}

HOST_DEVICE inline vec2 outer_billiards_reflect(const vec2& pivot, const vec2& p, float curvature) {
    const float nv = curved_norm(pivot, curvature);
    if (nv <= 1e-9f) return p;   // the pivot is not in the plane; nothing sensible to do
    const float a = 2.0f * (curvature * dot(p, pivot) + 1.0f) / nv;
    const float denom = a - 1.0f;
    // Only reachable outside the plane, where an isometry has nowhere to send p.
    if (denom > -1e-9f && denom < 1e-9f) return p;
    return (pivot * a - p) / denom;
}

struct SingularityGraphParams {
    vec2        verts[MAX_BILLIARD_VERTICES];   // counterclockwise and convex
    int         n;
    float       curvature;   // 0=euclidean, negative=hyperbolic
    vec2  lx_ty, rx_by;
    float world_per_pixel;
    float    web_opacity;
    float    depth;
    uint32_t line_color;
    float    singularity_rainbow;
    float    island_opacity;
    int      max_period;
    int      island_depth;
    float    periodicity_or_flow;   // 0=periodicity (island coloring), >0.5=flow (color by final position)
    float    flow_depth;            // iteration count (possibly fractional) at which to sample flow position
};

SHARED_FILE_SUFFIX
