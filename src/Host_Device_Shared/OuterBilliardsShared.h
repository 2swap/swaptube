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

HOST_DEVICE inline float curved_arcsinh(float x, float curvature) {
    const float k = -curvature;
    if (k < 1e-9f) return x;
    const float bend = sqrtf(k);
    return asinhf(bend * x) / bend;
}

HOST_DEVICE inline float curved_screen_scale(const vec2& q, float curvature) {
    const float n = curved_norm(q, curvature);
    if (n <= 0.0f) return 0.0f;
    return powf(n, 0.75f);
}

HOST_DEVICE inline float curved_closeness(const vec2& a, const vec2& b, float a_norm, float curvature) {
    const vec2 delta = a - b;
    const float flat = dot(delta, delta);
    if (curvature == 0.0f) return flat * 0.25f;

    const float nb = curved_norm(b, curvature);
    if (a_norm <= 0.0f || nb <= 0.0f) return 1e30f;

    const float cross = a.x * b.y - a.y * b.x;
    const float inner = 1.0f + curvature * dot(a, b);
    const float geo   = sqrtf(a_norm * nb);
    const float denom = 2.0f * geo * (inner + geo);
    if (denom <= 1e-30f) return 1e30f;
    return (flat + curvature * cross * cross) / denom;
}

// Positive when b is counterclockwise of a.
HOST_DEVICE inline float billiards_cross(const vec2& a, const vec2& b) { return a.x * b.y - a.y * b.x; }

// Caller guarantees p is outside the (convex, counterclockwise) table, so the
// clockwise-most vertex as seen from p is the tangent vertex, full stop - no
// need to double check by sweeping the winding again.
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

HOST_DEVICE inline float outer_billiards_singular_distance(const vec2* verts, int n, int pivot,
                                                           const vec2& p, float curvature) {
    const float norm_p = curved_norm(p, curvature);
    if (norm_p <= 0.0f) return 1e30f;
    const vec2 v = verts[pivot];
    float best = 1e30f;
    for (int i = 0; i < n; i++) {
        if (i == pivot) continue;
        const vec2 u = verts[i];

        const vec3 m(v.y - u.y, u.x - v.x, v.x * u.y - v.y * u.x);
        const float scale = m.x * m.x + m.y * m.y + curvature * m.z * m.z;
        if (scale <= 1e-20f) continue;   // coincident vertices span no line

        const vec3 w(m.x, m.y, curvature * m.z);
        const float past_v = (v.y * w.z - w.y) * p.x + (w.x - v.x * w.z) * p.y + (v.x * w.y - v.y * w.x);
        const float past_u = (u.y * w.z - w.y) * p.x + (w.x - u.x * w.z) * p.y + (u.x * w.y - u.y * w.x);
        if (past_v < 0.0f && past_u > 0.0f) continue;

        const float height = (m.x * p.x + m.y * p.y + m.z) / sqrtf(scale);
        const float d = curved_arcsinh(fabsf(height) / sqrtf(norm_p), curvature);
        if (d < best) best = d;
    }
    return best;
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
    float    island_opacity;
    int      max_period;
};

SHARED_FILE_SUFFIX
