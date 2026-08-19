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

HOST_DEVICE inline float curved_distance(const vec2& a, const vec2& b, float curvature) {
    const float half_sq = curved_closeness(a, b, curved_norm(a, curvature), curvature);
    if (half_sq >= 1e29f) return 1e30f;
    return 2.0f * curved_arcsinh(sqrtf(half_sq > 0.0f ? half_sq : 0.0f), curvature);
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

HOST_DEVICE inline vec2 outer_billiards_hop(const vec2* verts, int n, const vec2& p, float curvature) {
    return outer_billiards_reflect(verts[outer_billiards_tangent_vertex(verts, n, p)], p, curvature);
}

HOST_DEVICE inline vec2 outer_billiards_ray_origin(const vec2* verts, int n, int i) { return verts[i]; }

struct SingularRay {
    vec3 line;
    vec3 cap;
    vec2 origin;
};

HOST_DEVICE inline SingularRay outer_billiards_build_ray(const vec2* verts, int n, int i, float curvature) {
    SingularRay ray;
    const vec2 v = outer_billiards_ray_origin(verts, n, i);   // where the ray starts
    const vec2 u = verts[(i + 1) % n];                        // the far end of the side it extends
    ray.origin = v;

    const vec3 m(v.y - u.y, u.x - v.x, v.x * u.y - v.y * u.x);

    float scale = m.x * m.x + m.y * m.y + curvature * m.z * m.z;
    scale = (scale > 1e-20f) ? 1.0f / sqrtf(scale) : 0.0f;
    ray.line = vec3(m.x * scale, m.y * scale, m.z * scale);

    const vec3 w(m.x, m.y, curvature * m.z);
    ray.cap = vec3(v.y * w.z - w.y,
                   w.x - v.x * w.z,
                   v.x * w.y - v.y * w.x);
    return ray;
}

HOST_DEVICE inline float outer_billiards_ray_distance(const SingularRay& ray, const vec2& q,
                                                      float norm_q, float curvature) {
    if (ray.cap.x * q.x + ray.cap.y * q.y + ray.cap.z >= 0.0f) {
        const float height = ray.line.x * q.x + ray.line.y * q.y + ray.line.z;
        return curved_arcsinh(fabsf(height) / sqrtf(norm_q), curvature);
    }
    return curved_distance(q, ray.origin, curvature);
}

HOST_DEVICE inline float outer_billiards_singular_distance(const SingularRay* rays, int count, const vec2& q,
                                                           float curvature) {
    const float norm_q = curved_norm(q, curvature);
    if (norm_q <= 0.0f) return 1e30f;
    float best = 1e30f;
    for (int i = 0; i < count; i++) {
        const float d = outer_billiards_ray_distance(rays[i], q, norm_q, curvature);
        if (d < best) best = d;
    }
    return best;
}

struct SingularityGraphParams {
    vec2        verts[MAX_BILLIARD_VERTICES];   // counterclockwise and convex
    SingularRay rays[MAX_BILLIARD_VERTICES];
    int         n;
    int         ray_count;
    float       curvature;   // 0=euclidean, negative=hyperbolic
    vec2  lx_ty, rx_by;
    float world_per_pixel;
    float    web_opacity;
    float    depth;
    uint32_t line_color;
    float    island_opacity;
    int      max_period;
    int      island_depth;
};

SHARED_FILE_SUFFIX
