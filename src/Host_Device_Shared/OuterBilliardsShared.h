#pragma once

#include <cstdint>
#include "vec.h"
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

const int MAX_BILLIARD_VERTICES = 14;

HOST_DEVICE inline float curved_norm(const vec2& q, float curvature) {
    return 1.0f + curvature * dot(q, q);
}

HOST_DEVICE inline bool curved_in_plane(const vec2& q, float curvature) {
    return curved_norm(q, curvature) > 1e-7f;
}

// Klein (straight-chord) <-> Poincare (conformal disk) coordinates for the same
// hyperbolic plane. The billiard dynamics all live in Klein coordinates, where
// geodesics are straight; these maps are used only to render that field on a
// Poincare disk. Both are the identity to first order at the origin and reduce
// to the identity at curvature 0, so animating curvature through 0 stays smooth.
HOST_DEVICE inline vec2 klein_to_poincare(const vec2& k, float curvature) {
    const float s = sqrtf(1.0f + curvature * dot(k, k));
    return k * (2.0f / (1.0f + s));
}
HOST_DEVICE inline vec2 poincare_to_klein(const vec2& p, float curvature) {
    return p / (1.0f - 0.25f * curvature * dot(p, p));
}

// The Poincare disk holds the whole hyperbolic plane; its exterior is not part
// of the model. (poincare_to_klein would fold exterior points back inside via a
// circle inversion, so callers must reject them first.) Always true at
// curvature 0, where the model is the entire Euclidean plane.
HOST_DEVICE inline bool in_poincare_disk(const vec2& p, float curvature) {
    return 1.0f + 0.25f * curvature * dot(p, p) > 0.0f;
}

// Positive when b is counterclockwise of a.
HOST_DEVICE inline float billiards_cross(const vec2& a, const vec2& b) { return a.x * b.y - a.y * b.x; }

HOST_DEVICE inline bool point_in_triangle(const vec2& a, const vec2& b, const vec2& c, const vec2& q) {
    const float d0 = billiards_cross(b - a, q - a);
    const float d1 = billiards_cross(c - b, q - b);
    const float d2 = billiards_cross(a - c, q - c);
    const bool has_neg = d0 < 0.0f || d1 < 0.0f || d2 < 0.0f;
    const bool has_pos = d0 > 0.0f || d1 > 0.0f || d2 > 0.0f;
    return !(has_neg && has_pos);
}

HOST_DEVICE inline bool point_in_convex_hull(const vec2* pts, int n, const vec2& q) {
    const int m = n - 1;
    const vec2 extra = pts[m];

    bool inside_polygon = true;
    for (int i = 0; i < m; i++)
        if (billiards_cross(pts[(i + 1) % m] - pts[i], q - pts[i]) < 0.0f) { inside_polygon = false; break; }
    if (inside_polygon) return true;

    for (int i = 0; i < m; i++)
        if (point_in_triangle(extra, pts[i], pts[(i + 1) % m], q)) return true;
    return false;
}

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
    float    periodicity_or_flow;
    float    flow_depth;
    float    cycle_highlight;
    float    cycle_highlight_enable;
};

struct VertexFlowParams {
    vec2  fixed_verts[MAX_BILLIARD_VERTICES];
    int   n_fixed;
    vec2  ball_start;
    float curvature;
    vec2  lx_ty, rx_by;
    float flow_opacity;
    float flow_depth;
    float black_stripes;
};

SHARED_FILE_SUFFIX
