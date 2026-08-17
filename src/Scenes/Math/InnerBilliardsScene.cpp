#include "InnerBilliardsScene.h"
#include <cmath>

extern "C" void draw_convex_polygon(uint32_t* d_pixels, const ivec2& wh,
                                    const vec2* h_verts, int n,
                                    uint32_t color, float opacity);
extern "C" void cuda_render_path_from_host(uint32_t* d_pixels, const ivec2& wh,
                                           const vec2* h_path, int path_length,
                                           const vec2& lx_ty, const vec2& rx_by,
                                           uint32_t color, float opacity, float thickness, bool closed);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);

static vector<vec2> build_ball_path(const vec2& start, float angle, float distance, const vector<vec2>& verts) {
    vector<vec2> path{start};
    if (distance <= 0.0f || verts.size() < 3) return path;

    const int n = verts.size();

    auto nearest_wall = [&](const vec2& p, const vec2& d, float& t_out, vec2& normal_out) {
        float best_t = -1.0f;
        for (int i = 0; i < n; i++) {
            const vec2 a = verts[i], b = verts[(i + 1) % n];
            const vec2 e = b - a;
            const float denom = d.x * e.y - d.y * e.x;
            if (fabs(denom) < 1e-9f) continue;
            const vec2 c = a - p;
            const float t = (c.x * e.y - c.y * e.x) / denom;
            const float u = (c.x * d.y - c.y * d.x) / denom;
            if (t > 1e-4f && u >= 0.0f && u <= 1.0f && (best_t < 0.0f || t < best_t)) {
                best_t = t;
                normal_out = normalize(vec2(-e.y, e.x));
            }
        }
        t_out = best_t;
        return best_t >= 0.0f;
    };

    vec2 pos = start;
    vec2 dir(cos(angle), sin(angle));
    float remaining = distance;

    for (int bounce = 0; bounce < 256 && remaining > .001; bounce++) {
        float t; vec2 normal;
        if (!nearest_wall(pos, dir, t, normal) || t >= remaining) {
            pos = pos + dir * remaining;
            path.push_back(pos);
            break;
        }

        pos = pos + dir * t;
        path.push_back(pos);
        remaining -= t;

        bool at_corner = false;
        for (const vec2& v : verts) {
            if (length(pos - v) < .001) { at_corner = true; break; }
        }
        dir = at_corner ? -dir : dir - normal * (2.0f * dot(dir, normal));

        pos = pos + dir * .0001; // nudge off the wall so the next pass doesn't re-hit it immediately
    }
    return path;
}

InnerBilliardsScene::InnerBilliardsScene(const vec2& dimensions)
    : CoordinateScene(dimensions) {
    manager.set({
        {"ball_start_x", "0"},
        {"ball_start_y", "0"},
        {"ball_angle",   "0"},
        {"path_opacity", "1"},
        {"path_length",  "0"},
        {"ball_distance","0"},
        {"cue_opacity",  "1"},
        {"table_opacity","1"},
    });
}

void InnerBilliardsScene::draw_trail(const vector<vec2>& verts) {
    if (verts.size() < 3) return;

    const float opacity = state["path_opacity"];
    if (opacity < 0.01) return;

    const vec2 start(state["ball_start_x"], state["ball_start_y"]);
    const vector<vec2> path = build_ball_path(start, state["ball_angle"], state["path_length"], verts);
    if (path.size() < 2) return;

    const float thickness = get_geom_mean_size() / 800.0f;
    cuda_render_path_from_host(gpu_pix.get_ptr(), get_width_height(),
                               path.data(), path.size(),
                               vec2(state["left_x"], state["top_y"]),
                               vec2(state["right_x"], state["bottom_y"]),
                               0xffffffff, opacity, thickness, false);
}

void InnerBilliardsScene::draw_ball(const vector<vec2>& verts) {
    if (verts.size() < 3) return;

    const vec2 start(state["ball_start_x"], state["ball_start_y"]);
    const vec2 ball_pos = build_ball_path(start, state["ball_angle"], state["ball_distance"], verts).back();

    const float radius_px = get_geom_mean_size() / 120.0;
    draw_circle(gpu_pix.get_ptr(), get_width_height(), point_to_pixel(ball_pos), radius_px, 0xffffffff);
}

void InnerBilliardsScene::draw_cue_stick() {
    const float    CUE_LENGTH         = 3.f;
    const float    CUE_FRONT_WIDTH    = 0.05f;
    const float    CUE_BACK_WIDTH     = 0.1f;
    const float    CUE_TIP_FRACTION = .03f;
    const float    CUE_TAN_FRACTION = .58f;
    const float    CUE_BUTT_FRACTION = .96f;
    const uint32_t CUE_WHITE = 0xffeeeeee;
    const uint32_t CUE_TAN   = 0xffd2a679;
    const uint32_t CUE_BROWN = 0xff6c4a31;
    const uint32_t CUE_BLACK = 0xff111111;

    const float opacity = state["cue_opacity"];
    if (opacity < 0.01) return;

    const vec2 start(state["ball_start_x"], state["ball_start_y"]);
    const float angle = state["ball_angle"];
    const float ball_distance = state["ball_distance"];

    const vec2 back = -vec2(cos(angle), sin(angle));
    const float pullback = ball_distance < 0.0f ? sqrt(-ball_distance+.3)-sqrt(.3) : 0.0f;
    const vec2 tip      = start + back * pullback;
    const vec2 boundary1= tip + back * (CUE_LENGTH * CUE_TIP_FRACTION);
    const vec2 boundary2= tip + back * (CUE_LENGTH * CUE_TAN_FRACTION);
    const vec2 boundary3= tip + back * (CUE_LENGTH * CUE_BUTT_FRACTION);
    const vec2 butt     = tip + back * CUE_LENGTH;
    const vec2 perp     = vec2(back.y, -back.x);

    // Parallelogram midpoints
    const float boundary1_width = CUE_FRONT_WIDTH + (CUE_BACK_WIDTH - CUE_FRONT_WIDTH) * CUE_TIP_FRACTION;
    const float boundary2_width = CUE_FRONT_WIDTH + (CUE_BACK_WIDTH - CUE_FRONT_WIDTH) * CUE_TAN_FRACTION;
    const float boundary3_width = CUE_FRONT_WIDTH + (CUE_BACK_WIDTH - CUE_FRONT_WIDTH) * CUE_BUTT_FRACTION;

    auto draw_segment = [&](const vec2& near_end, float near_width, const vec2& far_end, float far_width, uint32_t color) {
        const vec2 near_side = perp * (near_width * 0.5f);
        const vec2 far_side  = perp * (far_width * 0.5f);
        vector<vec2> quad{near_end - near_side, near_end + near_side, far_end + far_side, far_end - far_side};
        for (vec2& p : quad) p = point_to_pixel(p);
        draw_convex_polygon(gpu_pix.get_ptr(), gpu_pix.get_wh(), quad.data(), quad.size(), color, opacity);
    };

    draw_segment(tip, CUE_FRONT_WIDTH, boundary1, boundary1_width, CUE_WHITE);
    draw_segment(boundary1, boundary1_width, boundary2, boundary2_width, CUE_TAN);
    draw_segment(boundary2, boundary2_width, boundary3, boundary3_width, CUE_BROWN);
    draw_segment(boundary3, boundary3_width, butt, CUE_BACK_WIDTH, CUE_BLACK);
}

void InnerBilliardsScene::draw() {
    vector<vec2> verts;
    int i = 0;
    while (true) {
        const string x_key = "v" + to_string(i) + ".x";
        if (!manager.contains(x_key)) break;
        verts.push_back(vec2(state[x_key], state["v" + to_string(i) + ".y"]));
        i++;
    }

    vector<vec2> pixel_verts;
    for (const vec2& v : verts) pixel_verts.push_back(point_to_pixel(v));
    draw_convex_polygon(gpu_pix.get_ptr(), gpu_pix.get_wh(), pixel_verts.data(), pixel_verts.size(), 0xff1a6b3a, state["table_opacity"]);

    draw_trail(verts);
    draw_ball(verts);
    draw_cue_stick();
}
