#include "OuterBilliardsVertexFlowScene.h"
#include <algorithm>
#include <cmath>

extern "C" void draw_convex_polygon(uint32_t* d_pixels, const ivec2& wh,
                                    const vec2* h_verts, int n,
                                    uint32_t color, float opacity);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void outer_billiards_vertex_flow_render(uint32_t* d_pixels, const ivec2& wh, const VertexFlowParams& params);

OuterBilliardsVertexFlowScene::OuterBilliardsVertexFlowScene(const vec2& dimensions)
    : CoordinateScene(dimensions) {
    manager.set({
        {"table_opacity", "1"},
        {"ball_start_x", "0"},
        {"ball_start_y", "0"},
        {"ball_opacity", "1"},
        {"curvature",    "0"},
        {"flow_opacity", "1"},
        {"flow_depth",   "0"},
    });
}

std::vector<vec2> OuterBilliardsVertexFlowScene::read_vertices() {
    std::vector<vec2> verts;
    update_state();
    for (int i = 0; i < MAX_BILLIARD_VERTICES; i++) {
        const std::string x_key = "v" + std::to_string(i) + ".x";
        if (!manager.contains(x_key)) break;
        verts.push_back(vec2(state[x_key], state["v" + std::to_string(i) + ".y"]));
    }
    return verts;
}

void OuterBilliardsVertexFlowScene::draw_flow_field(const std::vector<vec2>& verts) {
    const float flow_opacity = (float)state["flow_opacity"];
    const float flow_depth   = (float)state["flow_depth"];
    if (flow_opacity < 0.01 || flow_depth <= 0.0f) return;

    const int n_fixed = (int)verts.size();
    if (n_fixed < 2 || n_fixed >= MAX_BILLIARD_VERTICES) return;

    VertexFlowParams params;
    for (int i = 0; i < n_fixed; i++) params.fixed_verts[i] = verts[i];
    params.n_fixed     = n_fixed;
    params.ball_start  = vec2(state["ball_start_x"], state["ball_start_y"]);
    params.curvature   = (float)state["curvature"];
    params.lx_ty       = vec2(state["left_x"], state["top_y"]);
    params.rx_by       = vec2(state["right_x"], state["bottom_y"]);
    params.flow_opacity= flow_opacity;
    params.flow_depth  = flow_depth;

    outer_billiards_vertex_flow_render(gpu_pix.get_ptr(), gpu_pix.get_wh(), params);
}

void OuterBilliardsVertexFlowScene::draw_ball() {
    const float opacity = (float)state["ball_opacity"];
    if (opacity < 0.01) return;

    const vec2 ball_pos(state["ball_start_x"], state["ball_start_y"]);
    const float radius_px = (float)get_geom_mean_size() / 120.0f;
    draw_circle(gpu_pix.get_ptr(), get_width_height(), point_to_pixel(ball_pos), radius_px, 0xffffffff, opacity);
}

void OuterBilliardsVertexFlowScene::draw() {
    const std::vector<vec2> verts = read_vertices();

    draw_flow_field(verts);

    if (verts.size() < 3) throw runtime_error("OuterBilliardsVertexFlowScene::draw() table has fewer than 3 vertices");

    std::vector<vec2> pixel_verts;
    pixel_verts.reserve(verts.size());
    for (const vec2& v : verts) pixel_verts.push_back(point_to_pixel(v));

    draw_convex_polygon(gpu_pix.get_ptr(), gpu_pix.get_wh(), pixel_verts.data(), (int)pixel_verts.size(), 0xff1a6b3a, (float)state["table_opacity"]);

    draw_ball();

    CoordinateScene::draw();
}
