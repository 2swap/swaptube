#include "OuterBilliardsScene.h"
#include <algorithm>
#include <cmath>

extern "C" void draw_convex_polygon(uint32_t* d_pixels, const ivec2& wh,
                                    const vec2* h_verts, int n,
                                    uint32_t color, float opacity);
extern "C" void cuda_render_path_from_host(uint32_t* d_pixels, const ivec2& wh,
                                           const vec2* h_path, int path_length,
                                           const vec2& lx_ty, const vec2& rx_by,
                                           uint32_t color, float opacity, float thickness, bool closed);
extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void outer_billiards_singularity_render(uint32_t* d_pixels, const ivec2& wh, const SingularityGraphParams& params);
extern "C" void cuda_fill_pixels(uint32_t* d_pixels, const ivec2& wh, uint32_t color);

OuterBilliardsScene::OuterBilliardsScene(const vec2& dimensions)
    : CoordinateScene(dimensions) {
    manager.set({
        {"table_opacity", "1"},
        {"ball_start_x", "0"},
        {"ball_start_y", "0"},
        {"ball_distance","0"},
        {"ball_opacity", "1"},
        {"path_length",  "0"},
        {"path_opacity", "1"},
        {"curvature",        "0"},
        {"singularity_opacity",       "0"},
        {"singularity_depth",         "0"},
        {"singularity_rainbow",       "0"},
        {"island_opacity",      "0"},
        {"periodicity_or_flow", "0"},
        {"flow_depth",            "0"},
    });
}

static const float ISLAND_BOUNDARY_DEPTH = 3.0f;
static const int   MAX_ISLAND_DEPTH = 2000;

float OuterBilliardsScene::world_per_pixel() {
    const int height = get_height();
    if (height <= 0) return 1.0f;
    return (float)(state["bottom_y"] - state["top_y"]) / (float)height;
}

std::vector<vec2> OuterBilliardsScene::read_vertices() {
    std::vector<vec2> verts;
    for (int i = 0; i < MAX_BILLIARD_VERTICES; i++) {
        const std::string x_key = "v" + std::to_string(i) + ".x";
        if (!manager.contains(x_key)) break;
        verts.push_back(vec2(state[x_key], state["v" + std::to_string(i) + ".y"]));
    }
    return verts;
}

std::vector<vec2> OuterBilliardsScene::orbit(const vec2& start, int steps, const std::vector<vec2>& verts, float curvature) {
    std::vector<vec2> path{start};
    if (steps <= 0) return path;

    path.reserve(steps + 1);
    vec2 p = start;
    for (int i = 0; i < steps; i++) {
        const int pivot = outer_billiards_pivot(verts.data(), (int)verts.size(), p, curvature);
        if (pivot < 0) break;   // the orbit reached a point the map cannot continue from
        p = outer_billiards_reflect(verts[pivot], p, curvature);
        path.push_back(p);
    }
    return path;
}

void OuterBilliardsScene::draw_singularity_graph(const std::vector<vec2>& verts) {
    const float web_opacity    = (float)state["singularity_opacity"];
    const float island_opacity = (float)state["island_opacity"];
    const float depth          = (float)state["singularity_depth"];

    const int max_period = 100;

    const bool wants_web     = web_opacity    > 0.01 && depth > 0.0f;
    const bool wants_islands = island_opacity > 0.01 && max_period > 1 && depth > 0.0f;
    if (!wants_web && !wants_islands) return;

    const float curvature = (float)state["curvature"];
    const int n = (int)verts.size();
    if (n < 3 || n > MAX_BILLIARD_VERTICES) return;

    const float wpp = world_per_pixel();

    SingularityGraphParams params;
    for (int i = 0; i < n; i++) params.verts[i] = verts[i];
    params.n = n;
    params.curvature = curvature;
    params.lx_ty = vec2(state["left_x"], state["top_y"]);
    params.rx_by = vec2(state["right_x"], state["bottom_y"]);
    params.world_per_pixel = wpp;

    params.web_opacity    = wants_web ? web_opacity : 0.0f;
    params.depth          = depth;
    params.line_color     = 0xffffffff;
    params.singularity_rainbow = (float)state["singularity_rainbow"];

    params.island_opacity = wants_islands ? island_opacity : 0.0f;
    params.max_period     = max_period;
    params.island_depth   = std::min((int)std::ceil(depth * ISLAND_BOUNDARY_DEPTH), MAX_ISLAND_DEPTH);
    params.periodicity_or_flow = (float)state["periodicity_or_flow"];
    params.flow_depth          = (float)state["flow_depth"];

    outer_billiards_singularity_render(gpu_pix.get_ptr(), gpu_pix.get_wh(), params);
}

std::vector<vec2> OuterBilliardsScene::build_orbit_path(double count) {
    const vec2 start(state["ball_start_x"], state["ball_start_y"]);

    if (!(count > 0.0)) return {start};

    const int whole = (int)std::floor(count);
    const float partial = (float)(count - whole);
    const bool wants_partial = partial > 1e-4f;

    std::vector<vec2> path = orbit(start, whole + (wants_partial ? 1 : 0), read_vertices(), (float)state["curvature"]);

    if (wants_partial && (int)path.size() == whole + 2) {
        path.back() = veclerp(path[path.size() - 2], path.back(), partial);
    }
    return path;
}

void OuterBilliardsScene::draw_orbit(float thickness) {
    const float opacity = (float)state["path_opacity"];
    if (opacity >= 0.01) {
        const std::vector<vec2> path = build_orbit_path((double)state["path_length"]);
        if (path.size() >= 2) {
            cuda_render_path_from_host(gpu_pix.get_ptr(), get_width_height(),
                                       path.data(), (int)path.size(),
                                       vec2(state["left_x"], state["top_y"]),
                                       vec2(state["right_x"], state["bottom_y"]),
                                       0xffffffff, opacity, thickness, false);
            cuda_render_path_from_host(gpu_pix.get_ptr(), get_width_height(),
                                       path.data(), (int)path.size(),
                                       vec2(state["left_x"], state["top_y"]),
                                       vec2(state["right_x"], state["bottom_y"]),
                                       0xffffffff, opacity*.2, thickness+1, false);
        }
    }

    const float ball_opacity = (float)state["ball_opacity"];
    if (ball_opacity < 0.01) return;

    const vec2 ball_pos = build_orbit_path((double)state["ball_distance"]).back();
    const float radius_px = (float)get_geom_mean_size() / 120.0f;
    draw_circle(gpu_pix.get_ptr(), get_width_height(), point_to_pixel(ball_pos), radius_px, 0xffffffff, ball_opacity);
}

void OuterBilliardsScene::add_dummy_point() {
    const int n = (int)read_vertices().size();

    const std::string v0 = "v0";
    const std::string vn = "v" + std::to_string(n-1);

    manager.set("v" + std::to_string(n) + ".x", "<" + v0 + ".x> <" + vn + ".x> + 2 /");
    manager.set("v" + std::to_string(n) + ".y", "<" + v0 + ".y> <" + vn + ".y> + 2 /");
}

void OuterBilliardsScene::draw() {
    const std::vector<vec2> verts = read_vertices();

    draw_singularity_graph(verts);

    const float thickness  = get_geom_mean_size() / 640.0f;

    if (verts.size() < 3) throw runtime_error("OuterBilliardsScene::draw() table has fewer than 3 vertices");

    std::vector<vec2> pixel_verts;
    pixel_verts.reserve(verts.size());
    for (const vec2& v : verts) pixel_verts.push_back(point_to_pixel(v));

    draw_convex_polygon(gpu_pix.get_ptr(), gpu_pix.get_wh(), pixel_verts.data(), (int)pixel_verts.size(), 0xff1a6b3a, (float)state["table_opacity"]);

    draw_orbit(thickness);

    CoordinateScene::draw();
}
