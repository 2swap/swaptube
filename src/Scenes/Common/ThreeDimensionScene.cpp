#include "ThreeDimensionScene.h"
#include "../../IO/Writer.h"
#include "../../IO/Latex.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/CameraProjection.h"
#include "../../Core/Smoketest.h"

extern "C" {
    void render_points_on_gpu(
        uint32_t* d_pixels, const ivec2& wh,
        float geom_mean_size, float points_opacity, float points_radius_multiplier,
        Point* h_points, int num_points,
        const quat& camera_direction, const vec3& camera_pos, float fov);

    void render_lines_on_gpu(
        uint32_t* d_pixels, const ivec2& wh,
        float geom_mean_size, int thickness, float lines_opacity,
        Line* h_lines, int num_lines,
        const quat& camera_direction, const vec3& camera_pos, float fov);
}

ThreeDimensionScene::ThreeDimensionScene(const vec2& dimensions)
    : SuperScene(dimensions) {
    manager.set({
        {"fov", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "1"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"lines_opacity", "1"},
        {"points_radius_multiplier", "1"},
        {"lines_thickness_multiplier", "1"},
        {"points_opacity", "1"},
        {"point_labels_size", "1"},
    });
}

vec2 ThreeDimensionScene::coordinate_to_pixel(vec3 coordinate, float& distance) {
    const vec3 p = ::coordinate_to_pixel(coordinate, camera_direction, camera_pos,
                                         fov, get_geom_mean_size(), get_width_height());
    distance = p.z;
    return vec2(p.x, p.y);
}

void ThreeDimensionScene::set_camera_direction() {
    fov = state["fov"];
    over_w_fov = 1/(get_geom_mean_size()*fov);
    camera_direction = normalize(quat(state["q1"], state["qi"], state["qj"], state["qk"]));
    vec3 focus = vec3(state["x"], state["y"], state["z"]);
    camera_pos = focus - rotate_vector(vec3(0,0,state["d"]), conjugate(camera_direction));
}

vector<Point> ThreeDimensionScene::read_state_points() const {
    vector<Point> pts;
    for (int i = 0; state.contains("point" + to_string(i) + ".x"); i++) {
        const string base = "point" + to_string(i) + ".";
        pts.push_back(Point(vec3(state[base + "x"], state[base + "y"], state[base + "z"]),
                            0xff808080, 1, 1));
    }
    return pts;
}

void ThreeDimensionScene::draw() {
    set_camera_direction();

    if (!lines.empty() && state["lines_opacity"] > .001) {
        int thickness = static_cast<int>(state["lines_thickness_multiplier"] * get_geom_mean_size() / 640.0);
        render_lines_on_gpu(
            gpu_pix.get_ptr(),
            get_width_height(),
            get_geom_mean_size(),
            thickness,
            state["lines_opacity"],
            lines.data(),
            static_cast<int>(lines.size()),
            camera_direction,
            camera_pos,
            fov
        );
    }

    auto render_point_set = [&](Point* data, int n) {
        if (n <= 0 || state["points_opacity"] <= .001 || state["points_radius_multiplier"] <= 0.001) return;
        render_points_on_gpu(
            gpu_pix.get_ptr(),
            get_width_height(),
            get_geom_mean_size(),
            state["points_opacity"],
            state["points_radius_multiplier"],
            data,
            n,
            camera_direction,
            camera_pos,
            fov
        );
    };

    render_point_set(points.data(), static_cast<int>(points.size()));

    vector<Point> state_points = read_state_points();
    render_point_set(state_points.data(), static_cast<int>(state_points.size()));

    const float labels_size = state["point_labels_size"];
    if (labels_size > 0.001) {
        for (const auto& [id, name] : point_names) {
            const string base = "point" + to_string(id) + ".";
            if (name.empty() || !state.contains(base + "x")) continue;
            float distance;
            const vec2 pos = coordinate_to_pixel(vec3(state[base + "x"], state[base + "y"], state[base + "z"]), distance);
            if (distance <= 0) continue;
            const vec2 dim = vec2(0.2, 0.1) * get_width_height() * labels_size;
            write_text(gpu_pix.get_ptr(), gpu_pix.get_wh(), latex_color(0xffffffff, name), pos, dim, 1, 0);
        }
    }
}

void ThreeDimensionScene::add_point(const Point& p) {
    points.push_back(p);
}

void ThreeDimensionScene::add_line(const Line& l) {
    lines.push_back(l);
}

void ThreeDimensionScene::clear_lines(){ lines.clear(); }
void ThreeDimensionScene::clear_points(){ points.clear(); }
