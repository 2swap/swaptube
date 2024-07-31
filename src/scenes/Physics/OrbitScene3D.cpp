#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "OrbitSim.cpp"
#include <vector>

extern "C" void render_predictions_cuda(
const std::vector<int>& planetcolors, const std::vector<glm::vec3>& positions, // Planet data
const int width, const int height, const int depth, const glm::vec3 screen_center, const float zoom, // Geometry of query
const float force_constant, const float collision_threshold_squared, const float drag, const float tick_duration, // Adjustable parameters
int* colors, int* times // outputs
);

class OrbitScene3D : public ThreeDimensionScene {
public:
    OrbitScene3D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : ThreeDimensionScene(width, height), simulation(sim) {}

    void fill_predictions_and_add_lines() {
        clear_lines();

        // Define the resolution of the 3D grid

        float zoom = state["zoom"];
        int width  = round(state["wireframe_width" ]);
        int height = round(state["wireframe_height"]);
        int depth  = round(state["wireframe_depth" ]);
        float collision_threshold_squared = square(state["collision_threshold"]);
        float tick_duration = state["tick_duration"];
        float drag = pow(state["drag"], tick_duration);

        vector<glm::vec3> planet_positions; vector<int> planet_colors;
        simulation->get_fixed_object_data_for_cuda(planet_positions, planet_colors, *dag);

        vector<int> colors (width * height * depth);
        vector<int> times  (width * height * depth);
        vector<int> borders(width * height * depth);

        render_predictions_cuda(planet_colors, planet_positions, width, height, depth, glm::vec3(0.f,0,0), zoom, global_force_constant, collision_threshold_squared, drag, tick_duration, colors.data(), times.data());

        // Identify border points
        for (int x = 0; x < width; ++x) for (int y = 0; y < height; ++y) for (int z = 0; z < depth; ++z) {
            const int idx = x + (y + z * height) * width;
            borders[idx] = true;
            if (x != width  - 1 && colors[idx] != colors[idx + 1           ]) continue;
            if (x != 0          && colors[idx] != colors[idx - 1           ]) continue;
            if (y != height - 1 && colors[idx] != colors[idx + width       ]) continue;
            if (y != 0          && colors[idx] != colors[idx - width       ]) continue;
            if (z != depth  - 1 && colors[idx] != colors[idx + width*height]) continue;
            if (z != 0          && colors[idx] != colors[idx - width*height]) continue;
            borders[idx] = false;
        }

        glm::vec3 midpoint(width*.5f, height*.5f, depth*.5f);

        // Draw lines between border points in a 3x3x3 cube
        for (int x = 0; x < width; ++x) for (int y = 0; y < height; ++y) for (int z = 0; z < depth; ++z) {
            int idx = x + (y + z * height) * width;
            if (borders[idx] && colors[idx] == planet_colors[0]) {
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                    if (dx*9+dy*3+dz == 0) continue; // Do not draw lines bidirectionally
                    int nx = x+dx; int ny = y+dy; int nz = z+dz;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                        int neighbor_idx = nx + (ny + nz * height) * width;
                        if (borders[neighbor_idx] && colors[neighbor_idx] == colors[idx]) {
                            glm::vec3 a = (glm::vec3( x, y, z) - midpoint) / zoom;
                            glm::vec3 b = (glm::vec3(nx,ny,nz) - midpoint) / zoom;
                            add_line(Line(a, b, colors[idx], 1));
                        }
                    }
                }
            }
        }
    }

    void pre_query() override {
        ThreeDimensionScene::pre_query();
        append_to_state_query(StateQuery{
            "tick_duration",
            "collision_threshold",
            "drag",
            "zoom",
            "wireframe_width",
            "wireframe_height",
            "wireframe_depth",
            "physics_multiplier",
        });
    }

    void sim_to_3d() {
        clear_points();

        for (const auto& obj : simulation->mobile_objects) add_point(Point(obj.position, obj.color, NORMAL, obj.opacity));
        for (const auto& obj : simulation->fixed_objects) add_point(Point(obj.get_position(*dag), obj.color, RING, obj.opacity));
    }

    bool scene_requests_rerender() const override { return true; }

    void draw() override {
        simulation->iterate_physics(round(state["physics_multiplier"]), *dag);
        fill_predictions_and_add_lines();
        sim_to_3d();
        ThreeDimensionScene::draw();
    }


protected:
    OrbitSim* simulation;
};

