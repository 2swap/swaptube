#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"

#include "OrbitSim.cpp"
#include <vector>

extern "C" void render_predictions_cuda(
const std::vector<glm::vec3>& positions, // Planet data
const int width, const int height, const int depth, const glm::vec3 screen_center, const float zoom, // Geometry of query
const float force_constant, const float collision_threshold_squared, const float drag, const float tick_duration, const float eps, // Adjustable parameters
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
        float eps = state["eps"];
        int width  = round(state["wireframe_width" ]);
        int height = round(state["wireframe_height"]);
        int depth  = round(state["wireframe_depth" ]);

        glm::vec3 bounding_box(width, height, depth);
        glm::vec3 midpoint = bounding_box*0.5f;

        float boundingbox_opacity = state["boundingbox.opacity"];
        if(boundingbox_opacity > 0.001){
            for(int x = 0; x < 2; x++) for(int y = 0; y < 2; y++) for(int z = 0; z < 2; z++) {
                glm::vec3 corner1 = (glm::vec3(x,y,z)-glm::vec3(0.5f)) * bounding_box / zoom;
                glm::vec3 corner2 = (glm::vec3(.5,y,z)-glm::vec3(0.5f)) * bounding_box / zoom;
                glm::vec3 corner3 = (glm::vec3(x,.5,z)-glm::vec3(0.5f)) * bounding_box / zoom;
                glm::vec3 corner4 = (glm::vec3(x,y,.5)-glm::vec3(0.5f)) * bounding_box / zoom;
                add_line(Line(corner1, corner2, OPAQUE_WHITE, boundingbox_opacity));
                add_line(Line(corner1, corner3, OPAQUE_WHITE, boundingbox_opacity));
                add_line(Line(corner1, corner4, OPAQUE_WHITE, boundingbox_opacity));
            }
        }

        float collision_threshold_squared = square(state["collision_threshold"]);
        float tick_duration = state["tick_duration"];
        float drag = pow(state["drag"], tick_duration);

        vector<glm::vec3> planet_positions; vector<int> planet_colors; vector<float> opacities;
        simulation->get_fixed_object_data_for_cuda(planet_positions, planet_colors, opacities, state_manager);

        // Early bailout if we wont render anything
        float max_opacity = 0;
        for (float f : opacities) max_opacity = max(f, max_opacity);
        if(max_opacity < 0.001) return;

        vector<int> colors (width * height * depth);
        vector<int> times  (width * height * depth);
        vector<int> borders(width * height * depth);

        render_predictions_cuda(planet_positions, width, height, depth, glm::vec3(0.f,0,0), zoom, global_force_constant, collision_threshold_squared, drag, tick_duration, eps, colors.data(), times.data());

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

        // Draw lines between border points in a 3x3x3 cube
        float nonconverge_opacity = state["nonconverge_opacity"];
        for (int x = 0; x < width; ++x) for (int y = 0; y < height; ++y) for (int z = 0; z < depth; ++z) {
            int idx = x + (y + z * height) * width;
            int planet_number = colors[idx];
            float line_opacity = planet_number == -1 ? nonconverge_opacity : opacities[planet_number];
            int line_color     = planet_number == -1 ? OPAQUE_WHITE        : planet_colors[planet_number];
            if (borders[idx] && line_opacity > 0.001) {
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                    if (dx*9+dy*3+dz <= 0) continue; // Do not draw lines bidirectionally
                    int nx = x+dx; int ny = y+dy; int nz = z+dz;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                        int neighbor_idx = nx + (ny + nz * height) * width;
                        if (borders[neighbor_idx] && colors[neighbor_idx] == colors[idx]) {
                            glm::vec3 a = (glm::vec3( x, y, z) - midpoint) / zoom;
                            glm::vec3 b = (glm::vec3(nx,ny,nz) - midpoint) / zoom;
                            add_line(Line(a, b, line_color, line_opacity));
                        }
                    }
                }
            }
        }
    }

    const StateQuery populate_state_query() const override {
        ThreeDimensionScene::populate_state_query() + StateQuery{
            "tick_duration",
            "collision_threshold",
            "drag",
            "zoom",
            "wireframe_width",
            "wireframe_height",
            "wireframe_depth",
            "nonconverge.opacity",
            "physics_multiplier",
            "boundingbox.opacity",
            "eps",
        };
    }

    void sim_to_3d() {
        clear_points();

        for (const auto& obj : simulation->mobile_objects) add_point(Point(obj.position, obj.color, NORMAL, 1));
        for (const auto& obj : simulation->fixed_objects) add_point(Point(obj.get_position(state_manager), obj.color, RING, 1));
    }

    void mark_data_unchanged() override { simulation->mark_unchanged(); }
    void change_data() override {
        simulation->iterate_physics(round(state["physics_multiplier"]), state_manager);
    }
    bool check_if_data_changed() const override {
        return simulation->has_been_updated_since_last_scene_query();
    }
    void draw() override {
        fill_predictions_and_add_lines();
        sim_to_3d();
        ThreeDimensionScene::draw();
    }

protected:
    OrbitSim* simulation;
};


