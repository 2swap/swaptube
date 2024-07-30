#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "OrbitSim.cpp"
#include <vector>

class OrbitScene3D : public ThreeDimensionScene {
public:
    OrbitScene3D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : ThreeDimensionScene(width, height), simulation(sim) {
        append_to_state_query(StateQuery{
            "tick_duration",
            "collision_threshold",
            "drag",
            "physics_multiplier",
        });
    }

    void fill_predictions_and_add_lines() {
        lines.clear();

        // Define the resolution of the 3D grid
        const int grid_size = 20; // Example size, adjust as needed
        vector<vector<vector<int>>> grid_colors(grid_size, vector<vector<int>>(grid_size, vector<int>(grid_size, -1)));
        vector<vector<vector<glm::vec3>>> grid_points(grid_size, vector<vector<glm::vec3>>(grid_size, vector<glm::vec3>(grid_size)));

        float step = 1.0f / (grid_size - 1);

        // Fill the grid with predicted colors
        for (int x = 0; x < grid_size; ++x) {
            for (int y = 0; y < grid_size; ++y) {
                for (int z = 0; z < grid_size; ++z) {
                    glm::vec3 pos(x * step - 0.5f,
                                  y * step - 0.5f,
                                  z * step - 0.5f);
                    grid_colors[x][y][z] = simulation->predict_fate_of_object(pos, *dag);
                    grid_points[x][y][z] = pos;
                }
            }
        }

        // Add lines where neighboring points have different colors
        for (int x = 0; x < grid_size; ++x) {
            for (int y = 0; y < grid_size; ++y) {
                for (int z = 0; z < grid_size; ++z) {
                    if (x < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x + 1][y][z]) {
                        lines.push_back(Line(grid_points[x][y][z], grid_points[x + 1][y][z], grid_colors[x][y][z], 1));
                    }
                    if (y < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x][y + 1][z]) {
                        lines.push_back(Line(grid_points[x][y][z], grid_points[x][y + 1][z], grid_colors[x][y][z], 1));
                    }
                    if (z < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x][y][z + 1]) {
                        lines.push_back(Line(grid_points[x][y][z], grid_points[x][y][z + 1], grid_colors[x][y][z], 1));
                    }
                }
            }
        }
    }

    void sim_to_3d() {
        points.clear();

        for (const auto& obj : simulation->mobile_objects) {
            points.push_back(Point(obj.position, obj.color, NORMAL, obj.opacity));
        }
        for (const auto& obj : simulation->fixed_objects) {
            points.push_back(Point(obj.get_position(*dag), obj.color, RING, obj.opacity));
        }
    }

    bool scene_requests_rerender() const override { return true; }

    void draw() override {
        simulation->iterate_physics(round(state["physics_multiplier"]), *dag);
        if(lines.size() == 0) fill_predictions_and_add_lines();
        if(points.size() == 0) sim_to_3d();
        ThreeDimensionScene::draw();
    }


protected:
    OrbitSim* simulation;
};

