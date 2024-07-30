#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "OrbitSim.cpp"
#include <vector>

extern "C" void render_predictions_cuda(const vector<int>& planetcolors, const vector<glm::vec3>& positions, int width, int height, glm::vec3 screen_center, float zoom, int* colors, int* times, float force_constant, float collision_threshold_squared, float drag, float tick_duration);

class OrbitScene3D : public ThreeDimensionScene {
public:
    OrbitScene3D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : ThreeDimensionScene(width, height), simulation(sim) { }

    void fill_predictions_and_add_lines() {
        lines.clear();

        // Define the resolution of the 3D grid
        const int grid_size = 60; // Example size, adjust as needed
        vector<vector<vector<int>>> grid_colors(grid_size, vector<vector<int>>(grid_size, vector<int>(grid_size, -1)));
        vector<vector<vector<glm::vec3>>> grid_points(grid_size, vector<vector<glm::vec3>>(grid_size, vector<glm::vec3>(grid_size)));
        vector<vector<vector<bool>>> border_points(grid_size, vector<vector<bool>>(grid_size, vector<bool>(grid_size, false)));

        float step = 1.0f / (grid_size - 1);
        float zoom = grid_size * 1.f; // Adjusted for grid size
        float collision_threshold_squared = square(state["collision_threshold"]);
        float tick_duration = state["tick_duration"];
        float drag = pow(state["drag"], tick_duration);

        int num_positions = simulation->fixed_objects.size();
        vector<glm::vec3> positions(num_positions);
        vector<int> planetcolors(num_positions);
        int i = 0;
        for (const FixedObject& fo : simulation->fixed_objects) {
            positions[i] = fo.get_position(*dag);
            planetcolors[i] = fo.color;
            i++;
        }

        // Fill the grid with predicted colors using CUDA for each slice
        for (int z = 0; z < grid_size; ++z) {
            vector<int> colors(grid_size * grid_size);
            vector<int> times(grid_size * grid_size);

            glm::vec3 slice_center(0,0,z*step-0.5f);

            render_predictions_cuda(planetcolors, positions, grid_size, grid_size, slice_center, zoom, colors.data(), times.data(), global_force_constant, collision_threshold_squared, drag, tick_duration);

            for (int y = 0; y < grid_size; ++y) {
                for (int x = 0; x < grid_size; ++x) {
                    grid_colors[x][y][z] = colors[y * grid_size + x];
                    grid_points[x][y][z] = glm::vec3(x * step - 0.5f, y * step - 0.5f, z * step - 0.5f);
                }
            }
        }

        // Identify border points
        for (int x = 0; x < grid_size; ++x) {
            for (int y = 0; y < grid_size; ++y) {
                for (int z = 0; z < grid_size; ++z) {
                    bool is_border = false;
                    if (x < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x + 1][y][z]) is_border = true;
                    if (x > 0             && grid_colors[x][y][z] != grid_colors[x - 1][y][z]) is_border = true;
                    if (y < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x][y + 1][z]) is_border = true;
                    if (y > 0             && grid_colors[x][y][z] != grid_colors[x][y - 1][z]) is_border = true;
                    if (z < grid_size - 1 && grid_colors[x][y][z] != grid_colors[x][y][z + 1]) is_border = true;
                    if (z > 0             && grid_colors[x][y][z] != grid_colors[x][y][z - 1]) is_border = true;
                    border_points[x][y][z] = is_border;
                }
            }
        }

        // Draw lines between border points in a 3x3x3 cube
        for (int x = 0; x < grid_size; ++x) {
            for (int y = 0; y < grid_size; ++y) {
                for (int z = 0; z < grid_size; ++z) {
                    if (border_points[x][y][z]) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dz = -1; dz <= 1; ++dz) {
                                    if (dx == 0 && dy == 0 && dz == 0) continue;
                                    int nx = x + dx;
                                    int ny = y + dy;
                                    int nz = z + dz;
                                    if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && nz >= 0 && nz < grid_size) {
                                        if (border_points[nx][ny][nz] && grid_colors[x][y][z] == grid_colors[nx][ny][nz]) {
                                            lines.push_back(Line(grid_points[x][y][z], grid_points[nx][ny][nz], grid_colors[x][y][z], 1));
                                        }
                                    }
                                }
                            }
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
            "physics_multiplier",
        });
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
        /*if(lines.size() == 0)*/ fill_predictions_and_add_lines();
        if(points.size() == 0) sim_to_3d();
        ThreeDimensionScene::draw();
    }


protected:
    OrbitSim* simulation;
};

