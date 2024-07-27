#pragma once

#include "../Scene.cpp"
#include "OrbitSim.cpp"

extern "C" void render_predictions_cuda(const vector<int>& planetcolors, const vector<glm::vec3>& positions, int width, int height, glm::vec3 screen_center, float zoom, int* colors, float force_constant, float collision_threshold_squared, float drag);

class OrbitScene2D : public Scene {
public:
    OrbitScene2D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), simulation(sim) {}

    void render_predictions() {
        vector<int> colors(w*h);
        glm::vec3 screen_center((*dag)["screen_center_x"], (*dag)["screen_center_y"], (*dag)["screen_center_z"]);

        int num_positions = simulation->fixed_objects.size();

        vector<glm::vec3> positions(num_positions);
        vector<int> planetcolors(num_positions);
        for (int i = 0; i < num_positions; ++i) {
            positions[i] = simulation->fixed_objects[i].get_position(*dag);
            planetcolors[i] = simulation->fixed_objects[i].color;
        }
        float force_constant = (*dag)["force_constant"];
        float collision_threshold_squared = square((*dag)["collision_threshold"]);
        float drag = (*dag)["drag"];
        float zoom = (*dag)["zoom"] * h;
        render_predictions_cuda(planetcolors, positions, w, h, screen_center, zoom, colors.data(), force_constant, collision_threshold_squared, drag);

        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            int col = colors[y * w + x];
            pix.set_pixel(x, y, col);
        }
    }

    void sim_to_2d() {
        glm::vec3 screen_center(w*.5,h*.5,0);
        float zoom = (*dag)["zoom"] * h;

        for (const auto& obj : simulation->mobile_objects) {
            glm::vec3 pix_position = obj.position * zoom + screen_center;
            pix.fill_circle(pix_position.x, pix_position.y, w/500., obj.color);
        }
        for (const auto& obj : simulation->fixed_objects) {
            glm::vec3 pix_position = obj.get_position(*dag) * zoom + screen_center;
            pix.fill_circle(pix_position.x, pix_position.y, w/300., obj.color);
        }
    }

    void query(Pixels*& p) override {
        pix.fill(TRANSPARENT_BLACK);
        simulation->iterate_physics(physics_multiplier, *dag);
        render_predictions();
        sim_to_2d();
        p=&pix;
    }

    int physics_multiplier = 1;

protected:
    OrbitSim* simulation;
};
