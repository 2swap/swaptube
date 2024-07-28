#pragma once

#include "../Scene.cpp"
#include "OrbitSim.cpp"

extern "C" void render_predictions_cuda(const vector<int>& planetcolors, const vector<glm::vec3>& positions, int width, int height, glm::vec3 screen_center, float zoom, int* colors, float force_constant, float collision_threshold_squared, float drag, float tick_duration);

class OrbitScene2D : public Scene {
public:
    OrbitScene2D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), simulation(sim), predictions(width, height) {
        append_to_state_query(StateQuery{
            "tick_duration",
            "collision_threshold",
            "drag",
            "zoom",
            "screen_center_x",
            "screen_center_y",
            "screen_center_z",
            "predictions_opacity",
        });
    }

    void render_predictions() {
        vector<int> colors(w*h);
        glm::vec3 screen_center(state["screen_center_x"], state["screen_center_y"], state["screen_center_z"]);

        int num_positions = simulation->fixed_objects.size();

        vector<glm::vec3> positions(num_positions);
        vector<int> planetcolors(num_positions);
        int i = 0;
        for (const FixedObject& fo : simulation->fixed_objects) {
            positions[i] = fo.get_position(*dag);
            planetcolors[i] = fo.color;
            i++;
        }
        float collision_threshold_squared = square(state["collision_threshold"]);
        float tick_duration = state["tick_duration"];
        float drag = pow(state["drag"], tick_duration);
        float zoom = state["zoom"] * h;
        render_predictions_cuda(planetcolors, positions, w, h, screen_center, zoom, colors.data(), global_force_constant, collision_threshold_squared, drag, tick_duration);

        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            int col = colors[y * w + x];
            predictions.set_pixel(x, y, col | 0xff000000);
        }
        unsigned int opacity = state["predictions_opacity"]*255;
        unsigned int alpha = (opacity << 24) | 0x00ffffff;
        predictions.bitwise_and(alpha);
    }

    void sim_to_2d() {
        glm::vec3 screen_center(state["screen_center_x"], state["screen_center_y"], state["screen_center_z"]);
        glm::vec3 halfsize(w/2,h/2,0);
        float zoom = state["zoom"] * h;

        for (const auto& obj : simulation->mobile_objects) {
            glm::vec3 pix_position = (obj.position - screen_center) * zoom + halfsize;
            pix.fill_circle(pix_position.x, pix_position.y, w/300., obj.color);
        }
        for (const auto& obj : simulation->fixed_objects) {
            glm::vec3 pix_position = (obj.get_position(*dag) - screen_center) * zoom + halfsize;
            pix.fill_circle(pix_position.x, pix_position.y, w/100., obj.color);
            pix.fill_circle(pix_position.x, pix_position.y, w/200., OPAQUE_BLACK);
        }
    }

    // request re-rendering even if state hasn't changed
    bool does_subscene_want_to_rerender() const override { return true; }

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        simulation->iterate_physics(physics_multiplier, *dag);
        if(state["predictions_opacity"] > 0.001) {
            /*if(state != last_state)*/ render_predictions();
            pix.overwrite(predictions, 0, 0);
        }
        sim_to_2d();
    }

    int physics_multiplier = 1;

protected:
    OrbitSim* simulation;
    Pixels predictions;
};
