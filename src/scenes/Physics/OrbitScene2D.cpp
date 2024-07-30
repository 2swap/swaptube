#pragma once

#include "../Scene.cpp"
#include "OrbitSim.cpp"

extern "C" void render_predictions_cuda(const vector<int>& planetcolors, const vector<glm::vec3>& positions, int width, int height, glm::vec3 screen_center, float zoom, int* colors, int* times, float force_constant, float collision_threshold_squared, float drag, float tick_duration);

class OrbitScene2D : public Scene {
public:
    OrbitScene2D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), simulation(sim), predictions(width, height) {
        append_to_state_query(StateQuery{
            "point_path.x",
            "point_path.y",
            "point_path.opacity",
            "tick_duration",
            "collision_threshold",
            "drag",
            "zoom",
            "screen_center_x",
            "screen_center_y",
            "screen_center_z",
            "predictions_opacity",
            "physics_multiplier",
        });
    }

    bool scene_requests_rerender() const override {
        return true;
    }

    void render_point_path(){
        glm::vec3 pos(state["point_path.x"], state["point_path.y"], 0);
        glm::vec3 vel(0.f,0.f,0.f);
        for(int i = 0; i < 10000; i++){
            glm::vec3 last_pos = pos;
            int dont_care_which_planet;
            bool doneyet = simulation->get_next_step(pos, vel, dont_care_which_planet, *dag);

            glm::vec3 screen_center(state["screen_center_x"], state["screen_center_y"], 0);
            glm::vec3 halfsize(w/2.,h/2.,0.f);
            float zoom = state["zoom"] * h;
            glm::vec3 last_pixel = (last_pos - screen_center) * zoom + halfsize;
            glm::vec3 this_pixel = (pos      - screen_center) * zoom + halfsize;

            pix.bresenham(last_pixel.x, last_pixel.y, this_pixel.x, this_pixel.y, OPAQUE_WHITE, 3);
            if(doneyet) return;
        }
    }

    void render_predictions() {
        vector<int> colors(w*h);
        vector<int> times(w*h);
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
        render_predictions_cuda(planetcolors, positions, w, h, screen_center, zoom, colors.data(), times.data(), global_force_constant, collision_threshold_squared, drag, tick_duration);

        unsigned int opacity = state["predictions_opacity"]*255;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            int col = colors[y * w + x] & 0x00ffffff;
            unsigned int time = static_cast<unsigned int>(opacity/* / (times[y * w + x]/300.+1)*/) << 24;
            predictions.set_pixel(x, y, col | time);
        }
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

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        simulation->iterate_physics(state["physics_multiplier"], *dag);
        if(state["point_path.opacity"] > 0.001) {
            /*if(state != last_state)*/ render_point_path();
            unsigned int alpha = state["point_path.opacity"] * 255;
            unsigned int point_path_opacity = alpha << 24;
            pix.bitwise_and(point_path_opacity | 0x00ffffff);
        }
        if(state["predictions_opacity"] > 0.001) {
            /*if(state != last_state)*/ render_predictions();
            pix.overwrite(predictions, 0, 0);
        }
        sim_to_2d();
    }

protected:
    OrbitSim* simulation;
    Pixels predictions;
};
