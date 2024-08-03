#pragma once

#include "../Scene.cpp"
#include "OrbitSim.cpp"

extern "C" void render_predictions_cuda(
const std::vector<glm::vec3>& positions, // Planet data
const int width, const int height, const int depth, const glm::vec3 screen_center, const float zoom, // Geometry of query
const float force_constant, const float collision_threshold_squared, const float drag, const float tick_duration, // Adjustable parameters
int* colors, int* times // outputs
);

class OrbitScene2D : public Scene {
public:
    OrbitScene2D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), simulation(sim) {
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
        float opacity = state["point_path.opacity"];
        for(int i = 0; i < 10000; i++){
            glm::vec3 last_pos = pos;
            int dont_care_which_planet;
            bool doneyet = simulation->get_next_step(pos, vel, dont_care_which_planet, *dag);

            glm::vec3 screen_center(state["screen_center_x"], state["screen_center_y"], 0);
            glm::vec3 halfsize(w/2.,h/2.,0.f);
            float zoom = state["zoom"] * h;
            glm::vec3 last_pixel = (last_pos - screen_center) * zoom + halfsize;
            glm::vec3 this_pixel = (pos      - screen_center) * zoom + halfsize;

            pix.bresenham(last_pixel.x, last_pixel.y, this_pixel.x, this_pixel.y, OPAQUE_WHITE, opacity, 3);
            if(doneyet) return;
        }
    }

    void render_predictions() {
        vector<int> colors(w*h);
        vector<int> times(w*h);
        glm::vec3 screen_center(state["screen_center_x"], state["screen_center_y"], state["screen_center_z"]);

        vector<glm::vec3> planet_positions; vector<int> planet_colors; vector<float> opacities;
        simulation->get_fixed_object_data_for_cuda(planet_positions, planet_colors, opacities, *dag);

        float collision_threshold_squared = square(state["collision_threshold"]);
        float tick_duration = state["tick_duration"];
        float drag = pow(state["drag"], tick_duration);
        float zoom = state["zoom"] * h;
        render_predictions_cuda(planet_positions, w, h, 1 /*2d, depth is 1*/, screen_center, zoom, global_force_constant, collision_threshold_squared, drag, tick_duration, colors.data(), times.data());

        unsigned int opacity = state["predictions_opacity"]*255;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            int planet_id = colors[y * w + x];
            int col = (planet_id == -1? OPAQUE_WHITE : planet_colors[planet_id]) & 0x00ffffff;
            unsigned int time = static_cast<unsigned int>(opacity/* / (times[y * w + x]/300.+1)*/) << 24;
            pix.set_pixel(x, y, col | time);
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
        if(state["predictions_opacity"] > 0.001) {
            /*if(state != last_state)*/ render_predictions();
        }
        if(state["point_path.opacity"] > 0.001) {
            /*if(state != last_state)*/ render_point_path();
        }
        sim_to_2d();
    }

protected:
    OrbitSim* simulation;
};
