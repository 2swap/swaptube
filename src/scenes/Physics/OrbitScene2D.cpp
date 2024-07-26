#pragma once

#include "../Scene.cpp"
#include "OrbitSim.cpp"

class OrbitScene2D : public Scene {
public:
    OrbitScene2D(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height), simulation(sim) {zoom = h/2.;}

    void render_predictions(){
        glm::vec3 screen_center(w*.5,h*.5,0);
        for(int x = 0; x < w; x++) for(int y = 0; y < h; y++){
            glm::vec3 object_pos(x, y, 0);
            object_pos -= screen_center;
            object_pos /= zoom;
            int col = (simulation->predict_fate_of_object(object_pos, *dag) | 0xff000000) & 0x99ffffff;
            pix.set_pixel(x, y, col);
        }
    }

    void sim_to_2d() {
        glm::vec3 screen_center(w*.5,h*.5,0);

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

    float zoom;
    int physics_multiplier = 1;

protected:
    OrbitSim* simulation;
};
