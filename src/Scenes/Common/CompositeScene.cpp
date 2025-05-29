#pragma once

#include "SuperScene.cpp"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

extern "C" void cuda_overlay(
    unsigned int* h_background, const int bw, const int bh,
    unsigned int* h_foreground, const int fw, const int fh,
    const int dx, const int dy,
    const float opacity);

class CompositeScene : public SuperScene {
public:
    CompositeScene(const double width = 1, const double height = 1)
        : SuperScene(width, height) {}

    void add_scene_fade_in(shared_ptr<Scene> sc, string state_manager_name, double x = 0.5, double y = 0.5, bool micro = true, double opa=1){
        add_scene(sc, state_manager_name, x, y);
        const string key = state_manager_name + ".opacity";
        state_manager.set({{key, "0"}});
        StateSet map = {{key, to_string(opa)}};
        if(micro) state_manager.microblock_transition(map);
        else      state_manager.macroblock_transition(map);
    }

    void add_scene(shared_ptr<Scene> sc, string state_manager_name, double x = 0.5, double y = 0.5){
        state_manager.set({
            {state_manager_name + ".pointer_x", to_string(x)},
            {state_manager_name + ".pointer_y", to_string(y)},
            {state_manager_name + ".pointer_opacity", "0"},
            {state_manager_name + ".x", to_string(x)},
            {state_manager_name + ".y", to_string(y)},
        });
        add_subscene_check_dupe(state_manager_name, sc);
    }

    void draw() override {
        int w = get_width();
        int h = get_height();
        for (auto& subscene : subscenes){
            double opa = state[subscene.first + ".opacity"];
            if(opa < 0.01) continue;
            Pixels* p = nullptr;
            subscene.second->query(p);
            int x = w*state[subscene.first + ".x"];
            int y = h*state[subscene.first + ".y"];
            double pointer_opa = state[subscene.first + ".pointer_opacity"];
            if(pointer_opa > 0.01) {
                int px = w*state[subscene.first + ".pointer_x"];
                int py = h*state[subscene.first + ".pointer_y"];
                pix.bresenham(x, y, px, py, OPAQUE_WHITE, pointer_opa, h/100.);
            }

            cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                           p->pixels.data(), p->w, p->h,
                           x - subscene.second->get_width ()/2, y - subscene.second->get_height()/2, opa);
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery ret = SuperScene::populate_state_query();
        for (auto& subscene : subscenes){
            ret.insert(subscene.first + ".x");
            ret.insert(subscene.first + ".y");
            ret.insert(subscene.first + ".pointer_x");
            ret.insert(subscene.first + ".pointer_y");
            ret.insert(subscene.first + ".pointer_opacity");
        };
        return ret;
    }
};
