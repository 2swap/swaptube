#pragma once

#include "SuperScene.cpp"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

class CompositeScene : public SuperScene {
public:
    CompositeScene(const double width = 1, const double height = 1)
        : SuperScene(width, height) {}

    void add_scene_fade_in(Scene* sc, string state_manager_name, double x = 0.5, double y = 0.5, bool micro = true){
        add_scene(sc, state_manager_name, x, y);
        state_manager.set(unordered_map<string, string> {
            {state_manager_name + ".opacity", "0"},
        });
        unordered_map<string, string> map = {
            {state_manager_name + ".opacity", "1"},
        };
        if(micro) state_manager.microblock_transition(map);
        else      state_manager.macroblock_transition(map);
    }

    void add_scene(Scene* sc, string state_manager_name, double x = 0.5, double y = 0.5){
        sc->state_manager.set_parent(&state_manager);
        state_manager.set(unordered_map<string, string> {
            {state_manager_name + ".pointer_x", to_string(x)},
            {state_manager_name + ".pointer_y", to_string(y)},
            {state_manager_name + ".pointer_opacity", "0"},
            {state_manager_name + ".x", to_string(x)},
            {state_manager_name + ".y", to_string(y)},
            {state_manager_name + ".opacity", "1"},
        });
        NamedSubscene subscene = {sc, state_manager_name};
        add_subscene_check_dupe(subscene);
    }

    void draw() override {
        int w = get_width();
        int h = get_height();
        for (auto& subscene : subscenes){
            double opa = state[subscene.state_manager_name + ".opacity"];
            if(opa < 0.01) continue;
            Pixels* p = nullptr;
            subscene.ptr->query(p);
            double pointer_opa = state[subscene.state_manager_name + ".pointer_opacity"];
            int x = w*state[subscene.state_manager_name + ".x"];
            int y = h*state[subscene.state_manager_name + ".y"];
            if(pointer_opa > 0.01) {
                int px = w*state[subscene.state_manager_name + ".pointer_x"];
                int py = h*state[subscene.state_manager_name + ".pointer_y"];
                pix.bresenham(x, y, px, py, OPAQUE_WHITE, pointer_opa, h/100.);
            }
            pix.overlay(*p, x - subscene.ptr->get_width ()/2,
                            y - subscene.ptr->get_height()/2, opa);
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery ret;
        for (auto& subscene : subscenes){
            ret.insert(subscene.state_manager_name + ".x");
            ret.insert(subscene.state_manager_name + ".y");
            ret.insert(subscene.state_manager_name + ".opacity");
            ret.insert(subscene.state_manager_name + ".pointer_x");
            ret.insert(subscene.state_manager_name + ".pointer_y");
            ret.insert(subscene.state_manager_name + ".pointer_opacity");
        };
        return ret;
    }

    void send_to_front(const string& state_manager_name) {
        // Find the scene by its state_manager_name
        auto it = std::find_if(subscenes.begin(), subscenes.end(),
                               [&state_manager_name](const NamedSubscene& subscene) {
                                   return subscene.state_manager_name == state_manager_name;
                               });
        if (it == subscenes.end()) {
            throw runtime_error("Scene with specified name not found");
        }

        // Move the scene to the "front" (end of the vector, since reverse order)
        NamedSubscene subscene = *it;
        subscenes.erase(it);
        subscenes.push_back(subscene);
    }

    void send_to_back(const string& state_manager_name) {
        // Find the scene by its state_manager_name
        auto it = std::find_if(subscenes.begin(), subscenes.end(),
                               [&state_manager_name](const NamedSubscene& subscene) {
                                   return subscene.state_manager_name == state_manager_name;
                               });
        if (it == subscenes.end()) {
            throw runtime_error("Scene with specified name not found");
        }

        // Move the scene to the "back" (beginning of the vector, since reverse order)
        NamedSubscene subscene = *it;
        subscenes.erase(it);
        subscenes.insert(subscenes.begin(), subscene);
    }
};
