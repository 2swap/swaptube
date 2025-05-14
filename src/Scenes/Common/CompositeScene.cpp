#pragma once

#include "SuperScene.cpp"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

class CompositeScene : public SuperScene {
public:
    CompositeScene(const double width = 1, const double height = 1)
        : SuperScene(width, height) {}

    void add_scene_fade_in(shared_ptr<Scene> sc, string state_manager_name, double x = 0.5, double y = 0.5, bool micro = true){
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

    void add_scene(shared_ptr<Scene> sc, string state_manager_name, double x = 0.5, double y = 0.5){
        state_manager.set({
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
            pix.overlay(*p, x - subscene.second->get_width ()/2,
                            y - subscene.second->get_height()/2, opa);
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery ret = SuperScene::populate_state_query();
        for (auto& subscene : subscenes){
            ret.insert(subscene.first + ".x");
            ret.insert(subscene.first + ".y");
        };
        return ret;
    }
};
