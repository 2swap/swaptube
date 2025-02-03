#pragma once

#include "SuperScene.cpp"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

struct SceneWithPosition {
    Scene* scenePointer;
    string state_manager_name;
};

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
        SceneWithPosition swp = {sc, state_manager_name};
        scenes.push_back(swp);
    }

    void fade_out_all_scenes() {
        for (auto& swp : scenes) {
            unordered_map<string, string> map = {
                {swp.state_manager_name + ".opacity", "0"}
            };
            state_manager.microblock_transition(map);
        }
    }

    void remove_all_scenes() {
        for (auto& swp : scenes){
            swp.scenePointer->state_manager.set_parent(nullptr);
        }
        scenes.clear();
    }

    void mark_data_unchanged() override { }
    bool check_if_data_changed() const override {return false;}

    void change_data() override {
        // I think this is a noop since data will be changed when each of the subscenes is queried
    }

    bool subscene_needs_redraw() const override {
        for (auto& swp : scenes){
            swp.scenePointer->update();
            if(state[swp.state_manager_name + ".opacity"] > 0.01 && swp.scenePointer->needs_redraw()) return true;
        }
        return false;
    }

    void on_end_transition() override {
        for(const auto& swp : scenes){
            swp.scenePointer->on_end_transition();
            swp.scenePointer->state_manager.close_microblock_transitions();
            swp.scenePointer->state_manager.close_macroblock_transitions();
        }
    }

    void draw() override {
        int w = get_width();
        int h = get_height();
        for (auto& swp : scenes){
            double opa = state[swp.state_manager_name + ".opacity"];
            if(opa < 0.01) continue;
            Pixels* p = nullptr;
            swp.scenePointer->query(p);
            double pointer_opa = state[swp.state_manager_name + ".pointer_opacity"];
            int x = w*state[swp.state_manager_name + ".x"];
            int y = h*state[swp.state_manager_name + ".y"];
            if(pointer_opa > 0.01) {
                int px = w*state[swp.state_manager_name + ".pointer_x"];
                int py = h*state[swp.state_manager_name + ".pointer_y"];
                pix.bresenham(x, y, px, py, OPAQUE_WHITE, pointer_opa, h/100.);
            }
            pix.overlay(*p, x - swp.scenePointer->get_width ()/2,
                            y - swp.scenePointer->get_height()/2, opa);
        }
    }

    void remove_scene(Scene* sc) {
        sc->state_manager.set_parent(nullptr);
        scenes.erase(std::remove_if(scenes.begin(), scenes.end(),
                                    [sc](const SceneWithPosition& swp) {
                                        return swp.scenePointer == sc;
                                    }), scenes.end());
    }

    const StateQuery populate_state_query() const override {
        StateQuery ret;
        for (auto& swp : scenes){
            ret.insert(swp.state_manager_name + ".x");
            ret.insert(swp.state_manager_name + ".y");
            ret.insert(swp.state_manager_name + ".opacity");
            ret.insert(swp.state_manager_name + ".pointer_x");
            ret.insert(swp.state_manager_name + ".pointer_y");
            ret.insert(swp.state_manager_name + ".pointer_opacity");
        };
        return ret;
    }

    void send_to_front(const string& state_manager_name) {
        // Find the scene by its state_manager_name
        auto it = std::find_if(scenes.begin(), scenes.end(),
                               [&state_manager_name](const SceneWithPosition& swp) {
                                   return swp.state_manager_name == state_manager_name;
                               });
        if (it == scenes.end()) {
            throw std::runtime_error("Scene with specified name not found");
        }

        // Move the scene to the "front" (end of the vector, since reverse order)
        SceneWithPosition swp = *it;
        scenes.erase(it);
        scenes.push_back(swp);
    }

    void send_to_back(const string& state_manager_name) {
        // Find the scene by its state_manager_name
        auto it = std::find_if(scenes.begin(), scenes.end(),
                               [&state_manager_name](const SceneWithPosition& swp) {
                                   return swp.state_manager_name == state_manager_name;
                               });
        if (it == scenes.end()) {
            throw std::runtime_error("Scene with specified name not found");
        }

        // Move the scene to the "back" (beginning of the vector, since reverse order)
        SceneWithPosition swp = *it;
        scenes.erase(it);
        scenes.insert(scenes.begin(), swp);
    }

private:
    vector<SceneWithPosition> scenes;
};
