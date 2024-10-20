#pragma once

#include <unordered_map>
#include "../Scene.cpp"

struct SceneWithPosition {
    Scene* scenePointer;
    string state_manager_name;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    void add_scene_fade_in(Scene* sc, string state_manager_name, double x = 0.5, double y = 0.5, bool micro = true){
        add_scene(sc, state_manager_name, x, y);
        state_manager.set(unordered_map<string, string> {
            {state_manager_name + ".y", to_string(y+.1)},
            {state_manager_name + ".opacity", "0"},
        });
        unordered_map<string, string> map = {
            {state_manager_name + ".y", to_string(y)},
            {state_manager_name + ".opacity", "1"},
        };
        if(micro) state_manager.microblock_transition(map);
        else      state_manager.macroblock_transition(map);
    }

    void add_scene(Scene* sc, string state_manager_name, double x = 0.5, double y = 0.5){
        sc->state_manager.set_parent(&state_manager);
        state_manager.set(unordered_map<string, string> {
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

    void change_data() override {
        for (auto& swp : scenes){
            swp.scenePointer->change_data();
        }
    }

    bool check_if_data_changed() const override {
        for (auto& swp : scenes){
            if(swp.scenePointer->check_if_data_changed()) return true;
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

    bool has_subscene_state_changed() const override {
        for (auto& swp : scenes){
            if(swp.scenePointer->check_if_state_changed()) return true;
        }
        return false;
    }

    void draw() override {
        for (auto& swp : scenes){
            Pixels* p = nullptr;
            swp.scenePointer->query(p);
            pix.overlay(*p, state[swp.state_manager_name + ".x"] * get_width () - swp.scenePointer->get_width ()/2,
                            state[swp.state_manager_name + ".y"] * get_height() - swp.scenePointer->get_height()/2, state[swp.state_manager_name + ".opacity"]);
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
        }; 
        return ret;
    }

private:
    vector<SceneWithPosition> scenes;
};
