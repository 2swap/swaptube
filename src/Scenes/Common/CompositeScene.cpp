#pragma once

#include <unordered_map>
#include "../Scene.cpp"

struct SceneWithPosition {
    Scene* scenePointer;
    string state_manager_name;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height) {}

    void add_scene(Scene* sc, string state_manager_name, double x, double y, double w, double h){
        sc->state_manager.set_parentstate_manager;
        const unordered_map<string, string> equations{
            {state_manager_name + ".x", to_string(x)},
            {state_manager_name + ".y", to_string(y)},
            {state_manager_name + ".w", to_string(w)},
            {state_manager_name + ".h", to_string(h)},
        };
        state_manager.add_equations(equations);
        SceneWithPosition swp = {sc, state_manager_name};
        scenes.push_back(swp);
    }

    bool update_data_objects_check_if_changed() override {
        for (auto& swp : scenes){
            if(swp.scenePointer->update_data_objects_check_if_changed()) return true;
        }
        return false;
    }

    bool has_subscene_state_changed() const override {
        for (auto& swp : scenes){
            if(swp.scenePointer->has_subscene_state_changed()) return true;
        }
        return false;
    }

    void draw() override {
        pix.fill(TRANSPARENT_BLACK);
        for (auto& swp : scenes){
            int  width_int = state[swp.state_manager_name + ".w"] * w;
            int height_int = state[swp.state_manager_name + ".h"] * h;
            if(swp.scenePointer->w != width_int || swp.scenePointer->h != height_int){
                swp.scenePointer->resize(width_int, height_int);
            }
            Pixels* p = nullptr;
            swp.scenePointer->query(p);
            pix.overlay(*p, state[swp.state_manager_name + ".x"] * w, state[swp.state_manager_name + ".y"] * h);
        }
    }

    void remove_scene(Scene* sc) {
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
            ret.insert(swp.state_manager_name + ".w");
            ret.insert(swp.state_manager_name + ".h");
        }; 
        return ret;
    }

private:
    vector<SceneWithPosition> scenes;
};
