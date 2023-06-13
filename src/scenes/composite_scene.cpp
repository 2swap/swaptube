#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    CompositeScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene) override {
        return new CompositeScene(config, scene);
    }

private:
    vector<scene_with_coords> scenes;
};

CompositeScene::CompositeScene(const json& config, const json& contents) : Scene(config, contents) {
    for (auto& j : contents["subscenes"]) {
        json config_new_resolution = config;
        config_new_resolution["width"] = j["width"];
        config_new_resolution["height"] = j["height"];
        scene_with_coords s = make_pair(create_scene_determine_type(config_new_resolution, j["subscene"]), make_pair(j["x"], j["y"]));
        scenes.push_back(s);
    }
}

Pixels CompositeScene::query(int& frames_left) {
    frames_left = -1;
    for (auto& swc : scenes){
        int this_scene_frames_left = 0;
        Pixels p = swc.first->query(this_scene_frames_left);
        frames_left = max(this_scene_frames_left, frames_left);
        pix.copy(p, swc.second.first, swc.second.second, 1);
    }
    time++;
    return pix;
}
