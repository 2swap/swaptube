#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    CompositeScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new CompositeScene(config, scene, writer);
    }

private:
    vector<scene_with_coords> scenes;
};

CompositeScene::CompositeScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents, writer) {
    for (auto& j : contents["subscenes"]) {
        json config_new_resolution = config;
        config_new_resolution["width"] = j["width"];
        config_new_resolution["height"] = j["height"];

        // TODO: Pass in null (or something) as the moviewriter to signal to the subscene not to write audio.
        scene_with_coords s = make_pair(create_scene_determine_type(config_new_resolution, j["subscene"], writer), make_pair(j["x"], j["y"]));
        scenes.push_back(s);
    }
    frontload_audio(contents, writer);
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
