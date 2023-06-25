#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    CompositeScene(const json& config, const json& contents, MovieWriter* writer);
    ~CompositeScene();
    Pixels query(int& frames_left) override;
    void update_variables(const map<string, double>& _variables) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new CompositeScene(config, scene, writer);
    }

private:
    vector<scene_with_coords> scenes;
};

CompositeScene::CompositeScene(const json& config, const json& contents, MovieWriter* writer) : Scene(config, contents, writer) {
    for (auto& j : contents["subscenes"]) {
        json config_new_resolution = config;
        config_new_resolution["width"] = config_new_resolution["width"].get<double>() * j["width"].get<double>();
        config_new_resolution["height"] = config_new_resolution["height"].get<double>() * j["height"].get<double>();

        json contents_with_duration = j["subscene"];
        contents_with_duration["duration_seconds"] = contents["duration_seconds"];

        pair<int, int> coords = make_pair(j["x"].get<double>()*config["width"].get<double>(), j["y"].get<double>()*config["height"].get<double>());

        // Pass in nullptr as the moviewriter to signal to the subscene not to write audio.
        scene_with_coords s = make_pair(create_scene_determine_type(config_new_resolution, contents_with_duration, nullptr), coords);
        scenes.push_back(s);
    }
    frontload_audio(contents, writer);
}

void CompositeScene::update_variables(const map<string, double>& _variables) {
    for (auto& swc : scenes) {
        swc.first->update_variables(_variables);
    }
}

CompositeScene::~CompositeScene() {
    for (auto& swc : scenes) {
        delete swc.first;
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
