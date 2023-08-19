#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    CompositeScene(const json& config, const json& contents, MovieWriter* writer);
    ~CompositeScene();
    Pixels query(bool& done_scene) override;
    void update_variables(const map<string, double>& _variables) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new CompositeScene(config, scene, writer);
    }

private:
    vector<scene_with_coords> scenes;
};

CompositeScene::CompositeScene(const json& config, const json& contents, MovieWriter* writer) : Scene(config, contents, writer) {
    for (auto& j : contents["subscenes"]) {
        cout << "determining subscene coordinates..." << endl;
        json config_new_resolution = config;
        config_new_resolution["width"] = config_new_resolution["width"].get<double>() * j["width"].get<double>();
        config_new_resolution["height"] = config_new_resolution["height"].get<double>() * j["height"].get<double>();
        pair<int, int> coords = make_pair(j["x"].get<double>()*config["width"].get<double>(), j["y"].get<double>()*config["height"].get<double>());

        // Pass in nullptr as the moviewriter to signal to the subscene not to write audio.
        scene_with_coords s = make_pair(create_scene_determine_type(config_new_resolution, j["subscene"], nullptr), coords);
        scenes.push_back(s);
    }
    add_audio(contents, writer);
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

Pixels CompositeScene::query(bool& done_scene) {
    done_scene = false;
    for (auto& swc : scenes){
        bool this_scene_done = false;
        Pixels p = swc.first->query(this_scene_done);
        done_scene = this_scene_done || done_scene;
        pix.copy(p, swc.second.first, swc.second.second, 1);
    }
    time++;
    return pix;
}
