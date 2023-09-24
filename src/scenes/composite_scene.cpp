#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    Scene* createScene(const int width, const int height, const json& scene) override {
        return new CompositeScene(width, height, scene);
    }

    CompositeScene(const int width, const int height, const json& contents) : Scene(width, height, contents) {
        for (auto& j : contents["subscenes"]) {
            cout << "determining subscene coordinates..." << endl;
            int subscene_width = width * j["width"].get<double>();
            int subscene_height = height * j["height"].get<double>();

            pair<int, int> coords = make_pair(j["x"].get<double>()*width, j["y"].get<double>()*height);

            // Pass in nullptr as the moviewriter to signal to the subscene not to write audio.
            scene_with_coords s = make_pair(create_scene_determine_type(subscene_width, subscene_height, j["subscene"]), coords);
            scenes.push_back(s);
        }
        add_audio(contents);
    }

    void update_variables(const map<string, double>& _variables) override {
        for (auto& swc : scenes) {
            swc.first->update_variables(_variables);
        }
    }

    ~CompositeScene() {
        for (auto& swc : scenes) {
            delete swc.first;
        }
    }

    const Pixels& query(bool& done_scene) override {
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

private:
    vector<scene_with_coords> scenes;
};
