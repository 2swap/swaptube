#pragma once

#include <unordered_map>
#include "scene.cpp"
#include "Connect4/c4.h"

typedef pair<Scene*, pair<int, int>> scene_with_coords;

class CompositeScene : public Scene {
public:
    Scene* createScene(const int width, const int height) override {
        return new CompositeScene(width, height);
    }

    CompositeScene(const int width, const int height) : Scene(width, height) {}

    void update_variables(const unordered_map<string, double>& _variables) override {
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
