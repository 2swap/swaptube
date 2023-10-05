#pragma once

#include <unordered_map>
#include "scene.cpp"
#include "Connect4/c4.h"

struct SceneWithPosition {
    Scene* scenePointer;
    double x, y;
    double scale;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width, const int height) : Scene(width, height) {}
    CompositeScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void add_scene(Scene* sc, double x, double y, double scale){
        SceneWithPosition swp = {sc, x, y, scale};
        scenes.push_back(swp);
    }

    void update_variables(const unordered_map<string, double>& _variables) override {
        for (auto& swc : scenes) {
            swc.scenePointer->update_variables(_variables);
        }
    }

    Pixels* query(bool& done_scene) override {
        for (auto& swc : scenes){
            bool this_scene_done = false;
            Pixels* p = swc.scenePointer->query(this_scene_done);
            pix.copy_and_scale_bilinear(*p, swc.x * w, swc.y * h, swc.scale);

            done_scene &= this_scene_done;
        }
        done_scene = time++>=scene_duration_frames;
        return &pix;
    }

private:
    vector<SceneWithPosition> scenes;
};
