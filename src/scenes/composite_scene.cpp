#pragma once

#include <unordered_map>
#include "scene.cpp"
#include "Connect4/c4.h"

struct SceneWithPosition {
    Scene* scenePointer;
    int x, y;
    double scale;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width, const int height) : Scene(width, height) {}
    CompositeScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void add_scene(Scene* sc, int x, int y, double scale){
        SceneWithPosition swp = {sc, x, y, scale};
        scenes.push_back(swp);
    }

    void update_variables(const unordered_map<string, double>& _variables) override {
        for (auto& swc : scenes) {
            swc.scenePointer->update_variables(_variables);
        }
    }

    Pixels* query(bool& done_scene) override {
        done_scene = time >= scene_duration_frames;
        for (auto& swc : scenes){
            bool this_scene_done = false;
            Pixels* p = swc.scenePointer->query(this_scene_done);
            pix.copy_and_scale_bilinear(*p, swc.x, swc.y, swc.scale);

            done_scene &= this_scene_done;
        }
        time++;
        return &pix;
    }

private:
    vector<SceneWithPosition> scenes;
};
