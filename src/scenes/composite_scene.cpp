#pragma once

#include <unordered_map>
#include "scene.cpp"

struct SceneWithPosition {
    Scene* scenePointer;
    double x, y;
    double width, height;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width, const int height) : Scene(width, height) {}
    CompositeScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void add_scene(Scene* sc, double x, double y, double width, double height){
        if(x<0||y<0||width<0||height<0||x>1||y>1||width>1||height>1)
            failout("Added scene with coords outside the range 0 to 1.");

        SceneWithPosition swp = {sc, x, y, width, height};
        scenes.push_back(swp);
    }

    void render_composite(){
        pix.fill(BLACK);
        for (auto& swc : scenes){
            int  width_int = static_cast<int>(swc.width  * w);
            int height_int = static_cast<int>(swc.height * h);
            if(swc.scenePointer->w != width_int || swc.scenePointer->h != height_int){
                swc.scenePointer->resize(width_int, height_int);
            }
            Pixels* p = nullptr;
            swc.scenePointer->query(p);
            pix.copy(*p, swc.x * w, swc.y * h, 1);
        }
    }

    void query(bool& done_scene, Pixels*& p) override {
        render_composite();
        p = &pix;
    }

private:
    vector<SceneWithPosition> scenes;
};
