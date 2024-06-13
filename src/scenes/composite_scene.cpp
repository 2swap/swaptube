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

    void update_variables(const unordered_map<string, double>& _variables) override {
        for (auto& swc : scenes) {
            swc.scenePointer->update_variables(_variables);
        }
    }

    void stage_transition(Scene* sc, double x, double y, double width, double height){
        upcoming_scenes = scenes;
        previous_scenes = scenes;
        is_transition = true;
        rendered = false;
        bool already_found = false;
        for (auto& swc : upcoming_scenes){
            if(swc.scenePointer == sc){
                assert(!already_found); // TODO currently only support one of each scenepointer
                already_found = true;
                swc.x = x; swc.y = y; swc.width = width; swc.height = height;
            }
        }
        assert(already_found); // TODO we currently assume there is a scene of this type
    }

    void post_transition(){
        scenes = upcoming_scenes;
        is_transition = false;
    }

    void interpolate(){
        double w = static_cast<double>(time)/scene_duration_frames;
        for(int i = 0; i < previous_scenes.size(); i++){
            scenes[i].x      = lerp(previous_scenes[i].x     , upcoming_scenes[i].x     , smoother2(w));
            scenes[i].y      = lerp(previous_scenes[i].y     , upcoming_scenes[i].y     , smoother2(w));
            scenes[i].width  = lerp(previous_scenes[i].width , upcoming_scenes[i].width , smoother2(w));
            scenes[i].height = lerp(previous_scenes[i].height, upcoming_scenes[i].height, smoother2(w));
        }
        rendered = false;
    }

    void render_composite(){
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
        if (is_transition) interpolate();
        if (!rendered) {
            render_composite();
            rendered = true;
        }
        done_scene = time++ >= scene_duration_frames;
        if(done_scene && is_transition) post_transition();
        p = &pix;
    }

private:
    vector<SceneWithPosition> scenes;
    vector<SceneWithPosition> previous_scenes;
    vector<SceneWithPosition> upcoming_scenes;
};
