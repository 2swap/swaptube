#pragma once

#include <unordered_map>
#include "../Scene.cpp"

struct SceneWithPosition {
    Scene* scenePointer;
    string dag_name;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {}

    void add_scene(Scene* sc, string dag_name, double x, double y, double w, double h, bool subjugate_dag){
        if(subjugate_dag) sc->dag = dag;
        dag->add_equations(unordered_map<string, string>{
            {dag_name + ".x", to_string(x)},
            {dag_name + ".y", to_string(y)},
            {dag_name + ".w", to_string(w)},
            {dag_name + ".h", to_string(h)},
        });
        SceneWithPosition swp = {sc, dag_name};
        scenes.push_back(swp);
    }

    void render_composite(){
        pix.fill(TRANSPARENT_BLACK);
        for (auto& swc : scenes){
            int  width_int = (*dag)[swc.dag_name + ".w"] * w;
            int height_int = (*dag)[swc.dag_name + ".h"] * h;
            if(swc.scenePointer->w != width_int || swc.scenePointer->h != height_int){
                swc.scenePointer->resize(width_int, height_int);
            }
            Pixels* p = nullptr;
            swc.scenePointer->query(p);
            pix.overlay(*p, (*dag)[swc.dag_name + ".x"] * w, (*dag)[swc.dag_name + ".y"] * h);
        }
    }

    void remove_scene(Scene* sc) {
        scenes.erase(std::remove_if(scenes.begin(), scenes.end(),
                                    [sc](const SceneWithPosition& swp) {
                                        return swp.scenePointer == sc;
                                    }), scenes.end());
    }

    void query(Pixels*& p) override {
        render_composite();
        p = &pix;
    }

private:
    vector<SceneWithPosition> scenes;
};
