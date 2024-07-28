#pragma once

#include <unordered_map>
#include "../Scene.cpp"

struct SceneWithPosition {
    Scene* scenePointer;
    string dag_name;
};

class CompositeScene : public Scene {
public:
    CompositeScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : Scene(width, height) {}

    void add_scene(Scene* sc, string dag_name, double x, double y, double w, double h, bool subjugate_dag){
        if(subjugate_dag) {
            sc->dag = dag;
        }
        const unordered_map<string, string> equations{
            {dag_name + ".x", to_string(x)},
            {dag_name + ".y", to_string(y)},
            {dag_name + ".w", to_string(w)},
            {dag_name + ".h", to_string(h)},
        };
        dag->add_equations(equations);
        SceneWithPosition swp = {sc, dag_name};
        scenes.push_back(swp);
    }

    void render_composite(){
        pix.fill(TRANSPARENT_BLACK);
        for (auto& swc : scenes){
            int  width_int = state[swc.dag_name + ".w"] * w;
            int height_int = state[swc.dag_name + ".h"] * h;
            if(swc.scenePointer->w != width_int || swc.scenePointer->h != height_int){
                swc.scenePointer->resize(width_int, height_int);
            }
            Pixels* p = nullptr;
            swc.scenePointer->query(p);
            pix.overlay(*p, state[swc.dag_name + ".x"] * w, state[swc.dag_name + ".y"] * h);
        }
    }

    void remove_scene(Scene* sc) {
        scenes.erase(std::remove_if(scenes.begin(), scenes.end(),
                                    [sc](const SceneWithPosition& swp) {
                                        return swp.scenePointer == sc;
                                    }), scenes.end());
    }

    void pre_query() override {
        state_query.clear();
        for(const SceneWithPosition& swp : scenes){
            append_to_state_query(StateQuery{swp.dag_name + ".x",
                                             swp.dag_name + ".y",
                                             swp.dag_name + ".w",
                                             swp.dag_name + ".h", });
            for(string s : swp.scenePointer->state_query){
                state_query.insert(s);
            }
        }
    }

    void draw() override{
        render_composite();
    }

private:
    vector<SceneWithPosition> scenes;
};
