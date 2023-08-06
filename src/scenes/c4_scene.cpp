#pragma once

#include "scene.h"
#include "sequential_scene.cpp"
#include "Connect4/c4.h"
#include "subscenes/C4Subscene.cpp"
using json = nlohmann::json;

class C4Scene : public SequentialScene {
public:
    C4Scene(const json& config, const json& contents, MovieWriter* writer);
    void render_non_transition(Pixels& p, int content_index);
    void render_transition(Pixels& p, int transition_index, double weight);
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new C4Scene(config, scene, writer);
    }
    Subscene* interpolate(Subscene* s1, Subscene* s2, double weight) override{
        C4Subscene* c4s1 = dynamic_cast<C4Subscene*>(s1);
        C4Subscene* c4s2 = dynamic_cast<C4Subscene*>(s2);
        return new C4Subscene(*c4s1, *c4s2, weight);
    }
};

C4Scene::C4Scene(const json& config, const json& contents, MovieWriter* writer) : SequentialScene(config, contents, writer) {
    int width = config["width"];
    int height = config["height"];
    vector<json> sequence_json = contents["sequence"];
    for (int i = 1; i < sequence_json.size()-1; i+=2) {
        cout << "constructing board " << i << endl;
        json curr = sequence_json[i];

        double threat_diagram = curr.contains("reduction") ? 1 : 0;
        double spread = curr.value("spread", false) ? 1 : 0;
        vector<string> reduction;
        vector<string> reduction_colors;
        if(threat_diagram == 1){
            reduction = curr["reduction"];
            reduction_colors = curr["reduction_colors"];
        }

        // Concatenate annotations into a single string
        string concatenatedAnnotations = "";
        if (curr.contains("annotations")) {
            vector<string> annotations = curr["annotations"];
            for (const auto& annotation : annotations) {
                concatenatedAnnotations += annotation;
            }
        }
        else{
            concatenatedAnnotations = "                                          ";
        }

        // Concatenate highlights into a single string
        string concatenatedHighlights = "";
        if (curr.contains("highlight")) {
            vector<string> highlights = curr["highlight"];
            for (const auto& highlight : highlights) {
                concatenatedHighlights += highlight;
            }
        }
        else{
            concatenatedHighlights = "                                          ";
        }

        Board b = Board(curr["representation"]);
        subscenes.push_back(new C4Subscene(width, height, threat_diagram, spread, reduction, reduction_colors, b, concatenatedAnnotations, concatenatedHighlights));
    }
}
