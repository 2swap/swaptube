#pragma once

#include "scene.h"
#include "Connect4/c4.h"
#include "calculator.cpp"
using json = nlohmann::json;

class VariableScene : public Scene {
public:
    VariableScene(const json& config, const json& contents, MovieWriter* writer);
    ~VariableScene();
    Pixels query(bool& done_scene) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new VariableScene(config, scene, writer);
    }

private:
    Scene* subscene;
};

VariableScene::VariableScene(const json& config, const json& contents, MovieWriter* writer) : Scene(config, contents, writer) {
    json contents_with_duration = contents["subscene"];
    contents_with_duration["duration_seconds"] = contents["duration_seconds"];

    subscene = create_scene_determine_type(config, contents_with_duration, nullptr);
    add_audio(contents, writer);
}

VariableScene::~VariableScene() {
    delete subscene;
}

map<string, double> evaluate_all(const json& variables, double time) {
    map<string, double> evaluatedVariables;
    for (auto it = variables.begin(); it != variables.end(); ++it) {
        const string& variableName = it.key();
        const string& expression = it.value();

        // Evaluate the expression based on the provided time
        double evaluatedValue = calculator(expression, time);
        evaluatedVariables[variableName] = evaluatedValue;
    }
    return evaluatedVariables;
}

Pixels VariableScene::query(bool& done_scene) {
    subscene->update_variables(evaluate_all(contents["variables"], time));
    Pixels p = subscene->query(done_scene);
    pix.copy(p, 0, 0, 1);
    time++;
    return pix;
}
