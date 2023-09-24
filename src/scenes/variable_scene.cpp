#pragma once

#include "scene.h"
#include "Connect4/c4.h"
#include "calculator.cpp"
using json = nlohmann::json;

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

class VariableScene : public Scene {
public:
    Scene* createScene(const int width, const int height, const json& scene) override {
        return new VariableScene(width, height, scene);
    }

    VariableScene(const int width, const int height, const json& contents) : Scene(width, height, contents) {
        json contents_with_duration = contents["subscene"];
        contents_with_duration["duration_seconds"] = contents["duration_seconds"];

        subscene = create_scene_determine_type(width, height, contents_with_duration);
        // DON'T add_audio(contents); -> The subscene will write audio! I don't need to!
    }

    ~VariableScene() {
        delete subscene;
    }

    const Pixels& query(bool& done_scene) override {
        subscene->update_variables(evaluate_all(contents["variables"], time));
        Pixels p = subscene->query(done_scene);
        pix.copy(p, 0, 0, 1);
        time++;
        return pix;
    }

private:
    Scene* subscene;
};