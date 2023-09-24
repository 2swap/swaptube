#pragma once

#include "scene.cpp"
#include "Connect4/c4.h"
#include "calculator.cpp"
#include <unordered_map>

class VariableScene : public Scene {
public:
    Scene* createScene(const int width, const int height) override {
        return new VariableScene(width, height);
    }

    VariableScene(const int width, const int height) : Scene(width, height) {}

    ~VariableScene() {
        delete subscene;
    }

    void set_subscene(Scene* _subscene){
        subscene = _subscene;
    }
    
    unordered_map<string, double> evaluate_all(const unordered_map<string, string>& variables, double time) const {
        unordered_map<string, double> evaluatedVariables;
        for (const auto& variable : variables) {
            const string& variableName = variable.first;
            const string& expression = variable.second;

            // Evaluate the expression based on the provided time
            double evaluatedValue = calculator(expression, time);
            evaluatedVariables[variableName] = evaluatedValue;
        }
        return evaluatedVariables;
    }

    const Pixels& query(bool& done_scene) override {
        subscene->update_variables(evaluate_all(variables, time));
        Pixels p = subscene->query(done_scene);
        pix.copy(p, 0, 0, 1);
        time++;
        return pix;
    }

private:
    Scene* subscene;
    unordered_map<string, string> variables;
};