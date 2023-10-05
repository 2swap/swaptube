#pragma once

#include "scene.cpp"
#include "Connect4/c4.h"
#include "calculator.cpp"
#include <unordered_map>

class VariableScene : public Scene {
public:
    VariableScene(const int width, const int height) : Scene(width, height) {}
    VariableScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {}

    void set_subscene(Scene* _subscene){
        subscene = _subscene;
    }

    void insert_variable(string variable, string equation) {
        variables[variable] = equation;
    }

    void print_variables() const {
        std::cout << "Variables:" << std::endl;
        std::cout << "-----------------------" << std::endl;
        for (const auto& variable : variables) {
            std::cout << variable.first << " : " << variable.second << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
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

    Pixels* query(bool& done_scene) override {
        subscene->update_variables(evaluate_all(variables, time));
        done_scene = time++>=scene_duration_frames;
        bool b;
        return subscene->query(b);
    }

private:
    Scene* subscene;
    unordered_map<string, string> variables;
};