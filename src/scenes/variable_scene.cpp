#pragma once

#include "scene.cpp"
#include "Connect4/c4.h"
#include "calculator.cpp"
#include <unordered_map>

class VariableScene : public Scene {
public:
    VariableScene(const int width, const int height, Scene* _subscene) : Scene(width, height), subscene(_subscene) {}
    VariableScene(Scene* _subscene) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), subscene(_subscene) {}

    void stage_transition(unordered_map<string, string> v){
        upcoming_variables = v;
        is_transition = true;
        rendered = false;
        prior_time = time;
        assert(v.size() == variables.size());
    }

    void set_variables(unordered_map<string, string> v){
        variables = v;
        rendered = false;
    }

    void post_transition(){
        variables = upcoming_variables;
        is_transition = false;
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
    
    unordered_map<string, double> evaluate_all(const unordered_map<string, string>& variables) const {
        unordered_map<string, double> evaluatedVariables;
        for (const auto& variable : variables) {
            const string& variableName = variable.first;
            const string& expression = variable.second;

            // Evaluate the expression based on the provided time
            double evaluatedValue = calculator(expression, time + prior_time);
            evaluatedVariables[variableName] = evaluatedValue;
        }
        return evaluatedVariables;
    }
    
    unordered_map<string, double> interpolate_variables(double time) const {
        unordered_map<string, double> pre = evaluate_all(variables);
        unordered_map<string, double> post = evaluate_all(upcoming_variables);
        assert(pre.size() == post.size());
        unordered_map<string, double> interpolated;

        double smooth = smoother2(time/static_cast<double>(scene_duration_frames));

        // Interpolating each value from 'pre' and 'post'
        for (const auto &pair : pre) {
            const string &key = pair.first;
            double preValue = pair.second;
            double postValue = post[key];
            
            interpolated[key] = lerp(preValue, postValue, smooth);
        }
        return interpolated;
    }

    Pixels* query(bool& done_scene) override {
        if(is_transition) {subscene->update_variables(interpolate_variables(time));}
        else {subscene->update_variables(evaluate_all(variables));}
        done_scene = time++>=scene_duration_frames;
        if(done_scene && is_transition) post_transition();
        bool b;
        return subscene->query(b);
    }

private:
    int prior_time = 0;
    Scene* subscene;
    unordered_map<string, string> variables;
    unordered_map<string, string> upcoming_variables;
};