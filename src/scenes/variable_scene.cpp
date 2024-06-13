#pragma once

#include "scene.cpp"
#include "calculator.cpp"
#include <unordered_map>

class VariableScene : public Scene {
public:
    VariableScene(const int width, const int height, Scene* _subscene) : Scene(width, height), subscene(_subscene) {set_variables_to_scene_default();}
    VariableScene(Scene* _subscene) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), subscene(_subscene) {set_variables_to_scene_default();}

    void set_variables_to_scene_default(){
        variables = subscene->get_default_variables();
    }

    void stage_transition(unordered_map<string, string> v){
        upcoming_variables = variables;
        update_map(upcoming_variables, v);
        is_transition = true;
        rendered = false;
        assert(upcoming_variables.size() == variables.size());
    }

    void update_map(std::unordered_map<std::string, string>& map1, const std::unordered_map<std::string, string>& map2) {
        for (const auto& pair : map2) {
            if(map1.find(pair.first) == map1.end()){
                cout << "Invalid variable key!" << endl;
            }
            map1[pair.first] = pair.second;
        }
    }

    void set_variables(unordered_map<string, string> v){
        update_map(variables, v);
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
            cout << evaluatedValue << endl;
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

    void query(bool& done_scene, Pixels*& p) override {
        if(is_transition) {subscene->update_variables(interpolate_variables(time));}
        else {subscene->update_variables(evaluate_all(variables));}
        done_scene = time++>=scene_duration_frames;
        if(done_scene && is_transition) post_transition();
        if(done_scene) prior_time += time;
        cout << "ccc" << endl;
        subscene->query(p);
    }

private:
    int prior_time = 0;
    Scene* subscene;
    unordered_map<string, string> variables;
    unordered_map<string, string> upcoming_variables;
};
