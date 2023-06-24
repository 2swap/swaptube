#pragma once

#include "scene.h"
#include "sequential_scene.cpp"
using json = nlohmann::json;

class LatexScene : public SequentialScene {
public:
    LatexScene(const json& config, const json& contents, MovieWriter& writer);
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new LatexScene(config, scene, writer);
    }
    bool show_cs;

private:
    vector<Pixels> equations;
    vector<vector<StepResult>> intersections;
    vector<Pixels> convolutions;
    vector<pair<int, int>> coords;
};

LatexScene::LatexScene(const json& config, const json& contents, MovieWriter& writer) : SequentialScene(config, contents, writer) {
    vector<json> sequence_json = contents["sequence"];

    // Frontload latex rendering
    for (int i = 0; i < sequence_json.size(); i++) {
        json blurb = sequence_json[i];
        if(blurb.find("transition") != blurb.end()) continue;
        string eqn = blurb["latex"].get<string>();
        cout << "rendering latex: " << eqn << endl;
        Pixels p = eqn_to_pix(eqn, pix.w / 320);
        equations.push_back(p);
        coords.push_back(make_pair((pix.w-p.w)/2, (pix.h-p.h)/2));
    }

    // Frontload convolution
    int equation_index = 0;
    for (int i = 2; i < sequence_json.size()-2; i+=2) {
        json blurb_json = sequence_json[i];
        bool is_transition = blurb_json.find("transition") != blurb_json.end();
        if(!is_transition) continue;
        cout << sequence_json[i-1]["latex"] << " <- Finding Intersections -> " << sequence_json[i+1]["latex"] << endl;
        intersections.push_back(find_intersections(equations[equation_index], equations[equation_index+1]));
        equation_index += 1;
    }

    frontload_audio(contents, writer);
}

void LatexScene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    p.copy(equations[which], coords[which].first, coords[which].second, 1);
}

void LatexScene::render_transition(Pixels& p, int which, double weight) {
    p.fill(BLACK);

    double tp = transparency_profile(weight);
    double tp1 = transparency_profile(1-weight);
    double smooth = smoother2(weight);

    if(which == equations.size()-1) {
        p.copy(equations[which], coords[which].first, coords[which].second, tp1);
        return;
    }
    if(which == -1) {
        p.copy(equations[which+1], coords[which+1].first, coords[which+1].second, tp);
        return;
    }

    for (int i = 0; i < intersections[which].size(); i++) {
        const StepResult& step = intersections[which][i];
        int x = round(lerp(coords[which].first , coords[which+1].first-step.max_x, smooth));
        int y = round(lerp(coords[which].second, coords[which+1].second-step.max_y, smooth));

        // Render the intersection at the interpolated position
        p.copy(step.induced1, x-step.current_p1.w, y-step.current_p1.h, tp1);
        p.copy(step.induced2, x-step.current_p2.w, y-step.current_p2.h, tp);

        //p.copy(step.intersection, -500, i*191-50, 1);
        //p.copy(step.map, 0, i*191-50, 1);
    }

    int num_intersections = intersections[which].size();
    p.copy(intersections[which][num_intersections-1].current_p1, coords[which].first, coords[which].second, tp1);
    p.copy(intersections[which][num_intersections-1].current_p2, coords[which+1].first, coords[which+1].second, tp);
}
