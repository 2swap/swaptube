#pragma once

#include "scene.h"
using json = nlohmann::json;

class LatexScene : public Scene {
public:
    LatexScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene) override {
        return new LatexScene(config, scene);
    }
    bool show_cs;

private:
    vector<Pixels> equations;
    vector<vector<StepResult>> intersections;
    int x_orig = 0;
    int y_orig = 0;
};

LatexScene::LatexScene(const json& config, const json& contents) : Scene(config, contents) {
    vector<json> blurbs = contents["blurbs"];

    // Frontload latex rendering
    for (int blurb_index = 0; blurb_index < blurbs.size(); blurb_index++) {
        json blurb = blurbs[blurb_index];
        string eqn = blurb["latex"].get<string>();
        cout << "rendering latex: " << eqn << endl;
        equations.push_back(eqn_to_pix(eqn, 2));
    }

    x_orig = (pix.w-equations[0].w)/2;
    y_orig = (pix.h-equations[0].h)/2;

    // Frontload convolution
    for (int i = 0; i < equations.size()-1; i++) {
        cout << blurbs[i]["latex"] << " <- CONVOLVING -> " << blurbs[i+1]["latex"] << endl;
        intersections.push_back(find_intersections(equations[i], equations[i+1]));
    }
}

void LatexScene::render_non_transition(Pixels& p, int which) {
    p.fill(0);
    p.copy(equations[which], x_orig, y_orig, 1, 1);
}

void LatexScene::render_transition(Pixels& p, int which, double weight) {
    p.fill(0);

    if(which == equations.size()-1) return;

    for (const StepResult& step : intersections[which]) {
        int new_x = step.max_x;
        int new_y = step.max_y;

        int x = lerp(x_orig, x_orig-new_x, smoother2(weight));
        int y = lerp(y_orig, y_orig-new_y, smoother2(weight));

        // Render the intersection at the interpolated position
        p.copy(step.intersection, x, y, 1, 1);
    }

    int num_intersections = intersections[which].size();
    p.copy(intersections[which][num_intersections-1].current_p1, x_orig, y_orig, 1, 1-weight);
    p.copy(intersections[which][num_intersections-1].current_p2, x_orig, y_orig, 1, weight);
}

Pixels LatexScene::query(int& frames_left) {
    vector<json> blurbs = contents["blurbs"];
    int time_left = 0;
    int time_spent = time;
    bool rendered = false;
    for (int i = 0; i < equations.size(); i++) {
        json blurb_json = blurbs[i];

        double duration_frames = blurb_json["duration_seconds"].get<int>() * framerate;
        if (rendered) time_left += duration_frames;
        else if (time_spent < duration_frames) {
            render_non_transition(pix, i);
            rendered = true;
            time_left += duration_frames - time_spent;
        } else {
            time_spent -= duration_frames;
        }

        double transition_frames = blurb_json["transition_seconds"].get<int>() * framerate;
        if (rendered) time_left += transition_frames;
        else if (time_spent < transition_frames) {
            render_transition(pix, i, time_spent/(double)transition_frames);
            rendered = true;
            time_left += transition_frames - time_spent;
        } else {
            time_spent -= transition_frames;
        }
    }
    frames_left = time_left;
    time++;
    return pix;
}