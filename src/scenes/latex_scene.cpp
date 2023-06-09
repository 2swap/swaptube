#pragma once

#include "scene.h"
using json = nlohmann::json;

class LatexScene : public Scene {
public:
    LatexScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition_fade(Pixels& p, int which, double weight);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene) override {
        return new LatexScene(config, scene);
    }
    bool show_cs;

private:
    vector<Pixels> equations;
    vector<Pixels> convolutions;
    vector<pair<int, int>> coords;
    vector<vector<StepResult>> intersections;
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

    int x = (pix.w-equations[0].w)/2;
    int y = (pix.h-equations[0].h)/2;
    coords.push_back(make_pair(x, y));

    // Frontload convolution
    for (int i = 0; i < equations.size()-1; i++) {
        int max_x = 0, max_y = 0;
        cout << blurbs[i]["latex"] << " <- CONVOLVING -> " << blurbs[i+1]["latex"] << endl;
        intersections.push_back(find_intersections(equations[i], equations[i+1]));
        convolutions.push_back(convolve_map(equations[i], equations[i+1], max_x, max_y));
        x += max_x;
        y += max_y;
        coords.push_back(make_pair(x, y));
    }
}

void LatexScene::render_non_transition(Pixels& p, int which) {
    int x = coords[which].first;
    int y = coords[which].second;
    p.fill(0);
    p.copy(equations[which], x, y, 1, 1);
}

void LatexScene::render_transition_fade(Pixels& p, int which, double weight) {
    p.fill(0);
    int x1 = coords[which].first;
    int y1 = coords[which].second;
    p.copy(equations[which], x1, y1, 1, 1-weight);

    if(which == coords.size()-1) return;

    p.copy(convolutions[which], 0, 0, 1, 1);
    int x2 = coords[which+1].first;
    int y2 = coords[which+1].second;
    p.copy(equations[which+1], x2, y2, 1, weight);
}

void LatexScene::render_transition(Pixels& p, int which, double weight) {
    p.fill(0);

    if(which == coords.size()-1) return;
    
    int x2 = coords[which+1].first;
    int y2 = coords[which+1].second;

    int i = 0;

    for (const StepResult& step : intersections[which]) {
        int old_x = 0;
        int old_y = 0;
        int new_x = step.max_x;
        int new_y = step.max_y;

        int x = lerp(old_x, new_x, smoother2(1-weight));
        int y = lerp(old_y, new_y, smoother2(1-weight));

        // Render the intersection at the interpolated position
        p.copy(step.intersection, x+x2, y+y2, 1, 1);

        //p.copy(step.map, 0, i, 1, 1);
        //p.copy(step.intersection, 400, i, 1, 1);
        //p.copy(step.current_p1, 600, i, 1, 1);
        //p.copy(step.current_p2, 800, i, 1, 1);
        i+=100;
    }
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