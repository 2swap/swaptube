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
    vector<Pixels> convolutions;
    vector<pair<int, int>> coords;
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
        cout << blurbs[i]["latex"] << " <- CONVOLVING -> " << blurbs[i+1]["latex"] << endl;
        intersections.push_back(find_intersections(equations[i], equations[i+1]));
        int max_x = 0, max_y = 0;
        convolutions.push_back(convolve_map(equations[i], equations[i+1], max_x, max_y));
        x += max_x;
        y += max_y;
        coords.push_back(make_pair(x, y));
    }
}

void LatexScene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    p.copy(equations[which], coords[which].first, coords[which].second, 1, 1);
}

void LatexScene::render_transition(Pixels& p, int which, double weight) {
    p.fill(BLACK);

    if(which == equations.size()-1) {
        p.copy(equations[which], coords[which].first, coords[which].second, 1, cube(1-weight));
        return;
    }

    for (int i = 0; i < intersections[which].size(); i++) {
        const StepResult& step = intersections[which][i];
        int x = round(lerp(coords[which].first , coords[which+1].first-step.max_x, smoother2(weight)));
        int y = round(lerp(coords[which].second, coords[which+1].second-step.max_y, smoother2(weight)));

        // Render the intersection at the interpolated position
        p.copy(step.induced1, x-step.current_p1.w, y-step.current_p1.h, 1, transparency_profile(1-weight));
        p.copy(step.induced2, x-step.current_p2.w, y-step.current_p2.h, 1, transparency_profile(weight));
    }

    int num_intersections = intersections[which].size();
    p.copy(intersections[which][num_intersections-1].current_p1, coords[which].first, coords[which].second, 1, transparency_profile(1-weight));
    p.copy(intersections[which][num_intersections-1].current_p2, coords[which+1].first, coords[which+1].second, 1, transparency_profile(weight));
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