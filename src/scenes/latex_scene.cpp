#pragma once

#include "scene.h"
using json = nlohmann::json;

class LatexScene : public Scene {
public:
    LatexScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
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
    vector<double> durations;
};

LatexScene::LatexScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents) {
    vector<json> blurbs_json = contents["blurbs"];

    // Frontload latex rendering
    for (int blurb_index = 0; blurb_index < blurbs_json.size(); blurb_index++) {
        json blurb = blurbs_json[blurb_index];
        if(blurb.find("transition") != blurb.end()) continue;
        string eqn = blurb["latex"].get<string>();
        cout << "rendering latex: " << eqn << endl;
        Pixels p = eqn_to_pix(eqn, 2);
        equations.push_back(p);
        coords.push_back(make_pair((pix.w-p.w)/2, (pix.h-p.h)/2));
    }

    // Frontload convolution
    int equation_index = 0;
    for (int blurb_index = 2; blurb_index < blurbs_json.size()-2; blurb_index+=2) {
        json blurb_json = blurbs_json[blurb_index];
        bool is_transition = blurb_json.find("transition") != blurb_json.end();
        if(!is_transition) continue;
        cout << blurbs_json[blurb_index-1]["latex"] << " <- Finding Intersections -> " << blurbs_json[blurb_index+1]["latex"] << endl;
        intersections.push_back(find_intersections(equations[equation_index], equations[equation_index+1]));
        equation_index += 1;
    }

    // Frontload audio
    if(contents.find("audio") != contents.end()){
        cout << "This scene has a single audio for all of its subscenes." << endl;
        double duration = writer.add_audio_get_length(contents["audio"].get<string>());
        for (int i = 0; i < blurbs_json.size(); i++){
            durations.push_back(duration/blurbs_json.size());
        }
    }
    else{
        cout << "This scene has a unique audio for each of its subscenes." << endl;
        for (int blurb_index = 0; blurb_index < blurbs_json.size(); blurb_index++) {
            json& blurb_json = blurbs_json[blurb_index];
            if (blurb_json.find("audio") != blurb_json.end()) {
                string audio_path = blurb_json["audio"].get<string>();
                durations.push_back(writer.add_audio_get_length(audio_path));
            } else {
                double duration = blurb_json["duration_seconds"].get<int>();
                writer.add_silence(duration);
                durations.push_back(duration);
            }
        }
    }
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
    }

    int num_intersections = intersections[which].size();
    p.copy(intersections[which][num_intersections-1].current_p1, coords[which].first, coords[which].second, tp1);
    p.copy(intersections[which][num_intersections-1].current_p2, coords[which+1].first, coords[which+1].second, tp);
}

Pixels LatexScene::query(int& frames_left) {
    vector<json> blurbs_json = contents["blurbs"];
    int time_left = 0;
    int time_spent = time;
    bool rendered = false;
    int content_index = -1;
    for (int i = 0; i < blurbs_json.size(); i++) {
        json blurb_json = blurbs_json[i];
        bool is_transition = blurb_json.find("transition") != blurb_json.end();

        double frames = durations[i] * framerate;
        if (rendered) time_left += frames;
        else if (time_spent < frames) {
            if(is_transition)render_transition(pix, content_index, time_spent/frames);
            else render_non_transition(pix, content_index);
            rendered = true;
            time_left += frames - time_spent;
        } else {
            time_spent -= frames;
        }
        if(is_transition) content_index += 1;
    }
    frames_left = time_left;
    time++;
    return pix;
}