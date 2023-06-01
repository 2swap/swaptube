#pragma once

using json = nlohmann::json;

class LatexScene : public Scene {
public:
    LatexScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);

private:
    vector<Pixels> equations;
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
        int max_x = 0, max_y = 0;
        cout << blurbs[i]["latex"] << " <- CONVOLVING -> " << blurbs[i+1]["latex"] << endl;
        convolutions.push_back(convolve_map(equations[i], equations[i+1], max_x, max_y, false));
        x -= max_x;
        y -= max_y;
        coords.push_back(make_pair(x, y));
    }
}

void LatexScene::render_non_transition(Pixels& p, int which) {
    int x = coords[which].first;
    int y = coords[which].second;
    p.fill(0);
    p.copy(equations[which], x, y, 1, 1);
}

void LatexScene::render_transition(Pixels& p, int which, double weight) {
    p.fill(0);
    int x1 = coords[which].first;
    int y1 = coords[which].second;
    p.copy(equations[which], x1, y1, 1, 1-weight);

    if(which != coords.size()-1){
        int x2 = coords[which+1].first;
        int y2 = coords[which+1].second;
        p.copy(equations[which+1], x2, y2, 1, weight);
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