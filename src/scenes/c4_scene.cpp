#pragma once

#include "Connect4/c4.h"
using json = nlohmann::json;

class C4Scene : public Scene {
public:
    C4Scene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);

private:
    vector<Board> boards;
};

C4Scene::C4Scene(const json& config, const json& contents) : Scene(config, contents) {
    vector<json> boards_json = contents["boards"];
    for (int i = 0; i < boards_json.size(); i++) {
        json board = boards_json[i];
        boards.push_back(Board(board["representation"], board["annotations"]));
    }
}

void C4Scene::render_non_transition(Pixels& p, int which) {
    p.fill(0);
    Board b = boards[which];
    pix.render_c4_board(b);
    Pixels board_title_pix = eqn_to_pix(latex_text(contents["boards"][which]["name"].get<string>()), 5);
    pix.copy(board_title_pix, (pix.w - board_title_pix.w)/2, pix.h-200, 1, 1);
}

void C4Scene::render_transition(Pixels& p, int which, double weight) {
    p.fill(0);

    Board transition = c4lerp(boards[which], (which == boards.size() - 1) ? Board("") : boards[which+1], weight);
    pix.render_c4_board(transition);
    
    string curr_title = "\\text{" + contents["boards"][which]["name"].get<string>() + "}";;
    string next_title = (which == contents["boards"].size() - 1)?"\\text{}":"\\text{" + contents["boards"][which+1]["name"].get<string>() + "}";

    Pixels curr_title_pix = eqn_to_pix(curr_title, 5);
    Pixels next_title_pix = eqn_to_pix(next_title, 5);
    
    if(next_title == curr_title)
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-200, 1, 1);
    else{
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-200, 1, weight);
        pix.copy(next_title_pix, (pix.w - next_title_pix.w)/2, pix.h-200, 1, 1-weight);
    }
}

Pixels C4Scene::query(int& frames_left) {
    vector<json> boards_json = contents["boards"];
    int time_left = 0;
    int time_spent = time;
    bool rendered = false;
    for (int i = 0; i < boards_json.size(); i++) {
        json board_json = boards_json[i];

        double duration_frames = board_json["duration_seconds"].get<int>() * framerate;
        if (rendered) time_left += duration_frames;
        else if (time_spent < duration_frames) {
            render_non_transition(pix, i);
            rendered = true;
            time_left += duration_frames - time_spent;
        } else {
            time_spent -= duration_frames;
        }

        double transition_frames = board_json["transition_seconds"].get<int>() * framerate;
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