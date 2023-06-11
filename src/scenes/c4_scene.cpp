#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

inline int C4_RED         = 0xff880000;
inline int C4_YELLOW      = 0xff888800;
inline int C4_EMPTY       = 0xff003388;

class C4Scene : public Scene {
public:
    C4Scene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene) override {
        return new C4Scene(config, scene);
    }

private:
    vector<Board> boards;
};

void draw_c4_disk(Pixels& p, int stonex, int stoney, int col, bool highlight, char annotation){
    double stonewidth = p.w/16.;
    int highlightcol = colorlerp(col, 0, .4);
    int textcol = colorlerp(col, 0, .7);
    double px = (stonex-WIDTH/2.+.5)*stonewidth+p.w/2;
    double py = (-stoney+HEIGHT/2.-.5)*stonewidth+p.h/2;
    if(highlight) p.fill_ellipse(px, py, stonewidth*.48, stonewidth*.48, highlightcol);
    p.fill_ellipse(px, py, stonewidth*.4, stonewidth*.4, col);

    switch (annotation) {
        case '+':
            p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw two rectangles to form a plus sign
            p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, textcol);
            break;
        case '-':
            p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw a rectangle to form a minus sign
            break;
        case '|':
            p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, textcol);  // Draw a rectangle to form a vertical bar
            break;
        case '=':
            p.fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, textcol);  // Draw two rectangles to form an equal sign
            p.fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, textcol);
            break;
        case 't':
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, textcol);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, textcol);
            break;
        case 'T':
            p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, textcol);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, textcol);
            break;
        case ':':
            p.fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, textcol);  // Draw an ellipse to form a ':'
            p.fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, textcol);
            break;
        case '0':
            p.fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, textcol);  // Draw a circle to form a '0'
            p.fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, col);
            break;
        case '.':
            p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, textcol);  // Draw a circle to form a '0'
            break;
        default:
            break;
    }
}

void render_c4_board(Pixels& p, Board b){
    int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};

    // background
    p.fill(BLACK);
    for(int stonex = 0; stonex < WIDTH; stonex++)
        for(int stoney = 0; stoney < HEIGHT; stoney++)
            draw_c4_disk(p, stonex, stoney, cols[b.grid[stoney][stonex]], b.highlight[stoney][stonex], b.get_annotation(stonex, stoney));

}

C4Scene::C4Scene(const json& config, const json& contents) : Scene(config, contents) {
    vector<json> boards_json = contents["boards"];
    for (int i = 0; i < boards_json.size(); i++) {
        json board = boards_json[i];
        boards.push_back(Board(board["representation"], board["annotations"]));
    }
}

void C4Scene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    Board b = boards[which];
    render_c4_board(pix, b);
    Pixels board_title_pix = eqn_to_pix(latex_text(contents["boards"][which]["name"].get<string>()), pix.w / 320);
    pix.copy(board_title_pix, (pix.w - board_title_pix.w)/2, pix.h-pix.w/12, 1, 1);
}

void C4Scene::render_transition(Pixels& p, int which, double weight) {
    p.fill(BLACK);

    Board transition = c4lerp(boards[which], (which == boards.size() - 1) ? Board("") : boards[which+1], weight);
    render_c4_board(pix, transition);
    
    string curr_title = "\\text{" + contents["boards"][which]["name"].get<string>() + "}";
    string next_title = (which == contents["boards"].size() - 1)?"\\text{}":"\\text{" + contents["boards"][which+1]["name"].get<string>() + "}";

    Pixels curr_title_pix = eqn_to_pix(curr_title, pix.w / 320);
    Pixels next_title_pix = eqn_to_pix(next_title, pix.w / 320);
    
    if(next_title == curr_title)
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1, 1);
    else{
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1, weight);
        pix.copy(next_title_pix, (pix.w - next_title_pix.w)/2, pix.h-pix.w/12, 1, 1-weight);
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