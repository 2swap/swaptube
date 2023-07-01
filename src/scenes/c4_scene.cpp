#pragma once

#include "scene.h"
#include "sequential_scene.cpp"
#include "Connect4/c4.h"
using json = nlohmann::json;

inline int C4_RED           = 0xffcc6677;
inline int C4_YELLOW        = 0xffddcc77;
inline int C4_EMPTY         = 0xff222222;

class C4Scene : public SequentialScene {
public:
    C4Scene(const json& config, const json& contents, MovieWriter* writer);
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new C4Scene(config, scene, writer);
    }

private:
    int time_in_this_block = 0;
    vector<Board> boards;
    vector<string> names;
};

void draw_c4_disk(Pixels& p, int stonex, int stoney, int col_id, bool highlight, char annotation, double t){
    int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};
    int col = cols[col_id];

    double stonewidth = p.w/16.;
    col = colorlerp(col, BLACK, .4);
    int darkcol = colorlerp(col, BLACK, .4);
    double px = (stonex-WIDTH/2.+.5)*stonewidth+p.w/2;
    double py = (-stoney+HEIGHT/2.-.5)*stonewidth+p.h/2;
    if(col_id != 0){
        double ringsize = 1;
        if(highlight){
            double blink = bound(0, t+2*(t-static_cast<double>(stonex)/WIDTH), 2);
            ringsize = 1-.3*(.5*(-cos(blink*3.14159)+1));
        }
        p.fill_ellipse(px, py, ceil(stonewidth*(.4*ringsize+.07)), ceil(stonewidth*(.4*ringsize+.07)), col);
        p.fill_ellipse(px, py, ceil(stonewidth*(.4*ringsize    )), ceil(stonewidth*(.4*ringsize    )), darkcol);
    }
    else{
        darkcol = BLACK;
    }

    switch (annotation) {
        case '+':
            p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw two rectangles to form a plus sign
            p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, col);
            break;
        case '-':
            p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw a rectangle to form a minus sign
            break;
        case '|':
            p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, col);  // Draw a rectangle to form a vertical bar
            break;
        case '=':
            p.fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw two rectangles to form an equal sign
            p.fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, col);
            break;
        case 'r':
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_RED);
            break;
        case 'R':
            p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_RED);
            break;
        case 'y':
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_YELLOW);
            break;
        case 'Y':
            p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_YELLOW);
            break;
        case 'b':
            px -= stonewidth/8;
            py -= stonewidth/12;
            p.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
            px += stonewidth/4;
            py += stonewidth/6;
            p.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
            break;
        case 'B':
            px -= stonewidth/8;
            py -= stonewidth/12;
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
            px += stonewidth/4;
            py += stonewidth/6;
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
            break;
        case ':':
            p.fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, col);
            p.fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, col);
            break;
        case '0':
            p.fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, col);
            p.fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, col);
            break;
        case '.':
            p.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, col);
            p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, darkcol);
            break;
        default:
            if(col_id == 0)
                p.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, col);
            break;
    }
}

void render_c4_board(Pixels& p, Board b, double t){
    // background
    p.fill(BLACK);
    for(int stonex = 0; stonex < WIDTH; stonex++)
        for(int stoney = 0; stoney < HEIGHT; stoney++)
            draw_c4_disk(p, stonex, stoney, b.grid[stoney][stonex], b.highlight[stoney][stonex], b.get_annotation(stonex, stoney), t);
}

C4Scene::C4Scene(const json& config, const json& contents, MovieWriter* writer) : SequentialScene(config, contents, writer) {
    vector<json> sequence_json = contents["sequence"];
    for (int i = 1; i < sequence_json.size()-1; i+=2) {
        cout << "constructing board " << i << endl;
        json board = sequence_json[i];
        vector<string> annotations = board["annotations"];

        // Concatenate annotations into a single string
        string concatenatedAnnotations;
        for (const auto& annotation : annotations) {
            concatenatedAnnotations += annotation;
        }

        boards.push_back(Board(board["representation"], concatenatedAnnotations));
        names.push_back(board["name"]);
    }
    frontload_audio(contents, writer);
}

void C4Scene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    Board b = boards[which];
    render_c4_board(pix, b, static_cast<double>(time_in_this_block)/framerate);
    Pixels board_title_pix = eqn_to_pix(latex_text(names[which]), pix.w / 640 + 1);
    pix.copy(board_title_pix, (pix.w - board_title_pix.w)/2, pix.h-pix.w/12, 1);
    time_in_this_block++;
}

void C4Scene::render_transition(Pixels& p, int which, double weight) {
    time_in_this_block = 0;
    p.fill(BLACK);

    if(which == boards.size()-1) {
        Pixels x = p;
        render_non_transition(x, which);
        pix.copy(x, 0, 0, weight);
        return;
    }
    if(which == -1) {
        Pixels x = p;
        render_non_transition(x, which+1);
        pix.copy(x, 0, 0, 1-weight);
        return;
    }

    Board transition = c4lerp(boards[which], (which == boards.size() - 1) ? Board("") : boards[which+1], weight);
    render_c4_board(pix, transition, 0);
    
    string curr_title = "\\text{" + names[which] + "}";
    string next_title = (which == contents["boards"].size() - 1)?"\\text{}":"\\text{" + names[which+1] + "}";

    Pixels curr_title_pix = eqn_to_pix(curr_title, pix.w / 640 + 1);
    Pixels next_title_pix = eqn_to_pix(next_title, pix.w / 640 + 1);
    
    if(next_title == curr_title)
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1);
    else{
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1-weight);
        pix.copy(next_title_pix, (pix.w - next_title_pix.w)/2, pix.h-pix.w/12, weight);
    }
}
