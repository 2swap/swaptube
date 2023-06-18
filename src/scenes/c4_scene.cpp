#pragma once

#include "scene.h"
#include "Connect4/c4.h"
using json = nlohmann::json;

inline int C4_RED         = 0xff880000;
inline int C4_YELLOW      = 0xff888800;
inline int C4_EMPTY       = 0xff115599;

class C4Scene : public Scene {
public:
    C4Scene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    void render_non_transition(Pixels& p, int which);
    void render_transition(Pixels& p, int which, double weight);
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new C4Scene(config, scene, writer);
    }

private:
    vector<Board> boards;
    vector<double> durations;
    vector<string> names;
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
            p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 6, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 24, stonewidth / 2, C4_RED);
            p.fill_rect(px - stonewidth * 0, py - stonewidth / 8, stonewidth / 6, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 't'
            p.fill_rect(px - stonewidth * 0, py - stonewidth / 4, stonewidth / 24, stonewidth / 2, C4_YELLOW);
            break;
        case 'B':
            p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 4, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 24, stonewidth / 2, C4_RED);
            p.fill_rect(px - stonewidth * 0, py - stonewidth / 4, stonewidth / 4, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 'T'
            p.fill_rect(px - stonewidth * 0, py - stonewidth / 4, stonewidth / 24, stonewidth / 2, C4_YELLOW);
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

C4Scene::C4Scene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents) {
    vector<json> boards_json = contents["boards"];
    for (int i = 1; i < boards_json.size()-1; i+=2) {
        json board = boards_json[i];
        boards.push_back(Board(board["representation"], board["annotations"]));
        names.push_back(board["name"]);
    }

    // Frontload audio
    if(contents.find("audio") != contents.end()){
        cout << "This scene has a single audio for all of its subscenes." << endl;
        double duration = writer.add_audio_get_length(contents["audio"].get<string>());
        for (int i = 0; i < boards_json.size(); i++){
            durations.push_back(duration/boards_json.size());
        }
    }
    else{
        cout << "This scene has a unique audio for each of its subscenes." << endl;
        for (int blurb_index = 0; blurb_index < boards_json.size(); blurb_index++) {
            json& board_json = boards_json[blurb_index];
            if (board_json.find("audio") != board_json.end()) {
                string audio_path = board_json["audio"].get<string>();
                durations.push_back(writer.add_audio_get_length(audio_path));
            } else {
                double duration = board_json["duration_seconds"].get<int>();
                writer.add_silence(duration);
                durations.push_back(duration);
            }
        }
    }
}

void C4Scene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    Board b = boards[which];
    render_c4_board(pix, b);
    Pixels board_title_pix = eqn_to_pix(latex_text(names[which]), pix.w / 320);
    pix.copy(board_title_pix, (pix.w - board_title_pix.w)/2, pix.h-pix.w/12, 1);
}

void C4Scene::render_transition(Pixels& p, int which, double weight) {
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
    render_c4_board(pix, transition);
    
    string curr_title = "\\text{" + names[which] + "}";
    string next_title = (which == contents["boards"].size() - 1)?"\\text{}":"\\text{" + names[which+1] + "}";

    Pixels curr_title_pix = eqn_to_pix(curr_title, pix.w / 320);
    Pixels next_title_pix = eqn_to_pix(next_title, pix.w / 320);
    
    if(next_title == curr_title)
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1);
    else{
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1-weight);
        pix.copy(next_title_pix, (pix.w - next_title_pix.w)/2, pix.h-pix.w/12, weight);
    }
}

Pixels C4Scene::query(int& frames_left) {
    vector<json> boards_json = contents["boards"];
    int time_left = 0;
    int time_spent = time;
    bool rendered = false;
    int content_index = -1;
    for (int i = 0; i < boards_json.size(); i++) {
        json board_json = boards_json[i];
        bool is_transition = board_json.find("transition") != board_json.end();

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