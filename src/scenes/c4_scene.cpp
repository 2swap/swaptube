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

void draw_c4_disk(Pixels& p, int stonex, int stoney, int col_id, bool blink, char highlight, char annotation, double t, double spread, double threat_diagram){
    int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};
    int col = cols[col_id];

    double stonewidth = p.w/16.;
    col = colorlerp(col, BLACK, .4);
    double spreadx = lerp(1, 2.1, spread);
    double spready = lerp(0, -.75 + 1.5 * (stonex%2), spread*(1-threat_diagram));
    double px = round((stonex-WIDTH/2.+.5)*stonewidth*spreadx+p.w/2);
    double py = round((-stoney+spready  +HEIGHT/2.-.5)*stonewidth+p.h/2) - (threat_diagram*p.h/8);

    double ringsize = 1;

    if (highlight != ' '){
        int highlight_color = WHITE;
        if(highlight == 'd' || highlight == 'D') highlight_color = HASHMARKS ; // dead space
        if(highlight == 't' || highlight == 'T') highlight_color = DIAMONDS  ; // terminal threat
        if(highlight == 'z' || highlight == 'Z') highlight_color = HORIZONTAL; // zugzwang controlling threat
        if(highlight == 'n' || highlight == 'N') highlight_color = VERTICAL  ; // non-controlling threat
        p.rounded_rect(px - stonewidth * .4,
                       py - stonewidth * .14,
                       stonewidth * .8,
                       stonewidth * 1.8,
                       stonewidth * .4, highlight_color);
    }

    if(col_id != 0){
        if(blink || (col_id == 1 && (annotation == 'B' || annotation == 'R')) || (col_id == 2 && (annotation == 'B' || annotation == 'Y'))){
            double blink = bound(0, t, 2);
            ringsize = 1-.3*(.5*(-cos(blink*3.14159)+1));
        }
        int piece_fill_radius = ceil(stonewidth*(.4*ringsize));
        int piece_stroke_radius = ceil(stonewidth*(.4*ringsize+.07));
        p.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col    );
        p.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, BLACK, .4));
        return;
    }

    p.fill_ellipse(px, py, stonewidth*.3, stonewidth*.3, BLACK);

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
        case 'o':
        case 'O':
            p.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, col);
            p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
            break;
        case 'c': // c is an o but colorful
            p.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, 0xff0099cc);
            p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
            break;
        case 'x':
            {
                double rw = stonewidth*.3;
                double rh = stonewidth*.3;
                for(double dx = -rw+1; dx < rw; dx++)
                    for(double dy = -rh+1; dy < rh; dy++)
                        if(square(dx/rw)+square(dy/rh) < 1 && (abs(dx - dy) < stonewidth*.1 || abs(dx + dy) < stonewidth*.1))
                            p.set_pixel_with_transparency(px+dx, py+dy, col);
                break;
            }
        case '.':
            if(col_id == 0)
                p.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, col);
            break;
        default:
            break;
    }
}

void render_c4_board(Pixels& p, Board b, double t, double spread, double threat_diagram){
    // background
    p.fill(BLACK);
    for(int stonex = 0; stonex < WIDTH; stonex++)
        for(int stoney = 0; stoney < HEIGHT; stoney++){
            char this_highlight = b.get_highlight(stonex, stoney);
            char next_highlight = stoney == HEIGHT-1 ? ' ' : b.get_highlight(stonex, stoney+1);
            if(this_highlight != next_highlight) this_highlight = ' ';
            draw_c4_disk(p, stonex, stoney, b.grid[stoney][stonex], b.blink[stoney][stonex], this_highlight, b.get_annotation(stonex, stoney), t, spread, threat_diagram);
        }
    p.texture();
}

void render_threat_diagram(Pixels& p, const vector<string>& reduction, const vector<string>& reduction_colors, double threat_diagram, int diff_index, double weight, double spread){
    if(threat_diagram == 0) return;
    for(int x = 0; x < reduction.size(); x++){
        string column = reduction[x];
        for(int y = 0; y < column.size(); y++){
            char r = column.at(y);
            char rc = reduction_colors[x][y];
            int color = 0xff226688;
            if(rc == 'R' || rc == 'r') color = C4_RED;
            if(rc == 'Y' || rc == 'y') color = C4_YELLOW;
            string s = string(1,r);
            Pixels latex = eqn_to_pix("\\text{"+s+"}", p.w/640);
            latex.recolor(color);
            latex.mult_alpha(threat_diagram);
            double stonewidth = p.w/16.;
            double shifty = (x==diff_index && y != 0)?weight:0;
            double spreadx = lerp(1, 2.1, spread);
            double px = round((x-reduction.size()/2.+.5)*stonewidth*spreadx+p.w/2);
            double py = round((-(y-shifty)-.5)*stonewidth+p.h);
            if(y == 0 && x == diff_index)
                latex.mult_alpha(1-weight);
            p.copy(latex, px-latex.w/2, py-latex.h/2, 1);
        }
    }
}

C4Scene::C4Scene(const json& config, const json& contents, MovieWriter* writer) : SequentialScene(config, contents, writer) {
    vector<json> sequence_json = contents["sequence"];
    for (int i = 1; i < sequence_json.size()-1; i+=2) {
        cout << "constructing board " << i << endl;
        json board = sequence_json[i];

        // Concatenate annotations into a single string
        string concatenatedAnnotations = "";
        if (board.contains("annotations")) {
            vector<string> annotations = board["annotations"];
            for (const auto& annotation : annotations) {
                concatenatedAnnotations += annotation;
            }
        }

        // Concatenate highlights into a single string
        string concatenatedHighlights = "";
        if (board.contains("highlight")) {
            vector<string> highlights = board["highlight"];
            for (const auto& highlight : highlights) {
                concatenatedHighlights += highlight;
            }
        }

        boards.push_back(Board(board["representation"], concatenatedAnnotations, concatenatedHighlights));
        names.push_back(board["name"]);
    }
    frontload_audio(contents, writer);
}

void C4Scene::render_non_transition(Pixels& p, int which) {
    p.fill(BLACK);
    Board b = boards[which];
    json curr = contents["sequence"][which*2+1];
    double threat_diagram = curr.contains("reduction") ? 1 : 0;
    double curr_spread = curr.value("spread", false) ? 1 : 0;
    render_c4_board(pix, b, static_cast<double>(time_in_this_block)/framerate, curr_spread, threat_diagram);
    Pixels board_title_pix = eqn_to_pix(latex_text(names[which]), pix.w / 640);
    pix.copy(board_title_pix, (pix.w - board_title_pix.w)/2, pix.h-pix.w/12, 1);
    if(threat_diagram > 0) render_threat_diagram(pix, curr["reduction"], curr["reduction_colors"], threat_diagram, -1, 0, curr_spread);
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

    double curr_spread         = (0                           >  which*2+1 || !contents["sequence"][which*2+1].value   ("spread", false)) ? 0 : 1;
    double next_spread         = (contents["sequence"].size() <= which*2+3 || !contents["sequence"][which*2+3].value   ("spread", false)) ? 0 : 1;
    double curr_threat_diagram = (0                           >  which*2+1 || !contents["sequence"][which*2+1].contains("reduction"    )) ? 0 : 1;
    double next_threat_diagram = (contents["sequence"].size() <= which*2+3 || !contents["sequence"][which*2+3].contains("reduction"    )) ? 0 : 1;
    double spread         = lerp(curr_spread        , next_spread        , smoother2(weight));
    double threat_diagram = lerp(curr_threat_diagram, next_threat_diagram, smoother2(weight));
    render_c4_board(pix, transition, 0, spread, threat_diagram);
    if(curr_threat_diagram == 1 && next_threat_diagram == 1){
        int diff_index = -1;
        for(int i = 0; i < contents["sequence"][which*2+1]["reduction"].size(); i++)
            if(contents["sequence"][which*2+1]["reduction"][i] != contents["sequence"][which*2+3]["reduction"][i]){
                diff_index = i;
                break;
            }
        render_threat_diagram(pix, contents["sequence"][which*2+1]["reduction"], contents["sequence"][which*2+1]["reduction_colors"], threat_diagram, diff_index, smoother2(weight), spread);
    }
    if(curr_threat_diagram == 1 && next_threat_diagram == 0){
        render_threat_diagram(pix, contents["sequence"][which*2+1]["reduction"], contents["sequence"][which*2+1]["reduction_colors"], threat_diagram, -1, 0, spread);
    }
    if(curr_threat_diagram == 0 && next_threat_diagram == 1){
        render_threat_diagram(pix, contents["sequence"][which*2+3]["reduction"], contents["sequence"][which*2+3]["reduction_colors"], threat_diagram, -1, 0, spread);
    }
    
    string curr_title = "\\text{" + names[which] + "}";
    string next_title = (which == contents["sequence"].size() - 1)?"\\text{}":"\\text{" + names[which+1] + "}";

    Pixels curr_title_pix = eqn_to_pix(curr_title, pix.w / 640);
    Pixels next_title_pix = eqn_to_pix(next_title, pix.w / 640);
    
    if(next_title == curr_title)
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1);
    else{
        pix.copy(curr_title_pix, (pix.w - curr_title_pix.w)/2, pix.h-pix.w/12, 1-weight);
        pix.copy(next_title_pix, (pix.w - next_title_pix.w)/2, pix.h-pix.w/12, weight);
    }
}
