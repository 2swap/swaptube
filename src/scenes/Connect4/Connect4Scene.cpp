#pragma once

#include "../../io/visual_media.cpp"
#include "c4.h"
#include <algorithm>

inline int C4_RED           = 0xffff0000;
inline int C4_YELLOW        = 0xffffff00;
inline int C4_EMPTY         = 0xff222222;

inline string empty_annotations = "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       ";

class C4Scene : public Scene {
private:
    // non-transitions
    string representation = "";
    Board board;
    double stonewidth;
    string annotations = empty_annotations;

    // transitions
    Board b2;
    string annotations2;

public:
    C4Scene(const string& rep, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        :Scene(width, height), representation(rep), board(rep), stonewidth(min(width, height)/10.) {}

    void stage_transition(string final_rep){
        state_query.insert(final_rep);
        is_transition = true;
        b2 = Board(final_rep);
        annotations2 = annotations;
    }

    void post_transition(){
        board = b2;
        annotations = annotations2;
        is_transition = false;
        representation = b2.representation;
        state_query.erase(final_rep);
    }

    void set_annotations(string s){annotations = s; rendered = false;}
    string get_annotations(){return annotations;}

    void unannotate() {
        annotations = empty_annotations;
        rendered = false;
    }

    void interpolate(){
        double w = state["transition_fraction"];
        board = c4lerp(board, b2, w);
        annotations = w>.5?annotations:annotations2;
        rendered = false;
    }

    char get_annotation(int x, int y){
        return annotations[x+(HEIGHT-1-y)*WIDTH];
    }

    void draw_c4_disc(int disc_x, int disc_y){
        int col_id = board.grid[disc_y][disc_x];
        bool blink = board.blink[disc_y][disc_x];
        double px, py;
        get_disc_screen_coordinates(disc_x, disc_y, px, py);
        int col = vector<int>{C4_EMPTY, C4_RED, C4_YELLOW}[col_id];

        if(col_id != 0){
            double piece_fill_radius = ceil(stonewidth*.4);
            double piece_stroke_radius = ceil(stonewidth*(.47));
            double blink_radius = ceil(stonewidth*(.2));
            pix.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            pix.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, OPAQUE_BLACK, blink?.8:.4));
        }

        char annotation = get_annotation(disc_x, disc_y);
        if(annotation != ' ') {
            ScalingParams sp(stonewidth, stonewidth);
            Pixels latex = eqn_to_pix(latex_text(string()+annotation), sp);
            pix.overlay(latex, px-latex.w/2, py-latex.h/2);
        } else {
            if(col_id == 0) pix.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, 0xff666666);
        }
    }

    void get_disc_screen_coordinates(int stonex, int stoney, double& px, double& py){
        px = round(( stonex - WIDTH /2.+.5)*stonewidth+pix.w/2);
        py = round((-stoney + HEIGHT/2.-.5)*stonewidth+pix.h/2);
    }

    void render_c4(){
        pix.fill(TRANSPARENT_BLACK);
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++){
                double px = 0, py = 0, stonewidth = 0;
                draw_c4_disc(stonex, stoney);
            }
    }

    void draw() override{
        if (is_transition) interpolate();
        render_c4();
        //if(done_scene && is_transition) post_transition();
    }
};
