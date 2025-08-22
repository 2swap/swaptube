#pragma once

#include "../../io/VisualMedia.cpp"
#include "../../DataObjects/Connect4/C4Board.cpp"
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
    bool is_transition = false;
    // non-transitions
    string representation = "";
    C4Board board;
    double stonewidth;
    string annotations = empty_annotations;

    // transitions
    C4Board b2;
    string annotations2 = empty_annotations;

public:
    C4Scene(const string& rep, const double width = 1, const double height = 1)
        :Scene(width, height), representation(rep), board(rep) {}

    void stage_transition(string final_rep){
        is_transition = true;
        b2 = C4Board(final_rep);
        annotations2 = annotations;
    }

    void post_transition(){
        board = b2;
        annotations = annotations2;
        is_transition = false;
        representation = b2.representation;
    }

    void set_annotations(string s){annotations = s;}
    string get_annotations(){return annotations;}

    void unannotate() {
        annotations = empty_annotations;
    }

    void interpolate(){
        double w = state["microscene_fraction"];
        board = c4lerp(board, b2, w);
        annotations = w>.5?annotations:annotations2;
    }

    char get_annotation(int x, int y){
        return annotations[x+(board.BOARD_HEIGHT-1-y)*board.BOARD_WIDTH];
    }

    void draw_c4_disc(int disc_x, int disc_y){
        int col_id = board.piece_code_at(disc_x, board.BOARD_HEIGHT-1-disc_y);
        double px, py;
        get_disc_screen_coordinates(disc_x, disc_y, px, py);
        int col = vector<int>{C4_EMPTY, C4_RED, C4_YELLOW}[col_id];

        if(col_id != 0){
            double piece_fill_radius = ceil(stonewidth*.4);
            double piece_stroke_radius = ceil(stonewidth*(.47));
            pix.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            pix.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, OPAQUE_BLACK, 0.8));
        }

        char annotation = get_annotation(disc_x, disc_y);
        if(annotation != ' ') {
            ScalingParams sp(stonewidth, stonewidth);
            Pixels latex = latex_to_pix(string("\\text{")+annotation+"}", sp);
            pix.overlay(latex, px-latex.w/2, py-latex.h/2);
        } else {
            if(col_id == 0) pix.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, 0xff666666);
        }
    }

    void get_disc_screen_coordinates(int stonex, int stoney, double& px, double& py){
        px = round(( stonex - board.BOARD_WIDTH /2.+.5)*stonewidth+pix.w/2);
        py = round((-stoney + board.BOARD_HEIGHT/2.-.5)*stonewidth+pix.h/2);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"microblock_fraction"};
    }
    void mark_data_unchanged() override { }
    void change_data() override { } // No DataObjects
    bool check_if_data_changed() const override { return false; } // No DataObjects

    void draw() override{
        if (is_transition) interpolate();
        stonewidth = min(get_width(), get_height())/10;
        for(int stonex = 0; stonex < board.BOARD_WIDTH; stonex++)
            for(int stoney = 0; stoney < board.BOARD_HEIGHT; stoney++){
                double px = 0, py = 0;
                draw_c4_disc(stonex, stoney);
            }
        //if(done_scene && is_transition) post_transition();
    }
};
