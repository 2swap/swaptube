#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../../DataObjects/Connect4/C4Board.cpp"
#include "../../DataObjects/Connect4/C4Physics.cpp"
#include <algorithm>

inline int C4_RED           = 0xffff0000;
inline int C4_YELLOW        = 0xffffff00;
inline int C4_EMPTY         = 0xff666699;

inline string empty_annotations = "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       ";

class C4Scene : public Scene {
private:
    C4Physics board;
    string annotations = empty_annotations;
    string representation;

public:
    C4Scene(const string& rep, const double width = 1, const double height = 1)
        :Scene(width, height), board(7, 6), representation(rep){
            board.append_to_queue(rep);
    }

    void undo(int steps) {
        board.undo(steps);
        representation = representation.substr(0, representation.size()-steps);
    }

    void play(const string& rep){
        board.append_to_queue(rep);
        representation += rep;
    }

    void set_annotations(string s){annotations = s;}
    string get_annotations(){return annotations;}
    void unannotate() {annotations = empty_annotations;}

    char get_annotation(int x, int y){
        return annotations[x+(board.h-1-y)*board.w];
    }

    void draw_empty_board(){
        for(int x=0; x<board.w; x++){
            for(int y=0; y<board.h; y++){
                double px, py;
                get_disc_screen_coordinates(x, y, px, py);
                pix.fill_ellipse(px, py, get_stone_width()*.3, get_stone_width()*.3, C4_EMPTY);
            }
        }
    }

    void highlight_winning_discs(){
        double highlight = state["highlight"];
        if(highlight <= 0.1) return;

        C4Board b(representation);
        Bitboard winning_discs = b.winning_discs();
        for (int x=0; x<board.w; x++) {
            for (int y=0; y<board.h; y++) {
                if(!bitboard_at(winning_discs, x, y)) continue;
                double px, py;
                get_disc_screen_coordinates(x, y, px, py);
                double stone_width = get_stone_width();
                double ellipse_width = stone_width * .3 * highlight;
                pix.fill_ellipse(px, py, ellipse_width, ellipse_width, OPAQUE_WHITE);
            }
        }
    }

    void draw_c4_disc(int disc_x, double disc_y, bool is_red) {
        double px, py;
        get_disc_screen_coordinates(disc_x, disc_y, px, py);
        int col = is_red ? C4_RED : C4_YELLOW;

        double stone_width = get_stone_width();
        double piece_fill_radius = stone_width*.35;
        double piece_stroke_radius = stone_width*.47;
        pix.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, colorlerp(col, OPAQUE_BLACK, 0.5));
        pix.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, OPAQUE_BLACK, 0.0));
    }

    void draw_annotations(){
        for(int x=0; x<board.w; x++){
            for(int y=0; y<board.h; y++){
                char annotation = get_annotation(x, y);
                if(annotation != ' ') {
                    double px, py;
                    get_disc_screen_coordinates(x, y, px, py);
                    ScalingParams sp(get_stone_width(), get_stone_width());
                    Pixels latex = latex_to_pix(string("\\text{")+annotation+"}", sp);
                    pix.overlay(latex, px-latex.w/2, py-latex.h/2);
                }
            }
        }
    }

    void get_disc_screen_coordinates(int stonex, double stoney, double& px, double& py){
        px = round(( stonex - board.w /2.+.5)*get_stone_width()+pix.w/2);
        py = round((-stoney + board.h /2.-.5)*get_stone_width()+pix.h/2);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"highlight"};
    }

    void mark_data_unchanged() override { board.mark_unchanged(); }
    void change_data() override { board.iterate_physics(); }
    bool check_if_data_changed() const override { return board.has_been_updated_since_last_scene_query(); }

    double get_stone_width() const {
        return min(get_width(), get_height())/10;
    }

    void draw() override{
        draw_empty_board();
        for(const Disc& disc : board.discs){
            draw_c4_disc(disc.x, disc.py, disc.index%2==0);
        }
        draw_annotations();
    }
};
