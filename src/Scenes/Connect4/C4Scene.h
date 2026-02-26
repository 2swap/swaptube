#pragma once

#include "../../IO/VisualMedia.h"
#include "../../DataObjects/Connect4/C4Board.h"
#include "../../DataObjects/Connect4/C4Physics.h"
#include <algorithm>
#include <string>
#include <memory>

inline int C4_RED           = 0xffff0000;
inline int C4_RED_DARK      = 0xff7f0000;
inline int C4_YELLOW        = 0xffffff00;
inline int C4_YELLOW_DARK   = 0xff7f7f00;
inline int C4_EMPTY         = 0xff444466;

inline std::string empty_annotations = "       "
                                       "       "
                                       "       "
                                       "       "
                                       "       "
                                       "       ";

class C4Scene : public Scene {
private:
    C4Physics board;
    std::string annotations;
    std::string representation;

public:
    C4Scene(const std::string& rep, const double width = 1, const double height = 1);

    void undo(int steps);
    void use_up_queue();
    void flush_queue_undo_all();
    void play(const std::string& rep);

    void set_annotations(std::string s, TransitionType tt);
    void set_annotations_from_steadystate(TransitionType tt);
    std::string get_annotations();
    void clear_annotations(TransitionType tt);

    char get_annotation(int x, int y);

    void set_fast_mode(bool fast);

    void draw_empty_board();
    void highlight_winning_discs();
    void draw_c4_disc(int disc_x, double disc_y, bool is_red);
    void draw_annotations();
    void get_disc_screen_coordinates(int stonex, double stoney, double& px, double& py);

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    double get_stone_width() const;

    void draw() override;
};
