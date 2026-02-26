#pragma once

#include "../Scene.h"
#include "../../DataObjects/KlotskiBoard.h"
#include <string>
#include <vector>

int piece_color(int cell);

class KlotskiScene : public Scene {
private:
    KlotskiBoard kb;
    KlotskiMove staged_move{'.', 0, 0};

public:
    char highlight_char = '.';
    KlotskiScene(const KlotskiBoard& _kb, const double width = 1, const double height = 1);

    // No interactive state is needed for this static klotski scene.
    void on_end_transition_extra_behavior(const TransitionType tt) override;
    KlotskiBoard copy_staged_board();
    KlotskiBoard copy_board();
    const StateQuery populate_state_query() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    void stage_move(const KlotskiMove& km);

    void stage_random_move();

    // The draw method renders the board.
    void draw() override;
};
