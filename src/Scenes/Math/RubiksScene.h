#pragma once
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Rubiks.h"

class RubiksScene : public ThreeDimensionScene {
public:
    RubiksScene(const CubeStickerPattern& pattern, const vec2& dimensions = vec2(1, 1));
    RubiksScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
    const StateQuery populate_state_query() const override;
    void exec_move_from_slice(const std::string& token);
    Rubiks* the_cube;

protected:
    void on_end_transition_extra_behavior(const TransitionType tt) override;

private:
    quat rotation_quat;
    Cut cut;
    char d_stickers[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE];
};




