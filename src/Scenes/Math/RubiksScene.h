#pragma once
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Rubiks.h"

class RubiksScene : public ThreeDimensionScene {
public:
    RubiksScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
    const StateQuery populate_state_query() const override;
    void exec_move_from_slice(const char move, const int depth);
private:
    quat rotation_quat;
    Cut cut;
    Rubiks* the_cube;
};




