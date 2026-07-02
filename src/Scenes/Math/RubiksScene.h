#pragma once
#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Rubiks.h"

class RubiksScene : public ThreeDimensionScene {
public:
    RubiksScene(const vec2& dimensions = vec2(1, 1));
    void draw() override;
    Rubiks* m_data;
    const StateQuery populate_state_query() const override;
};




