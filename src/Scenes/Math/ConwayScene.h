#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/ConwayGrid.h"
#include "../../Core/State/StateManager.h"

class ConwayScene : public CoordinateScene {
private:
    const vec2 size_bitboards = vec2(10000,10000);
    ConwayGrid conway_grid;

public:
    ConwayScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    void change_data() override;

    bool check_if_data_changed() const override;
};
