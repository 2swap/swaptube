#pragma once

#include "../../DataObjects/Pendulum.h"
#include "../Common/CoordinateScene.h"
#include <math.h>
#include <vector>
#include <list>
#include <utility>

class PendulumGridScene : public CoordinateScene {
public:
    PendulumGridScene(const vector<PendulumGrid>& pgv, const double width = 1, const double height = 1);

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    void draw_grid();

    void draw() override;

    void draw_pendulum_trail();

    bool momentum_mode = false;
private:
    vector<PendulumGrid> grids;
};
