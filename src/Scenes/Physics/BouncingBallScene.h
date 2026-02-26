#pragma once

#include <algorithm>
#include "../../DataObjects/BouncingBalls.h"
#include "../Common/CoordinateScene.h"

class BouncingBallScene : public CoordinateScene {
private:
    BouncingBalls balls;

public:
    BouncingBallScene(
        const int n,
        const double simulation_width = 10,
        const double simulation_height = 10,
        const double width = 1,
        const double height = 1
    );

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    void draw() override;
};
