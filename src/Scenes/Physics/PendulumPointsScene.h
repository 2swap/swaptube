#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/Pendulum.h"
#include "../../Host_Device_Shared/vec.h"

class PendulumPointsScene : public CoordinateScene {
public:
    PendulumPointsScene(const PendulumGrid& pg, const vec2& dimensions = vec2(1, 1));

    void draw() override;
    void render_points();

    const StateQuery populate_state_query() const override;

private:
    PendulumGrid grid;
};
