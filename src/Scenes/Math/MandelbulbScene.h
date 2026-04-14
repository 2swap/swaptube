#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

class MandelbulbScene : public Scene {
public:
    MandelbulbScene(const vec2& dimensions = vec2(1,1));
    const StateQuery populate_state_query() const override;
    void draw() override;
};
