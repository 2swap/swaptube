#pragma once

#include <array>
#include <memory>
#include <string>
#include "../../IO/Latex.h"
#include "../../IO/PNG.h"
#include "../Math/MandelbrotScene.h"
#include "../../Core/State/StateManager.h"

void stripey_effect(Pixels& in, Pixels& out, const float amount);

class TwoswapScene : public MandelbrotScene {
private:
    DevicePointer latex_twoswap;
    DevicePointer latex_seef;
    DevicePointer latex_swaptube;

public:
    TwoswapScene(const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;
    void draw() override;
    std::string swaptube_commit_hash();
};
