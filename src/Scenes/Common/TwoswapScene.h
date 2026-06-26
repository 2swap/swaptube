#pragma once

#include <array>
#include <memory>
#include <string>
#include "../../IO/SVG.h"
#include "../../IO/PNG.h"
#include "../Math/MandelbrotScene.h"
#include "../../Core/State/StateManager.h"

void stripey_effect(Pixels& in, Pixels& out, const float amount);

class TwoswapScene : public MandelbrotScene {
    uint32_t* twoswap = nullptr;
    uint32_t* seef = nullptr;
    uint32_t* swaptube = nullptr;

    ivec2 twoswap_wh, seef_wh, swaptube_wh;

public:
    TwoswapScene(const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;
    void draw() override;
    std::string swaptube_commit_hash();
};
