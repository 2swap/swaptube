#pragma once

#include <array>
#include <memory>
#include <string>
#include "../../IO/VisualMedia.h"
#include "../Math/MandelbrotScene.h"
#include "../../Core/State/StateManager.h"

void stripey_effect(Pixels& in, Pixels& out, const float amount);

class TwoswapScene : public MandelbrotScene {
public:
    TwoswapScene(const double width = 1, const double height = 1);

    const StateQuery populate_state_query() const override;
    void draw() override;
    std::string swaptube_commit_hash();
};
