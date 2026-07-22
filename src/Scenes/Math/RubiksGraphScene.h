#pragma once

#include "../Common/CompositeScene.h"

class RubiksGraphScene : public CompositeScene {
public:
    RubiksGraphScene(const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;
};
