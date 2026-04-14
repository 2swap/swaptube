#pragma once

#include "../Scene.h"

class PauseScene : public Scene {
public:
    PauseScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;
};
