#pragma once

#include "../Scene.h"


class PermutationScene : public Scene {
    public:
        PermutationScene(const vec2& dimensions = vec2(1, 1));
        void draw() override;
        const StateQuery populate_state_query() const override;
};