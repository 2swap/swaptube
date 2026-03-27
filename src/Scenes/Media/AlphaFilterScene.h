#pragma once
#include "../Common/SuperScene.h"
#include <memory>
#include <string>

class AlphaFilterScene : public SuperScene {
public:
    AlphaFilterScene(
        std::shared_ptr<Scene> sc,
        const unsigned int preserve_col,
        const bool inv = false,
        const vec2& dimensions = vec2(1, 1)
    );

    void draw() override;

private:
    unsigned int preserve_col;
    unsigned int inversed;
};
