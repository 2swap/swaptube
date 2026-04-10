#pragma once
#include "../Common/SuperScene.h"
#include <memory>
#include <string>

class AlphaFilterScene : public SuperScene {
public:
    AlphaFilterScene(
        std::shared_ptr<Scene> sc,
        Color preserve_col,
        const bool inv = false,
        const vec2& dimensions = vec2(1, 1)
    );

    void draw() override;

private:
    Color preserve_col;
    bool inversed;
};
