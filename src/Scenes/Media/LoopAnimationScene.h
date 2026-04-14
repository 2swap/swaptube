#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "../../IO/VisualMedia.h"
#include "../Scene.h"

class LoopAnimationScene : public Scene {
public:
    LoopAnimationScene(const std::vector<std::string>& pn, const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    const std::vector<std::string> picture_names;
    int memo_w = 0;
    int memo_h = 0;
    std::unordered_map<std::string, Pixels> pixel_cache;
};
