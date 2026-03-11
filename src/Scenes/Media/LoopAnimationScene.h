#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "../../IO/VisualMedia.h"
#include "../Scene.h"

class LoopAnimationScene : public Scene {
public:
    LoopAnimationScene(const std::vector<std::string>& pn, const vec2& dimensions = vec2(1, 1));

    bool check_if_data_changed() const override;
    void mark_data_unchanged() override;
    void change_data() override;

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    const std::vector<std::string> picture_names;
    vec2 memo_size;
    std::unordered_map<std::string, Pixels> pixel_cache;
};
