#pragma once
#include "../../IO/VisualMedia.h"
#include "../Scene.h"
#include <vector>
#include <string>

class WhitePaperScene : public Scene {
public:
    WhitePaperScene(const string& prefix, const string& author, const vector<int>& page_numbers, const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    const string prefix;
    const string author;
    const vector<int> page_numbers;
};
