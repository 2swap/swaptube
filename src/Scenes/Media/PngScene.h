#pragma once

#include <string>
#include "../Scene.h"

using std::string;

class PngScene : public Scene {
public:
    PngScene(string pn, const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    string picture_name;
};
