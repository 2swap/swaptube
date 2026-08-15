#pragma once

#include <string>
#include "../Scene.h"

using std::string;

class PngScene : public Scene {
public:
    PngScene(string pn, const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    string picture_name;
};
