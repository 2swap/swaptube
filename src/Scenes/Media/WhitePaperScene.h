#pragma once
#include "../../IO/PNG.h"
#include "../../IO/Latex.h"
#include "../Scene.h"
#include <vector>
#include <string>

class WhitePaperScene : public Scene {
public:
    WhitePaperScene(const string& prefix, const string& author, const vector<int>& page_numbers, const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    shared_ptr<DevicePointer> author_pixels;
    const string prefix;
    const string author;
    const vector<int> page_numbers;
};
