#pragma once

#include <string>
#include <unordered_set>

#include "../Common/ThreeDimensionScene.h"

class GeographyScene : public ThreeDimensionScene {
private:
    std::unordered_set<std::string> manifold_names;
    uint32_t* d_map;
    int map_w;
    int map_h;

public:
    GeographyScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

    ~GeographyScene();
};
