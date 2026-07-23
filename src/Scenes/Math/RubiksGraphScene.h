#pragma once

#include "../Common/CompositeScene.h"
#include "RubiksScene.h"

class RubiksGraphScene : public CompositeScene {
public:
    RubiksGraphScene(const vec2& dimensions = vec2(1, 1));

    void add_cube(const string& alg);
    void draw();
private:
    unordered_map<double, shared_ptr<RubiksScene>> cubes;
};
