#pragma once

#include "../Common/CompositeScene.h"
#include "RubiksScene.h"
#include "GraphScene.h"

class RubiksGraphScene : public CompositeScene {
public:
    RubiksGraphScene(const vec2& dimensions = vec2(1, 1));

    void add_cube(const string& alg);
    void add_children();
    void draw();
private:
    shared_ptr<GraphScene> gs;
    unordered_map<double, shared_ptr<RubiksScene>> cubes;
    unordered_map<double, string> algs;
};
