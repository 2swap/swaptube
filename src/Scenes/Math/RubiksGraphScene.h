#pragma once

#include "../Common/CompositeScene.h"
#include "RubiksScene.h"
#include "GraphScene.h"
#include <unordered_set>

class RubiksGraphScene : public CompositeScene {
public:
    RubiksGraphScene(const vec2& dimensions = vec2(1, 1));

    void add_cube(const string& alg, bool cube_or_not);
    void add_cube(const CubeStickerPattern& pattern, bool cube_or_not);
    void add_children(unordered_set<string> move_set, bool cube_or_not);
    void draw();
    const StateQuery populate_state_query() const override;
    shared_ptr<GraphScene> gs;
private:
    const int cube_size;
    unordered_map<double, shared_ptr<RubiksScene>> cubes;
    //unordered_map<double, string> algs;
    unordered_map<double, CubeStickerPattern> patterns;
};
