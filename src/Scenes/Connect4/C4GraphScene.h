#pragma once

#include "../Math/GraphScene.h"
#include "../../DataObjects/Connect4/C4Board.h"
#include "C4Scene.h"
#include <memory>
#include <string>

using std::shared_ptr;
using std::string;

class C4GraphScene : public GraphScene {
public:
    C4GraphScene(shared_ptr<Graph> g, const string& rep, const C4BranchMode mode, const vec2& dimensions = vec2(1, 1));

private:
    string root_node_representation;
};
