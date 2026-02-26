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
    C4GraphScene(shared_ptr<Graph> g, bool surfaces_on, const string& rep, const C4BranchMode mode, const double width = 1, const double height = 1);

    int get_edge_color(const Node& node, const Node& neighbor);

private:
    string root_node_representation;
};
