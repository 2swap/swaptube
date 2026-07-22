#include "RubiksGraphScene.h"
#include "GraphScene.h"
#include "RubiksScene.h"

RubiksGraphScene::RubiksGraphScene(const vec2& dimensions)
    : CompositeScene(dimensions) {
    manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });

    shared_ptr<GraphScene> gs = make_shared<GraphScene>();
    gs->manager.set({
        {"repel", "1"},
        {"attract", "1"},
        {"decay", ".95"},
        {"physics_multiplier", "1"},
        {"dimensions", "3"},
        {"edge_weights_size", "0"},
        {"node_labels_size", "1"},
        {"midpoint_multiplier", "1"},
        {"q1", "[q1]"},
        {"qi", "[qi]"},
        {"qj", "[qj]"},
        {"qk", "[qk]"},
        {"x", "[x]"},
        {"y", "[y]"},
        {"z", "[z]"},
        {"d", "[d]"},
    });
    CompositeScene::add_scene(gs, "gs");

    unordered_map<double, shared_ptr<RubiksScene>> cubes;
}

void add_cube(const string& alg) {
    shared_ptr<RubiksScene> rs = make_shared<RubiksScene>(alg);
    CompositeScene::add_scene(rs, "rs");
}

const StateQuery RubiksGraphScene::populate_state_query() const {
    return CompositeScene::populate_state_query();
}
