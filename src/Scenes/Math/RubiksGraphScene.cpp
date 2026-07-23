#include "RubiksGraphScene.h"
#include "GraphScene.h"

RubiksGraphScene::RubiksGraphScene(const vec2& dimensions)
    : CompositeScene(dimensions) {
    manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "10"},
    });

    shared_ptr<GraphScene> gs = make_shared<GraphScene>();
    gs->graph->add_node(1);
    gs->graph->add_node(2);
    gs->graph->add_edge(1,2);
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
}

void RubiksGraphScene::add_cube(const string& alg) {
    shared_ptr<RubiksScene> rs = make_shared<RubiksScene>(alg, vec2(.1, .1));
    double hash = 0.9;
    string key = "rs" + to_string(hash);
    //CompositeScene::add_scene_fade_in(MICRO, rs, key, vec2(-1, -1), .5, true);
    cubes[hash] = rs;
    Rubiks* cube = rs->the_cube;
}

void RubiksGraphScene::draw() {
    shared_ptr<GraphScene> gs = dynamic_pointer_cast<GraphScene>(subscenes["gs"]);
    Graph* g = gs->graph;
    for(auto& pair : g->nodes){
        double d = pair.first;
        cout << d << endl;
    }
    CompositeScene::draw();
}
