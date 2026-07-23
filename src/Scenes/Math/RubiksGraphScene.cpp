#include "RubiksGraphScene.h"

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
        {"d", "40"},
        {"repel", "1"},
        {"attract", "1"},
        {"decay", ".95"},
        {"physics_multiplier", "1"},
        {"dimensions", "3"},
        {"edge_weights_size", "0"},
        {"node_labels_size", "1"},
        {"midpoint_multiplier", "1"},
    });

    gs = make_shared<GraphScene>();
    gs->manager.set({
        {"repel", "[repel]"},
        {"attract", "[attract]"},
        {"decay", "[decay]"},
        {"physics_multiplier", "[physics_multiplier]"},
        {"dimensions", "[dimensions]"},
        {"edge_weights_size", "[edge_weights_size]"},
        {"node_labels_size", "[node_labels_size]"},
        {"midpoint_multiplier", "[midpoint_multiplier]"},
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

void RubiksGraphScene::add_children() {
    Graph* g = gs->graph;
    auto nodes = g->nodes; // true copy
    for(auto& pair : nodes) {
        double hash = pair.first;
        string alg = algs[hash];
        for(string s : {"U", "R", "L", "B", "F", "D"}) {
            Rubiks child(3);
            string child_alg = (alg.size() != 0?alg+" ":"") + s + "2";
            child.exec(child_alg);
            if(alg == "F2" && s == "B") {cout << "F2 B2" << endl; child.print();}
            if(alg == "B2" && s == "F") {cout << "B2 F2" << endl; child.print();}
            double child_hash = child.get_hash();
            if (!g->node_exists(child_hash)){
                algs[child_hash] = child_alg;
                add_cube(algs[child_hash]);
            }
        }
    }
}

void RubiksGraphScene::add_cube(const string& alg) {
    Rubiks cube(3);
    cube.exec(alg);
    double hash = cube.get_hash();

    shared_ptr<RubiksScene> rs = make_shared<RubiksScene>(alg, vec2(.1, .1));
    string key = "rs" + to_string(hash);
    CompositeScene::add_scene_fade_in(MICRO, rs, key, vec2(-1, -1), .5, true);
    cubes[hash] = rs;

    Graph* g = gs->graph;
    g->add_node(hash);
    gs->config->set_node_label(hash, to_string(hash));
    for(string s : {"U", "R", "L", "B", "F", "D"}) {
        Rubiks child(3);
        string child_alg = (alg.size() != 0?alg+" ":"") + s + "2";
        child.exec(child_alg);
        double child_hash = child.get_hash();
        if (g->node_exists(child_hash)){
            g->add_edge(hash, child_hash);
        }
    }
}

void RubiksGraphScene::draw() {
    Graph* g = gs->graph;
    vec2 wh = get_width_height();
    for(auto& pair : g->nodes){
        double hash = pair.first;
        string key = "rs" + to_string(hash);
        vec3 position = pair.second.position;
        bool behind;
        vec2 pixel = gs->coordinate_to_pixel(position, behind);
        vec2 fraction = pixel / wh;
        manager.set({
            {key+".x", to_string(fraction.x)},
            {key+".y", to_string(fraction.y)}
        });
    }
    CompositeScene::draw();
}
