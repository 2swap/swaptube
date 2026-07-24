#include "RubiksGraphScene.h"
#include <unordered_map>
#include <unordered_set>

const unordered_set<string> move_set = {"R", "U", "F", "R'", "U'", "F'", "L", "D", "B", "L'", "D'", "B'"};

RubiksGraphScene::RubiksGraphScene(const vec2& dimensions)
    : CompositeScene(dimensions), cube_size(3) {
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
        {"rubiks_scene_size", "0.1"}
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

const StateQuery RubiksGraphScene::populate_state_query() const{
    StateQuery ret = CompositeScene::populate_state_query();
    ret.insert("rubiks_scene_size");
    return ret;
}

void RubiksGraphScene::add_children(unordered_set<string> move_set, bool cube_or_not) {
    Graph* g = gs->graph;
    auto nodes = g->nodes; // true copy
    for(auto& pair : nodes) {
        double hash = pair.first;
        for(string s : move_set) {
            Rubiks child(patterns[hash]);
            child.exec(s);
            double child_hash = child.get_hash(cube_size);
            if (!g->node_exists(child_hash)){
                patterns[child_hash] = child.pattern;
                add_cube(child.pattern, cube_or_not);
                cout << "+" << flush;
            }
        }
    }
}

void RubiksGraphScene::add_cube(const string& alg, bool cube_or_not) {
    Rubiks cube;
    cube.exec(alg);
    add_cube(cube.pattern, cube_or_not);
}

void RubiksGraphScene::add_cube(const CubeStickerPattern& pattern, bool cube_or_not) {
    Rubiks cube(pattern);
    double hash = cube.get_hash(cube_size);

    if (cube_or_not){
        shared_ptr<RubiksScene> rs = make_shared<RubiksScene>(pattern, vec2(0.001, 0.001));
        rs->manager.set({
            {"w", "[rubiks_scene_size]"},
            {"h", "[rubiks_scene_size]"},
            {"q1", "[q1]"},
            {"qi", "[qi] 0.25 +"},
            {"qj", "[qj] 0.25 -"},
            {"qk", "[qk]"},
            {"cube_size", to_string(cube_size)},
        });
        string key = "rs" + to_string(hash);
        CompositeScene::add_scene(rs, key);
        cubes[hash] = rs;
    }

    Graph* g = gs->graph;
    g->add_node(hash);
    gs->config->set_node_label(hash, to_string(hash));
    for(string s : move_set) {
        Rubiks child(pattern);
        child.exec(s);
        double child_hash = child.get_hash(cube_size);
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
        state.set(key+".x", (fraction.x));
        state.set(key+".y", (fraction.y));
    }
    CompositeScene::draw();
}
