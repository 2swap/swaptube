#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"

#define GRAPH_WIDTH 10

double a_star_heuristic(vec4 node, vec4 goal) {
    return length(node - goal);
}

void color_graph_from_a_star(shared_ptr<Graph> g,
        const std::unordered_set<double>& open_set,
        const std::unordered_map<double, double>& came_from,
        double current
) {
    cout << "Coloring white" << endl;
    for(auto& [hash, node] : g->nodes) {
        node.color = 0xffffffff; // white
    }

    cout << "Coloring open set red" << endl;
    for(double hash : open_set) {
        g->nodes.find(hash)->second.color = 0xffff0000;
    }

    // Reconstruct path from current to start
    cout << "Coloring path from current to start" << endl;
    double hash = current;
    while(came_from.find(hash) != came_from.end()) {
        g->nodes.find(hash)->second.color = 0xff0000ff;
        hash = came_from.at(hash);
    }
    g->nodes.find(hash)->second.color = 0xff0000ff;
    cout << "Done coloring" << endl;
}

void run_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs) {
    double start = HashableString("0").get_hash();
    double goal = HashableString("99").get_hash();
    cout << "Running Dijkstra's algorithm from " << start << " to " << goal << endl;

    // Priority queue for A* algorithm
    std::unordered_set<double> open_set;
    open_set.insert(start);

    std::unordered_map<double, double> came_from;

    std::unordered_map<double, double> g_score;
    for(auto& [hash, node] : g->nodes) {
        g_score[hash] = std::numeric_limits<double>::infinity();
    }
    g_score[start] = 0;

    while(open_set.size() > 0) {
        double current = -1;
        double current_g_score = std::numeric_limits<double>::infinity();
        for(double hash : open_set) {
            if(g_score[hash] < current_g_score) {
                current_g_score = g_score[hash];
                current = hash;
            }
        }

        color_graph_from_a_star(g, open_set, came_from, current);
        gs->render_microblock();

        if(current == goal) {
            cout << "Reached goal!" << endl;
            break;
        }

        open_set.erase(current);
        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            double tentative_g_score = g_score[current] + 1; // Assuming uniform cost

            if(tentative_g_score < g_score[neighbor]) {
                came_from[neighbor] = current;
                g_score[neighbor] = tentative_g_score;
                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                }
            }
        }
    }
}

void run_a_star(shared_ptr<Graph> g, shared_ptr<GraphScene> gs) {
    double start = HashableString("0").get_hash();
    double goal = HashableString("99").get_hash();

    // Priority queue for A* algorithm
    std::unordered_set<double> open_set;
    open_set.insert(start);

    std::unordered_map<double, double> came_from;

    std::unordered_map<double, double> g_score;
    std::unordered_map<double, double> f_score;
    for(auto& [hash, node] : g->nodes) {
        g_score[hash] = std::numeric_limits<double>::infinity();
        f_score[hash] = std::numeric_limits<double>::infinity();
    }
    g_score[start] = 0;
    f_score[start] = a_star_heuristic(g->nodes.find(start)->second.position, g->nodes.find(goal)->second.position);

    while(open_set.size() > 0) {
        double current = -1;
        double current_f_score = std::numeric_limits<double>::infinity();
        for(double node : open_set) {
            if(f_score[node] < current_f_score) {
                current_f_score = f_score[node];
                current = node;
            }
        }

        color_graph_from_a_star(g, open_set, came_from, current);
        gs->render_microblock();

        if(current == goal) {
            break;
        }

        open_set.erase(current);

        // Get neighbors of current node
        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            double tentative_g_score = g_score[current] + 1; // Assuming uniform cost

            if(tentative_g_score < g_score[neighbor]) {
                came_from[neighbor] = current;
                g_score[neighbor] = tentative_g_score;
                f_score[neighbor] = g_score[neighbor] + a_star_heuristic(g->nodes.find(neighbor)->second.position, g->nodes.find(goal)->second.position);
                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                }
            }
        }
    }
}

void render_video() {
    cout << "Building graph..." << endl;
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"decay",".8"},
        {"dimensions","2"},
        {"d", "1"},
        {"surfaces_opacity","1"},
        {"points_opacity","1"},
        {"points_radius_multiplier","3"},
        {"physics_multiplier","5"},
    });

    stage_macroblock(SilenceBlock(3), 75);

    for(int xplusy = 0; xplusy <= (GRAPH_WIDTH - 1) * 2; xplusy++) {
        for(int y = 0; y <= xplusy; y++) {
            int x = xplusy - y;
            int i = x + y * GRAPH_WIDTH;
            bool is_i_prime = i > 1;
            for(int d = 2; d <= sqrt(i); d++) {
                if(i % d == 0) {
                    is_i_prime = false;
                    break;
                }
            }
            if(is_i_prime) continue;
            if(x < 0 || x >= GRAPH_WIDTH || y < 0 || y >= GRAPH_WIDTH) continue;
            HashableString node(to_string(i));
            HashableString left(to_string(i-1));
            HashableString up(to_string(i-GRAPH_WIDTH));
            vector<double> neighbor_hashes;
            if(x > 0) neighbor_hashes.push_back(left.get_hash());
            if(y > 0) neighbor_hashes.push_back(up.get_hash());
            g->add_node_with_neighbors(new HashableString(to_string(i)), neighbor_hashes);
            gs->render_microblock();
        }
    }

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    /*
    stage_macroblock(SilenceBlock(4), 100);
    run_dijkstra(g, gs);
    while(remaining_microblocks_in_macroblock) {
        gs->render_microblock();
    }

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(4), 100);
    run_a_star(g, gs);
    while(remaining_microblocks_in_macroblock) {
        gs->render_microblock();
    }

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();
    */

    shared_ptr<PngScene> ps = make_shared<PngScene>("nyc");
    gs->add_surface(Surface("ps"), ps);
    gs->manager.set("ps.opacity", "0");
    gs->manager.transition(MICRO, "ps.opacity", "0.2");
    CompositeScene cs;
    cs.add_scene(gs, "gs");

    stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    quat down = quat(1, -1, 0, 0);
    quat right = quat(1, 0, 0, .2);
    quat rot = down * right;

    gs->manager.transition(MICRO, {
        {"q1", to_string(rot.u)},
        {"qi", to_string(rot.i)},
        {"qj", to_string(rot.j)},
        {"qk", to_string(rot.k)},
        {"x","0"},
        {"y","0"},
        {"z",".1"},
    });

    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    gs->manager.set("physics_multiplier", "0");
    gs->manager.set("centering_strength", "0");

    stage_macroblock(SilenceBlock(3), 1);
    for(auto& [hash, node] : g->nodes) {
        int zdiff = rand() % 15;
        gs->transition_node_position(MICRO, hash, vec4(0,0,zdiff,0));
    }
    cs.render_microblock();

    right = quat(1, 0, 0, .8);
    rot = down * right;

    gs->manager.transition(MICRO, {
        {"q1", to_string(rot.u)},
        {"qi", to_string(rot.i)},
        {"qj", to_string(rot.j)},
        {"qk", to_string(rot.k)},
        {"x","0"},
        {"y","0"},
        {"z",".1"},
    });

    stage_macroblock(SilenceBlock(3), 1);
    cs.render_microblock();
}
