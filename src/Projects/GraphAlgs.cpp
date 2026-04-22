#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"

#define GRAPH_WIDTH 10

double a_star_heuristic(vec4 node, vec4 goal) {
    return length(node - goal);
}

void color_graph_from_a_star(shared_ptr<Graph> g, shared_ptr<GraphScene> gs,
        const std::unordered_set<double>& open_set,
        const std::unordered_map<double, double>& came_from,
        double current
) {
    cout << "Coloring white" << endl;
    gs->config->transition_all_node_colors(MICRO, 0xffffffff);

    cout << "Coloring open set red" << endl;
    for(double hash : open_set) {
        gs->config->transition_node_color(MICRO, hash, 0xffff0000);
    }

    // Reconstruct path from current to start
    cout << "Coloring path from current to start" << endl;
    double hash = current;
    while(came_from.find(hash) != came_from.end()) {
        gs->config->transition_node_color(MICRO, hash, 0xff0000ff);
        hash = came_from.at(hash);
    }
    gs->config->transition_node_color(MICRO, hash, 0xff0000ff);
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

        color_graph_from_a_star(g, gs, open_set, came_from, current);
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

        color_graph_from_a_star(g, gs, open_set, came_from, current);
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

void slide3() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"physics_multiplier","0"},
        {"d", ".5"},
        {"points_radius_multiplier","3"},
        {"centering_strength","0"},
        {"x","0"},
        {"y","0"},
        {"z","0"},
    });

    // Load USA map
    shared_ptr<PngScene> ps = make_shared<PngScene>("nyc");
    gs->add_surface(Surface("ps"), ps);
    gs->manager.set("ps.opacity", "0");
    gs->manager.transition(MICRO, "ps.opacity", "0.2");

    stage_macroblock(FileBlock("If we simplified the US to a grid of 100 cities, it would look like this."), 100);

    // Plot cities as nodes and roads as edges, expanding east->west
    for(int i = 0; i < 100; i++) {
        double x = i / 50. - 1;
        double y = (rand() % 500)/250. - 1;
        double scale = 1.5;
        double hash = HashableString(to_string(i)).get_hash();
        g->add_node(new HashableString(to_string(i)));
        vec4 position = scale*vec4(x, y, 0, 0);
        g->move_node(hash, position);
        // Add edges to 3 nearest neighbors
        vector<double> neighbor_hashes;
        for(auto& [other_hash, node] : g->nodes) {
            if(other_hash == hash) continue;
            double distance = length(position - node.position);
            if(distance < .5) neighbor_hashes.push_back(other_hash);
        }
        for(int j = 0; j < min(3, (int)neighbor_hashes.size()); j++) {
            g->add_edge(hash, neighbor_hashes[j]);
        }
        gs->render_microblock();
    }

    // Highlight central park in Green and Golden Gate Park in Green
    // CP is 0, GG is 99
    gs->config->transition_node_color(MICRO, HashableString("0").get_hash(), 0xff00ff00);
    gs->config->transition_node_color(MICRO, HashableString("99").get_hash(), 0xff00ff00);

    // Quickly flicker a bunch of different paths which monotonically decrease in distance.

    stage_macroblock(FileBlock("The number of possible paths is on the order of 10 to the 24th power."), 30);
    double endpoint_hash = HashableString("99").get_hash();
    while(remaining_microblocks_in_macroblock) {
        cout << "Flickering path " << remaining_microblocks_in_macroblock << endl;
        gs->config->transition_all_edge_colors(MICRO, 0xffffffff);
        while(true) {
            double current_node = HashableString("0").get_hash();
            while(current_node != endpoint_hash) {
                double close_dist = length(g->nodes.find(current_node)->second.position - g->nodes.find(endpoint_hash)->second.position);
                vector<double> acceptable_neighbors;
                for(double neighbor : g->get_neighbors(current_node)) {
                    double distance = length(g->nodes.find(neighbor)->second.position - g->nodes.find(endpoint_hash)->second.position);
                    if(distance < close_dist) {
                        acceptable_neighbors.push_back(neighbor);
                    }
                }
                // Pick a random neighbor that’s closer to the target than the current node. If there are no such neighbors, break.
                if(acceptable_neighbors.size() == 0) {
                    break;
                }
                int rand_index = rand() % acceptable_neighbors.size();
                double neighbor = acceptable_neighbors[rand_index];
                gs->config->transition_edge_color(MICRO, current_node, neighbor, 0xffff0000);
                current_node = neighbor;
            }
            if(current_node == endpoint_hash) {
                break;
            }
        }
        gs->render_microblock();
    }
}

// lat long map
unordered_map<string, vec2> netherlands_cities = {
    {"Amsterdam", vec2(52.3676, 4.9041)},
    {"The Hague", vec2(52.0705, 4.3007)},
    {"Rotterdam", vec2(51.9244, 4.4777)},
    {"Utrecht", vec2(52.0907, 5.1214)},
    {"Breda", vec2(51.5719, 4.7683)},
    {"Eindhoven", vec2(51.4416, 5.4697)},
    {"Maastricht", vec2(50.8514, 5.6900)},
    {"Arnhem", vec2(51.9851, 5.8987)},
    {"Zwolle", vec2(52.5168, 6.0830)},
    {"Emmen", vec2(52.7795, 6.9061)},
    {"Groningen", vec2(53.2194, 6.5665)},
    {"Leeuwarden", vec2(53.2012, 5.7999)},
    {"'s-Hertogenbosch", vec2(51.6978, 5.3037)},
    {"Tillburg", vec2(51.5555, 5.0913)},
    {"Meppel", vec2(52.7917, 6.1789)},
};

unordered_map<string, string> netherlands_labels = {
    {"Amsterdam", "A"},
    {"The Hague", "H"},
    {"Rotterdam", "R"},
    {"Utrecht", "U"},
    {"Breda", "B"},
    {"Eindhoven", "I"},
    {"Maastricht", "M"},
    {"Arnhem", "N"},
    {"Zwolle", "Z"},
    {"Emmen", "E"},
    {"Groningen", "G"},
    {"Leeuwarden", "L"},
    {"'s-Hertogenbosch", "C"},
    {"Tillburg", "T"},
    {"Meppel", "P"},
};

vector<pair<string, string>> netherlands_edges_1 = {
    {"Rotterdam", "Utrecht"},
    {"Rotterdam", "Breda"},
    {"Breda", "Tillburg"},
    {"Tillburg", "Eindhoven"},
    {"Tillburg", "'s-Hertogenbosch"},
    {"Eindhoven", "Maastricht"},
    {"Eindhoven", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Utrecht"},
    {"Utrecht", "Arnhem"},
    {"Arnhem", "Zwolle"},
    {"Zwolle", "Emmen"},
    {"Zwolle", "Meppel"},
    {"Meppel", "Groningen"},
    {"Groningen", "Leeuwarden"},
    {"Meppel", "Leeuwarden"},
};

vector<pair<string, string>> netherlands_edges_2 = {
    {"Amsterdam", "The Hague"},
    {"Amsterdam", "Zwolle"},
    {"Amsterdam", "Utrecht"},
    {"The Hague", "Rotterdam"},
    {"The Hague", "Utrecht"},
    {"Rotterdam", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Arnhem"},
};

vector<pair<string, string>> netherlands_edges_3 = {
};

vector<uint32_t> colors_by_depth = {
    0xffff0000, // red
    0xffffa500, // orange
    0xffffff00, // yellow
    0xff00ff00, // green
    0xff00ffff, // cyan
    0xff0000ff, // blue
    0xff800080, // purple
    0xffff00ff, // magenta
    0xffff0080, // pink
};

void bfs(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, double target, int& depth) {
    if(border.size() == 0) {
        depth++;
        border = next_border;
        visited.insert(next_border.begin(), next_border.end());
        next_border.clear();
        return;
    }

    else {
        double node = *border.begin();
        if(node == target) {
            return;
        }
        unordered_set<double> neighbors = g->get_neighbors(node);
        for(double neighbor : neighbors) {
            if (next_border.find(neighbor) == next_border.end() && visited.find(neighbor) == visited.end()) {
                next_border.insert(neighbor);
                gs->config->transition_node_color(MICRO, neighbor, colors_by_depth[depth+1]);
                gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
                return;
            }
        }
        border.erase(node);
    }
}

void bfs_rung(shared_ptr<Graph> g, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, double target, int& depth, shared_ptr<GraphScene> gs) {
    while(border.size() > 0 && visited.find(target) == visited.end()) {
        bfs(g, gs, border, next_border, visited, target, depth);
        gs->render_microblock();
    }
    if(visited.find(target) == visited.end()) {
        bfs(g, gs, border, next_border, visited, target, depth);
    }
    gs->render_microblock();
}

void slide8() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"physics_multiplier","0"},
        {"d", "8"},
        {"points_radius_multiplier","3"},
        {"centering_strength","0"},
        {"x","0"},
        {"y","0"},
        {"z","0"},
    });

    // Load Netherlands map
    stage_macroblock(FileBlock("Here’s a simplified graph of the Netherlands:"), netherlands_cities.size());

    // Plot cities as nodes and roads as edges, expanding east->west
    for(auto& [city, coords] : netherlands_cities) {
        double lat = coords.x - netherlands_cities["Arnhem"].x;
        double lon = coords.y - netherlands_cities["Arnhem"].y;
        double x = lon;
        double y = -lat;
        double scale = 1.5;
        double hash = HashableString(city).get_hash();
        g->add_node(new HashableString(city));
        vec4 position = scale*vec4(x, y, 0, 0);
        g->move_node(hash, position);
        // Add edges to 3 nearest neighbors
        vector<double> neighbor_hashes;
        for(auto& [other_hash, node] : g->nodes) {
            if(other_hash == hash) continue;
            double distance = length(position - node.position);
            if(distance < .5) neighbor_hashes.push_back(other_hash);
        }
        gs->render_microblock();
    }



    // Plot cities as nodes and roads as edges, expanding west->east
    stage_macroblock(FileBlock("cities are the nodes"), 2);
    gs->manager.set("label_size", "0");
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), city);
    }
    gs->manager.set("label_size", "{microblock_fraction} {microblock_fraction} 1 - *");
    gs->render_microblock();
    gs->manager.transition(MICRO, "label_size", "1");
    gs->render_microblock();

    stage_macroblock(FileBlock("and the roads between them are the edges."), netherlands_edges_1.size());
    for(pair<string, string> edge : netherlands_edges_1) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("We’ll also give each city a letter, so it’s a little easier to reference them."), 3);
    gs->render_microblock();

    gs->manager.transition(MICRO, "label_size", "0");
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), netherlands_labels[city]);
    }
    gs->manager.transition(MICRO, "label_size", "1");
    gs->render_microblock();

    double groningen_hash = HashableString("Groningen").get_hash();
    vector<string> path = {"Rotterdam", "Utrecht", "Arnhem", "Zwolle", "Meppel", "Groningen"};
    stage_macroblock(FileBlock("If we want to get from R to G, this path seems immediately obvious."), path.size() + 3);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("Rotterdam").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, groningen_hash, 0xffff0000);
    gs->render_microblock();
    for(int i = 0; i < path.size() - 1; i++) {
        string city = path[i];
        string neighbor = path[i+1];
        gs->config->transition_edge_color(MICRO, HashableString(city).get_hash(), HashableString(neighbor).get_hash(), 0xffff0000);
        gs->config->transition_node_color(MICRO, HashableString(neighbor).get_hash(), 0xffff0000);
        gs->render_microblock();
    }
    gs->render_microblock();

    // Add more nodes
    stage_macroblock(FileBlock("And it is the shortest path, but what about now?"), netherlands_edges_2.size() + 2);
    gs->render_microblock();
    gs->config->transition_all_edge_colors(MICRO, 0xffffffff);
    for(pair<string, string> edge : netherlands_edges_2) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }
    gs->render_microblock();

    // Add even more nodes
    stage_macroblock(FileBlock("And now?"), netherlands_edges_3.size() + 2);
    gs->render_microblock();
    for(pair<string, string> edge : netherlands_edges_3) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }
    gs->render_microblock();

    // Slide 9

    stage_macroblock(FileBlock("Here's a possible method:"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Starting from the source, we explore all of its neighboring nodes to see if the target is there."), 7);
    gs->render_microblock();
    uint32_t edge_dark = 0xff606060;
    gs->config->transition_all_node_colors(MICRO, edge_dark);
    gs->config->transition_all_edge_colors(MICRO, edge_dark);

    unordered_set<double> border;
    unordered_set<double> next_border;
    unordered_set<double> visited;
    border.insert(HashableString("Rotterdam").get_hash());
    visited.insert(HashableString("Rotterdam").get_hash());

    int depth = 0;
    bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);

    stage_macroblock(FileBlock("If it isn't, we move on to those explored nodes and then check their neighbors for the target."), 9);
    bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);

    stage_macroblock(FileBlock("If we don't find it, we expand again."), 7);
    bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);

    stage_macroblock(FileBlock("And we keep expanding out until we reach the target node."), 4);
    while(border.size() > 0 && visited.find(groningen_hash) == visited.end()) {
        bfs(g, gs, border, next_border, visited, groningen_hash, depth);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all of the nodes at every level."), 34);
    depth = 0;
    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(HashableString("Rotterdam").get_hash());
    gs->config->transition_node_color(MICRO, HashableString("Rotterdam").get_hash(), 0xffff0000);

    while(border.size() > 0 && visited.find(groningen_hash) == visited.end()) {
        bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);
        gs->config->transition_all_edge_colors(MICRO, 0xffffffff);
    }

    depth = 0;
    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(HashableString("Rotterdam").get_hash());
    gs->config->transition_node_color(MICRO, HashableString("Rotterdam").get_hash(), 0xffff0000);

    stage_macroblock(FileBlock("If there was a path from R to G in two steps, we would’ve found it on the second iteration."), 16);
    for(int i = 0; i < 2; i++) {
        bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);
    }

    stage_macroblock(FileBlock("So five steps must be the shortest path."), 13);
    for(int i = 0; i < 2; i++) {
        bfs_rung(g, border, next_border, visited, groningen_hash, depth, gs);
    }

    gs->manager.set("physics_multiplier", "1");
    gs->manager.set("decay", "0");
    gs->manager.transition(MICRO, "decay", "0.5");
    gs->manager.transition(MICRO, "d", "3");
    stage_macroblock(FileBlock("But there’s a problem. This algorithm values all of the edges the same. Just one step."), 1);
    gs->render_microblock();

    for(auto& [city, coords] : netherlands_cities) {
        vec4 position(coords.y - netherlands_cities["Arnhem"].y, -(coords.x - netherlands_cities["Arnhem"].x), 0, 0);
        gs->transition_node_position(MICRO, HashableString(city).get_hash(), position);
    }
    stage_macroblock(FileBlock("Here’s that same graph, but now the edges have different, positive weights to represent things like distance or travel time. Now the shortest path is much harder to figure out. Dijkstra needed a better method."), 1);
    gs->render_microblock();
}

void slide12to15() {
    stage_macroblock(FileBlock("Dijkstra’s algorithm works like this. Each node has a cost, or how far it is from the source."), 1);
    stage_macroblock(FileBlock("At the start, the source has a cost of zero, and every other node has a cost of infinity. There’s two things to keep in mind:"), 1);
    stage_macroblock(FileBlock("One, Dijkstra’s algorithm will always explore nodes from lowest to highest cost."), 1);
    stage_macroblock(FileBlock("And two, it updates costs as it finds shorter and shorter paths. That’s why the other nodes start at infinity. It hasn’t explored any paths yet. Let’s see it in action."), 1);
    stage_macroblock(FileBlock("A is first, so the algorithm checks all of its edges."), 1);
    stage_macroblock(FileBlock("The current path from A to B costs 1, which is less than B’s current cost of infinity. It updates, or relaxes, B’s cost to 1. The algorithm orders nodes in a priority queue from lowest to highest cost."), 1);
    stage_macroblock(FileBlock("A also relaxes C and D to 3 and 2, respectively. That’s all of A’s neighbors, so the algorithm marks A as explored."), 1);
    stage_macroblock(FileBlock("B is next since it has the lowest cost out of all the unexplored nodes. B relaxes G’s cost to 6 and E’s cost to 3. Remember, the cost is the distance from the source, so for G and E, we have to also add the cost to get to B."), 1);
    stage_macroblock(FileBlock("But B can’t relax C since the path A B C costs 6, when C’s current cost is 3. The algorithm only keeps the shortest paths."), 1);
    stage_macroblock(FileBlock("This continues for the rest of the graph. If there’s any ties in the lowest cost, the algorithm can explore them in any order."), 1);
    stage_macroblock(FileBlock("And if the algorithm runs into a node it’s relaxed once, like going from E to G, it still compares the current path to the node’s current cost. In this case, the path A - B - E - G is shorter than A - B - G. So the algorithm updates G’s cost."), 1);
    stage_macroblock(FileBlock("When the target M has the next lowest cost, or it’s next up to be explored, the algorithm has built the shortest path up to M. It returns the shortest path length — 10. This is also M’s final cost."), 1);
    stage_macroblock(FileBlock("And if we mark what nodes are able to relax others, a sort of predecessor list, we can easily build directions for the shortest paths."), 1);
    stage_macroblock(FileBlock("Here’s a few more examples of Dijkstra’s running through different graphs. Just like breadth-first, there’s a search-frontier that slowly spreads through the nodes. But Dijkstra’s frontier jumps around in different directions based on the edge weights."), 1);
}

void slide15to16() {
    stage_macroblock(FileBlock("But how do we know that Dijkstra’s algorithm always returns the shortest path? After all, in some of these examples, it doesn’t explore all the nodes."), 1);
    stage_macroblock(FileBlock("When the target is the next node to be explored, that means it has the lowest cost of all the unexplored nodes. Say the target has a cost of ten."), 1);
    stage_macroblock(FileBlock("To get to that point, Dijkstra’s algorithm must have already explored all the paths that cost nine, eight, seven and so on. None of those paths reached the target, otherwise the algorithm would have relaxed its cost."), 1);
    stage_macroblock(FileBlock("So by always exploring from lowest to highest cost, Dijkstra’s algorithm will always find the shortest path."), 1);
}

void slide19() {
    stage_macroblock(FileBlock("But there’s a problem with it."), 1);
    stage_macroblock(FileBlock("Even if we implemented Dijkstra’s algorithm as efficiently as we could, the average query runtimes on the North American network would still be around 7 seconds per path."), 1);
    stage_macroblock(FileBlock("One small note. We get this average by picking lots of random points, and then averaging their runtime. But since random points on a graph this large tend to be far apart, this average favors longer paths. Dijkstra runs a lot quicker within one city. But it’s a useful benchmark we’ll use to mark our progress."), 1);
}

void slide21to24() {
    stage_macroblock(FileBlock("So, why does Dijkstra’s take so long?"), 1);
    stage_macroblock(FileBlock("Say we want to the path from Berlin to Rome. Here’s what that looks like."), 1);
    stage_macroblock(FileBlock("Dijkstra’s checks all the one hour journeys, and then the two hour ones and so on until it finally reaches Rome — a 16 hour journey."), 1);
    stage_macroblock(FileBlock("Its search frontier, or the boundary between explored and unexplored nodes, reaches cities that are way off like Amsterdam [7 hrs] and Copenhagen [8 hr 4 min by car] before it ever reaches Italy."), 1);
    stage_macroblock(FileBlock("If we know the rough direction of the target, is there a way we could punish illogical routes?"), 1);
    stage_macroblock(FileBlock("We’d like to prioritize cities that are closer to Rome. Using longitudes and latitudes, we can easily calculate the straight line distance of any city to Rome."), 1);
    stage_macroblock(FileBlock("If we instead order nodes by their cost plus this straight line distance, we can really direct the search towards the target."), 1);
    stage_macroblock(FileBlock("Let’s see these two algorithms side by side. Dijkstra’s search frontier spreads out in all directions. But this modified Dijsktra’s, also called A* [A-star] immediately heads towards Rome."), 1);
    stage_macroblock(FileBlock("It doesn’t need to search nearly as many nodes!"), 1);
    stage_macroblock(FileBlock("Each node’s height is now their straight line distance to the target. This distance is also called a heuristic."), 1);
    stage_macroblock(FileBlock("Now we can really see how the heuristic funnels the search directly towards Rome."), 1);
    stage_macroblock(FileBlock("Say there’s a super fast highway just north of Munich and now the shortest path to Rome goes north first."), 1);
    stage_macroblock(FileBlock("When the heuristic is zero, A* is the same as Dijkstra’s, so it still finds the shortest path."), 1);
    stage_macroblock(FileBlock("As we raise the heuristic, our search becomes more and more directed. At some point, it will be so aggressively headed toward Rome that it won't even explore Munich. In that case, A* won't find the shortest path."), 1);
    stage_macroblock(FileBlock("But as long as we pick the heuristic to be an underestimate, it should always return the shortest path."), 1);
}

void slide27() {
    stage_macroblock(FileBlock("What if we ran the search in both directions?"), 1);
    stage_macroblock(FileBlock("If two points are a distance R away, the search frontier covers all the nodes within a circle of radius R and area πR^2. But two searches only need to cover circles with a radius of R/2 before they meet. The total area it covers is half that of the single search!"), 1);
    stage_macroblock(FileBlock("Here’s a bidirectional Dijkstra and A* compared to their single direction versions."), 1);
    stage_macroblock(FileBlock("Both algorithms search way fewer nodes!"), 1);
}

void play() {
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

void render_video() {
    slide8();
}
