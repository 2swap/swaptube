uint32_t opaque_white = 0x20ffffff;

quat lat_long_to_quat(vec2 lat_long) {
    float lon = -lat_long.x;
    float lat = lat_long.y;

    // Convert latitude and longitude from degrees to radians
    float lat_rad = lat * (M_PI / 180.0f);
    float lon_rad = lon * (M_PI / 180.0f);

    // Calculate the quaternion components
    float cy = cos(lon_rad * 0.5f);
    float sy = sin(lon_rad * 0.5f);
    float cp = cos(lat_rad * 0.5f);
    float sp = sin(lat_rad * 0.5f);

    return quat(
        cy * cp,
        sy * cp,
        cy * sp,
        sy * sp 
    );
}

vec4 lat_long_to_xyz(vec2 lat_long) {
    float lat = lat_long.x;
    float lon = lat_long.y;

    // Convert latitude and longitude from degrees to radians
    float lat_rad = lat * (M_PI / 180.0f);
    float lon_rad = lon * (M_PI / 180.0f);

    // Calculate the Cartesian coordinates
    lon_rad -= M_PI/2;
    float x = cos(lat_rad) * cos(lon_rad);
    float y = sin(lat_rad);
    float z = cos(lat_rad) * sin(lon_rad);

    return vec4(x, y, z, 0);
}

void quat_mult_string(
    const string& ru, const string& ri, const string& rj, const string& rk,
    const string& su, const string& si, const string& sj, const string& sk,
    string& out_u, string& out_i, string& out_j, string& out_k) {

    out_u = ru + " " + su + " * " + ri + " " + si + " * - " + rj + " " + sj + " * - " + rk + " " + sk + " * -";
    out_i = ru + " " + si + " * " + ri + " " + su + " * + " + rj + " " + sk + " * + " + rk + " " + sj + " * -";
    out_j = ru + " " + sj + " * " + ri + " " + sk + " * - " + rj + " " + su + " * + " + rk + " " + si + " * +";
    out_k = ru + " " + sk + " * " + ri + " " + sj + " * + " + rj + " " + si + " * - " + rk + " " + su + " * +";
}

void set_camera_to_lat_long(shared_ptr<GraphScene> gs, vec2 lat_long, bool set, TransitionType tt) {
    vec4 f = lat_long_to_xyz(lat_long);
    StateSet focus({
        {"x",to_string(f.x)},
        {"y",to_string(f.y)},
        {"z",to_string(f.z)},
    });
    if(set) {
        gs->manager.set(focus);
    } else {
        gs->manager.transition(tt, focus);
    }
    gs->manager.set({
        {"theta", ".5"},
        {"phi", "{t} .11 * sin .2 *"},
    });
    quat rot = lat_long_to_quat(lat_long);
    string ru = "<theta> 2 / cos";
    string ri = "<theta> 2 / sin";
    string rj = "0";
    string rk = "0";
    string su = "<phi> 2 / cos";
    string si = "0";
    string sj = "0";
    string sk = "<phi> 2 / sin";
    string au, ai, aj, ak;
    quat_mult_string(ru, ri, rj, rk, su, si, sj, sk, au, ai, aj, ak);
    string cu = to_string(rot.u);
    string ci = to_string(rot.i);
    string cj = to_string(rot.j);
    string ck = to_string(rot.k);
    string q1, qi, qj, qk;
    quat_mult_string(au, ai, aj, ak, cu, ci, cj, ck, q1, qi, qj, qk);
    StateSet rot_state({
        {"q1", q1},
        {"qi", qi + " {t} .09 * sin -.1 * +"},
        {"qj", qj + " {t} .07 * sin -.1 * +"},
        {"qk", qk},
    });
    if(set) {
        gs->manager.set(rot_state);
    } else {
        gs->manager.transition(tt, rot_state);
    }
}

void trace_path(shared_ptr<GraphScene> gs, vector<string> path, int color) {
    for(int i = 0; i < path.size() - 1; i++) {
        string node1 = path[i];
        string node2 = path[i + 1];
        gs->config->transition_node_color(MICRO, HashableString(node1).get_hash(), color);
        gs->config->transition_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), color);
        gs->render_microblock();
    }
    gs->config->transition_node_color(MICRO, HashableString(path.back()).get_hash(), color);
    gs->render_microblock();
}

// lat long map

// File format:
// File starts with line NODES
// Then nodes are listed: id (integer), latitude (float), longitude (float)
// Then line EDGES
// Then edges are listed: node1 (integer), node2 (integer)
// Ignore any nodes or edges that are outside the given radius from the center point
void load_graph_from_file(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, vec2 center, float radius, unordered_map<double, double>& edge_weights) {
    ifstream file("io_in/graph_speeds.txt");
    string line;
    enum Section { NONE, NODES, EDGES };
    Section section = NONE;
    int node_count = 0;
    int edge_count = 0;
    while(getline(file, line)) {
        if(line == "NODES") {
            section = NODES;
            continue;
        } else if(line == "EDGES") {
            section = EDGES;
            continue;
        }
        if(section == NODES) {
            stringstream ss(line);
            int id;
            float lat, longi;
            ss >> id >> lat >> longi;
            double hash = HashableString(to_string(id)).get_hash();
            vec4 position = lat_long_to_xyz(vec2(lat, longi));
            if (length(vec2(lat, longi) - center) > radius) continue;
            g->add_node(new HashableString(to_string(id)));
            g->move_node(hash, position);
            gs->config->set_node_radius(hash, 0);
            gs->config->set_node_color(hash, 0x00000000);
            node_count++;
        } else if(section == EDGES) {
            stringstream ss(line);
            int id1, id2;
            double weight;
            ss >> id1 >> id2 >> weight;
            double hash1 = HashableString(to_string(id1)).get_hash();
            double hash2 = HashableString(to_string(id2)).get_hash();
            edge_weights[hash1 * 5 + hash2] = weight;
            edge_weights[hash2 * 5 + hash1] = weight;
            if(g->nodes.find(hash1) == g->nodes.end() || g->nodes.find(hash2) == g->nodes.end()) {
                continue;
            }
            g->add_edge(hash1, hash2);
            gs->config->add_edge_if_missing(hash1, hash2);
            gs->config->set_edge_color(hash1, hash2, opaque_white);
            edge_count++;
        }
    }
    cout << "Loaded graph with " << node_count << " nodes and " << edge_count << " edges." << endl;
}

void reset_graph(shared_ptr<Graph> g, shared_ptr<GraphScene> gs) {
    // Set all edges to opaque white except for the longest one, which we set to bright green
    // distance is measured by position of nodes, not by edge weight
    double longest_edge = -1;
    double longest_edge_node1 = -1;
    double longest_edge_node2 = -1;
    for(auto& [hash, node] : g->nodes) {
        unordered_set<double> neighbors = g->get_neighbors(hash);
        for(double neighbor : neighbors) {
            double weight = length(g->nodes.find(hash)->second.position - g->nodes.find(neighbor)->second.position);
            if(weight > longest_edge) {
                longest_edge = weight;
                longest_edge_node1 = hash;
                longest_edge_node2 = neighbor;
            }
        }
    }
    for(auto& [hash, node] : g->nodes) {
        unordered_set<double> neighbors = g->get_neighbors(hash);
        for(double neighbor : neighbors) {
            if((hash == longest_edge_node1 && neighbor == longest_edge_node2) || (hash == longest_edge_node2 && neighbor == longest_edge_node1)) {
                gs->config->set_edge_color(hash, neighbor, 0xff00ff00);
            } else {
                gs->config->set_edge_color(hash, neighbor, opaque_white);
            }
        }
    }
}

// Run dijkstra's algorithm up until some node within max_dist of the goal is added to the visited set.
// Color all searched edges blue.
bool run_large_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double start, double goal, double max_dist, float heuristic_mult, unordered_map<double, double> edge_weights, unordered_set<double> highlighted_nodes = {}) {
    cout << "Running large dijkstra with max_dist " << max_dist << " and heuristic_mult " << heuristic_mult << endl;
    bool reset = highlighted_nodes.size() == 2;
    if(reset) {
        reset_graph(g, gs);
    }
    std::unordered_set<double> visited;
    std::unordered_map<double, double> gScore;
    std::unordered_map<double, double> fScore;

    std::unordered_set<double> open_set;

    std::unordered_map<double, double> came_from;

    bool transition_not_fade = highlighted_nodes.size() == 0;
    open_set.insert(start);
    gScore[start] = 0;
    fScore[start] = length(g->nodes.find(start)->second.position - g->nodes.find(goal)->second.position) * heuristic_mult;
    vec4 start_pos = g->nodes.find(start)->second.position;

    for(auto& [hash, node] : g->nodes) {
        if(hash == start) continue;
        gScore[hash] = std::numeric_limits<double>::infinity();
        fScore[hash] = std::numeric_limits<double>::infinity();
    }

    while(open_set.size() > 0) {
        // Find node in open set with lowest cost
        double current = -1;
        double current_cost = std::numeric_limits<double>::infinity();
        for(double hash : open_set) {
            if(fScore[hash] < current_cost) {
                current_cost = fScore[hash];
                current = hash;
            }
        }

        vec4 current_pos = g->nodes.find(current)->second.position;

        if (current == goal) {
            cout << "Reached goal!" << endl;
            gs->config->transition_node_color(MICRO, current, 0xff00ff01);
            // Color the path from current to start green
            double path_node = current;
            while(path_node != start) {
                double parent = came_from[path_node];
                gs->config->set_edge_color(path_node, parent, 0xff00ffff);
                path_node = parent;
            }
            return true;
        }

        open_set.erase(current);

        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            if(visited.find(neighbor) != visited.end()) {
                continue;
            }
            double tentative_gScore = gScore[current] + edge_weights[current * 5 + neighbor];
            double tentative_fScore = tentative_gScore + length(g->nodes.find(neighbor)->second.position - g->nodes.find(goal)->second.position) * heuristic_mult;

            if(tentative_gScore < max_dist) {
                int color = 0x10ff8080;
                if(highlighted_nodes.find(neighbor) != highlighted_nodes.end()) {
                    color = 0x2000ff00;
                }
                if(reset) {
                    gs->config->set_edge_color(current, neighbor, color);
                } else if(transition_not_fade) {
                    gs->config->transition_edge_color(MICRO, current, neighbor, color);
                } else {
                    gs->config->fade_edge_color(MICRO, current, neighbor, color);
                }
            }

            if(tentative_gScore < gScore[neighbor]) {
                came_from[neighbor] = current;
                gScore[neighbor] = tentative_gScore;
                fScore[neighbor] = tentative_fScore;

                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                }
            }
        }
        visited.insert(current);
        if (gScore[current] > max_dist) {
            return false;
        }
    }
    return false;
}

double get_nearest_node_in_graph(shared_ptr<Graph> g, vec2 lat_long) {
    double nearest_node = -1;
    double nearest_distance = std::numeric_limits<double>::infinity();
    vec3 node_xyz = lat_long_to_xyz(lat_long);
    for(auto& [hash, node] : g->nodes) {
        vec3 position = node.position;
        double distance = length(position - node_xyz);
        if(distance < nearest_distance) {
            nearest_distance = distance;
            nearest_node = hash;
        }
    }
    return nearest_node;
}
