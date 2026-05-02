#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"

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

void set_camera_to_lat_long(shared_ptr<GraphScene> gs, vec2 lat_long) {
    vec4 focus = lat_long_to_xyz(lat_long);
    gs->manager.set({
        {"x",to_string(focus.x)},
        {"y",to_string(focus.y)},
        {"z",to_string(focus.z)},
    });
    quat rot = lat_long_to_quat(lat_long);
    gs->manager.set({
        {"q1", to_string(rot.u)},
        {"qi", to_string(rot.i) + " {t} .1 * sin .2 * +"},
        {"qj", to_string(rot.j) + " {t} .1 * cos .2 * +"},
        {"qk", to_string(rot.k)},
    });
}

void transition_camera_to_lat_long(TransitionType tt, shared_ptr<GraphScene> gs, vec2 lat_long) {
    vec4 focus = lat_long_to_xyz(lat_long);
    gs->manager.transition(tt, {
        {"x",to_string(focus.x)},
        {"y",to_string(focus.y)},
        {"z",to_string(focus.z)},
    });
    quat rot = lat_long_to_quat(lat_long);
    gs->manager.transition(MICRO, {
        {"q1", to_string(rot.u)},
        {"qi", to_string(rot.i) + " {t} .1 * sin .2 * +"},
        {"qj", to_string(rot.j) + " {t} .1 * cos .2 * +"},
        {"qk", to_string(rot.k)},
    });
}

unordered_map<string, vec2> us_cities = {
    {"New York", vec2(40.7128, -74.0060)},
    {"Los Angeles", vec2(34.0522, -118.2437)},
    {"Chicago", vec2(41.8781, -87.6298)},
    {"Houston", vec2(29.7604, -95.3698)},
    {"Phoenix", vec2(33.4484, -112.0740)},
    {"Philadelphia", vec2(39.9526, -75.1652)},
    {"San Antonio", vec2(29.4241, -98.4936)},
    {"San Diego", vec2(32.7157, -117.1611)},
    {"Dallas", vec2(32.7767, -96.7970)},
    {"Austin", vec2(30.2672, -97.7431)},
    {"Jacksonville", vec2(30.3322, -81.6557)},
    {"Columbus", vec2(39.9612, -82.9988)},
    {"Charlotte", vec2(35.2271, -80.8431)},
    {"San Francisco", vec2(37.7749, -122.4194)},
    {"Indianapolis", vec2(39.7684, -86.1581)},
    {"Seattle", vec2(47.6062, -122.3321)},
    {"Denver", vec2(39.7392, -104.9903)},
    {"Washington", vec2(38.9072, -77.0369)},
    {"Boston", vec2(42.3601, -71.0589)},
    {"El Paso", vec2(31.7619, -106.4850)},
    {"Nashville", vec2(36.1627, -86.7816)},
    {"Detroit", vec2(42.3314, -83.0458)},
    {"Oklahoma City", vec2(35.4676, -97.5164)},
    {"Portland", vec2(45.5152, -122.6784)},
    {"Las Vegas", vec2(36.1699, -115.1398)},
    {"Memphis", vec2(35.1495, -90.0490)},
    {"Louisville", vec2(38.2527, -85.7585)},
    {"Baltimore", vec2(39.2904, -76.6122)},
    {"Milwaukee", vec2(43.0389, -87.9065)},
    {"Albuquerque", vec2(35.0844, -106.6504)},
    {"Tucson", vec2(32.2226, -110.9747)},
    {"Fresno", vec2(36.7378, -119.7871)},
    {"Atlanta", vec2(33.7490, -84.3880)},
    {"Kansas City", vec2(39.0997, -94.5786)},
    {"Miami", vec2(25.7617, -80.1918)},
    {"Raleigh", vec2(35.7796, -78.6382)},
    {"Omaha", vec2(41.2565, -95.9345)},
    {"Virginia Beach", vec2(36.8529, -75.9780)},
    {"Minneapolis", vec2(44.9778, -93.2650)},
    {"Tulsa", vec2(36.1540, -95.9928)},
    {"Tampa", vec2(27.9506, -82.4572)},
    {"New Orleans", vec2(29.9511, -90.0715)},
    {"Wichita", vec2(37.6872, -97.3301)},
    {"Cleveland", vec2(41.4993, -81.6944)},
    {"Bakersfield", vec2(35.3733, -119.0187)},
    {"Corpus Christi", vec2(27.8006, -97.3964)},
    {"Lexington", vec2(38.0406, -84.5037)},
    {"Cincinnati", vec2(39.1031, -84.5120)},
    {"St. Louis", vec2(38.6270, -90.1994)},
    {"Pittsburgh", vec2(40.4406, -79.9959)},
    {"Lincoln", vec2(40.8136, -96.7026)},
    {"Orlando", vec2(28.5383, -81.3792)},
    {"Toledo", vec2(41.6528, -83.5379)},
    {"Fort Wayne", vec2(41.0793, -85.1394)},
    {"Laredo", vec2(27.5306, -99.4803)},
    {"Madison", vec2(43.0731, -89.4012)},
    {"Buffalo", vec2(42.8864, -78.8784)},
    {"Lubbock", vec2(33.5779, -101.8552)},
    {"Reno", vec2(39.5296, -119.8138)},
    {"Boise", vec2(43.6150, -116.2023)},
    {"Richmond", vec2(37.5407, -77.4360)},
    {"Baton Rouge", vec2(30.4515, -91.1871)},
    {"Spokane", vec2(47.6588, -117.4260)},
    {"Des Moines", vec2(41.5868, -93.6250)},
    {"Greensboro", vec2(36.0726, -79.7920)},
    {"Salt Lake City", vec2(40.7608, -111.8910)},
    {"Cheyenne", vec2(41.1400, -104.8202)},
    {"Billings", vec2(45.7833, -108.5007)},
    {"Sioux Falls", vec2(43.5446, -96.7311)},
    {"Fargo", vec2(46.8772, -96.7898)},
    {"Bismarck", vec2(46.8083, -100.7837)},
    {"Rapid City", vec2(44.0805, -103.2310)},
    {"Amarillo", vec2(35.2220, -101.8313)},
    {"Midland", vec2(31.9974, -102.0779)},
    {"Shreveport", vec2(32.5252, -93.7502)},
    {"Little Rock", vec2(34.7465, -92.2896)},
    {"Mobile", vec2(30.6954, -88.0399)},
    {"Savannah", vec2(32.0809, -81.0912)},
    {"Charleston", vec2(32.7765, -79.9311)},
    {"Knoxville", vec2(35.9606, -83.9207)},
    {"Birmingham", vec2(33.5186, -86.8104)},
    {"Chattanooga", vec2(35.0456, -85.3097)},
    {"Roanoke", vec2(37.2709, -79.9414)},
    {"Scranton", vec2(41.4089, -75.6624)},
    {"Burlington", vec2(44.4759, -73.2121)},
    {"Manchester", vec2(42.9956, -71.4548)},
    {"Portland", vec2(43.6591, -70.2568)}, // Maine
    {"Wilmington", vec2(34.2257, -77.9447)}, // North Carolina
    {"Erie", vec2(42.1292, -80.0851)},
    {"Green Bay", vec2(44.5133, -88.0133)},
    {"Duluth", vec2(46.7867, -92.1005)},
    {"Springfield", vec2(37.2089, -93.2923)}, // Missouri
    {"Eugene", vec2(44.0521, -123.0868)},
    {"Medford", vec2(42.3265, -122.8756)},
    {"Santa Fe", vec2(35.6870, -105.9378)},
    {"Yakima", vec2(46.6021, -120.5059)},
    {"Casper", vec2(42.8501, -106.3252)}, // Wyoming
    {"Flagstaff", vec2(35.1983, -111.6513)},
    {"Grand Junction", vec2(39.0639, -108.5506)},
    {"St. George", vec2(37.0965, -113.5684)},
    {"Tallahassee", vec2(30.4383, -84.2807)},
};

vector<pair<string, string>> us_edges;
unordered_map<string, int> edges_per_city;

float get_angle_abc(vec2 a, vec2 b, vec2 c) {
    vec2 ba = a - b;
    vec2 bc = c - b;

    double dot_product = ba.x * bc.x + ba.y * bc.y;
    double magnitude_ba = sqrt(ba.x * ba.x + ba.y * ba.y);
    double magnitude_bc = sqrt(bc.x * bc.x + bc.y * bc.y);

    double cosine = dot_product / (magnitude_ba * magnitude_bc);

    return acos(cosine);
}

void populate_us_edges() {
    // Greedily connect edges, smallest first, such that:
    // 1) No city has more than 4 edges
    // 2) no two edges, incident on node n are separated by less than pi/4 angle at n
    // 3) no edge is longer than distance 6 in lat-long space
    // Continue adding until no more edges can be added
    int max_edges_per_city = 4;
    int max_distance = 5;
    float min_angle = M_PI/4;

    // Start by making a list of all possible edges and sorting them by length
    vector<tuple<string, string, double>> possible_edges;
    for(auto& [city1, coords1] : us_cities) {
        for(auto& [city2, coords2] : us_cities) {
            if(city1 >= city2) continue;
            double distance = length(coords1 - coords2);
            if(distance < max_distance) {
                possible_edges.push_back({city1, city2, distance});
            }
        }
    }

    // Sort edges by length
    sort(possible_edges.begin(), possible_edges.end(), [](auto& a, auto& b) {
        return get<2>(a) < get<2>(b);
    });

    // Iterate through edges and add them if they satisfy the constraints
    for(auto& [city1, city2, distance] : possible_edges) {
        if(edges_per_city[city1] >= max_edges_per_city || edges_per_city[city2] >= max_edges_per_city) {
            continue;
        }

        bool valid = true;
        for(pair<string, string> edge : us_edges) {
            string existing_city1 = edge.first;
            string existing_city2 = edge.second;
            if(existing_city1 == city1 || existing_city2 == city1) {
                string other_city = existing_city1 == city1 ? existing_city2 : existing_city1;
                if(get_angle_abc(us_cities[other_city], us_cities[city1], us_cities[city2]) < min_angle) {
                    valid = false;
                    break;
                }
            }
            if(existing_city1 == city2 || existing_city2 == city2) {
                string other_city = existing_city1 == city2 ? existing_city2 : existing_city1;
                if(get_angle_abc(us_cities[other_city], us_cities[city2], us_cities[city1]) < min_angle) {
                    valid = false;
                    break;
                }
            }
        }
        if(!valid) continue;

        us_edges.push_back({city1, city2});
        edges_per_city[city1]++;
        edges_per_city[city2]++;
    }

    // Manually add a few extras
    us_edges.push_back({"Boise", "Portland"});
    us_edges.push_back({"Boise", "Salt Lake City"});
    us_edges.push_back({"Boise", "Billings"});
    us_edges.push_back({"Salt Lake City", "Reno"});
    us_edges.push_back({"Denver", "Wichita"});
    us_edges.push_back({"Denver", "Lincoln"});
    us_edges.push_back({"Rapid City", "Sioux Falls"});
    us_edges.push_back({"Spokane", "Billings"});
    us_edges.push_back({"Bismarck", "Billings"});
    us_edges.push_back({"Albuquerque", "Flagstaff"});
    us_edges.push_back({"Phoenix", "Los Angeles"});
    us_edges.push_back({"Tucson", "San Diego"});
    us_edges.push_back({"Birmingham", "Memphis"});
}

void find_closest_pair() {
    double min_distance = std::numeric_limits<double>::infinity();
    pair<string, string> closest_pair;
    for(auto& [city1, coords1] : us_cities) {
        for(auto& [city2, coords2] : us_cities) {
            if(city1 >= city2) continue;
            double distance = length(coords1 - coords2);
            if(distance < min_distance) {
                min_distance = distance;
                closest_pair = {city1, city2};
            }
        }
    }
    cout << "Closest pair of cities: " << closest_pair.first << " and " << closest_pair.second << " with distance " << min_distance << endl;
}

void slide3() {
    find_closest_pair();
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    vec2 center_coords = us_cities["Wichita"];
    set_camera_to_lat_long(gs, center_coords);
    gs->manager.set({
        {"globe_opacity", "1"},
        {"d", ".7"},
    });
    gs->manager.transition(MACRO, "globe_opacity", "0.4");
    stage_macroblock(FileBlock("If we simplified the US to a grid of 100 cities"), 100);
    // Add nodes in east to west
    unordered_set<string> remaining_cities;
    for(auto& [city, coords] : us_cities) {
        remaining_cities.insert(city);
    }
    while(remaining_cities.size() > 0) {
        string city = "";
        for(string candidate : remaining_cities) {
            if(city == "" || us_cities[candidate].y > us_cities[city].y) {
                city = candidate;
            }
        }
        double hash = HashableString(city).get_hash();
        vec4 position = lat_long_to_xyz(us_cities[city]);
        g->add_node(new HashableString(city));
        g->move_node(hash, position);
        gs->render_microblock();
        remaining_cities.erase(city);
    }

    stage_macroblock(FileBlock("it would look like this."), 1);
    populate_us_edges();
    for(pair<string, string> edge : us_edges) {
        string city1 = edge.first;
        string city2 = edge.second;
        double hash1 = HashableString(city1).get_hash();
        double hash2 = HashableString(city2).get_hash();
        g->add_edge(hash1, hash2);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("The number of possible paths from one corner to the other is in the septillions."), 40);
    while (remaining_microblocks_in_macroblock) {
        // color a random non-self-intersecting path from Miami to Seattle red
        vector<string> path;
        while(true) {
            path = {"Miami"};
            while(path.back() != "Seattle") {
                string current_city = path.back();
                vector<string> neighbors;
                for(pair<string, string> edge : us_edges) {
                    if(edge.first == current_city && find(path.begin(), path.end(), edge.second) == path.end()) {
                        neighbors.push_back(edge.second);
                    } else if(edge.second == current_city && find(path.begin(), path.end(), edge.first) == path.end()) {
                        neighbors.push_back(edge.first);
                    }
                }
                if(neighbors.size() == 0) {
                    break;
                }
                string next_city = neighbors[rand() % neighbors.size()];
                path.push_back(next_city);
            }
            if(path.back() == "Seattle") {
                break;
            }
        }
        for(int i = 0; i < path.size() - 1; i++) {
            string city1 = path[i];
            string city2 = path[i+1];
            gs->config->set_edge_color(HashableString(city1).get_hash(), HashableString(city2).get_hash(), 0xffff0000);
            gs->config->set_node_color(HashableString(city1).get_hash(), 0xffff0000);
        }
        gs->render_microblock();
        // Undo
        for(int i = 0; i < path.size() - 1; i++) {
            string city1 = path[i];
            string city2 = path[i+1];
            gs->config->set_edge_color(HashableString(city1).get_hash(), HashableString(city2).get_hash(), 0xffffffff);
            gs->config->set_node_color(HashableString(city1).get_hash(), 0xffffffff);
        }
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
    //{"Maastricht", vec2(50.8514, 5.6900)},
    {"Arnhem", vec2(51.9851, 5.8987)},
    {"Zwolle", vec2(52.5168, 6.0830)},
    {"Emmen", vec2(52.7795, 6.9061)},
    {"Groningen", vec2(53.2194, 6.5665)},
    {"Leeuwarden", vec2(53.2012, 5.7999)},
    {"'s-Hertogenbosch", vec2(51.6978, 5.3037)},
    {"Tillburg", vec2(51.5555, 5.0913)},
    {"Meppel", vec2(52.7917, 6.1789)},
};

vector<pair<string, string>> netherlands_edges_1 = {
    {"Rotterdam", "Utrecht"},
    {"Rotterdam", "Breda"},
    {"Breda", "Tillburg"},
    {"Tillburg", "Eindhoven"},
    //{"Eindhoven", "Maastricht"},
    {"Eindhoven", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Utrecht"},
    {"Utrecht", "Arnhem"},
    {"Arnhem", "Zwolle"},
    {"Zwolle", "Emmen"},
    {"Zwolle", "Meppel"},
    {"Meppel", "Groningen"},
    {"Meppel", "Leeuwarden"},
    {"Amsterdam", "The Hague"},
    {"Amsterdam", "Utrecht"},
};

vector<pair<string, string>> netherlands_edges_2 = {
    {"Amsterdam", "Zwolle"},
    {"The Hague", "Rotterdam"},
    {"Tillburg", "'s-Hertogenbosch"},
};

vector<pair<string, string>> netherlands_edges_3 = {
    {"Groningen", "Leeuwarden"},
    {"The Hague", "Utrecht"},
    {"Rotterdam", "'s-Hertogenbosch"},
    {"'s-Hertogenbosch", "Arnhem"},
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

void bfs_simul(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, int depth) {
    double node = *border.begin();
    // Find node with least neighbors in the border
    for(double candidate : border) {
        if(g->get_neighbors(candidate).size() < g->get_neighbors(node).size()) {
            node = candidate;
        }
    }
    gs->render_microblock();
    unordered_set<double> neighbors = g->get_neighbors(node);
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
    }
    gs->render_microblock();
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        if (next_border.find(neighbor) == next_border.end() && border.find(neighbor) == border.end()) {
            next_border.insert(neighbor);
            gs->config->transition_node_color(MICRO, neighbor, colors_by_depth[depth]);
        }
    }
    gs->render_microblock();
    border.erase(node);
    visited.insert(node);
    gs->config->transition_node_color(MICRO, node, 0xffffffff);
    gs->render_microblock();

    if(border.size() == 0) {
        border = next_border;
        next_border.clear();
    }
}

void bfs(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, unordered_set<double>& border, unordered_set<double>& next_border, unordered_set<double>& visited, int depth, bool simul) {
    if(simul) {
        bfs_simul(g, gs, border, next_border, visited, depth);
        return;
    }
    double node = *border.begin();
    gs->manager.transition(MICRO, {
        {"x", to_string(g->nodes.find(node)->second.position.x)},
        {"y", to_string(g->nodes.find(node)->second.position.y)},
        {"z", to_string(g->nodes.find(node)->second.position.z)},
    });
    unordered_set<double> neighbors = g->get_neighbors(node);
    for(double neighbor : neighbors) {
        if(visited.find(neighbor) != visited.end()) continue;
        gs->config->transition_edge_color(MICRO, node, neighbor, colors_by_depth[depth]);
        gs->render_microblock();
        if (next_border.find(neighbor) == next_border.end() && border.find(neighbor) == border.end()) {
            next_border.insert(neighbor);
            gs->config->transition_node_color(MICRO, neighbor, colors_by_depth[depth]);
            gs->render_microblock();
        }
    }
    border.erase(node);
    visited.insert(node);
    gs->config->transition_node_color(MICRO, node, 0xff00ff00);
    gs->render_microblock();

    if(border.size() == 0) {
        border = next_border;
        next_border.clear();
    }
}

void run_dijkstra(shared_ptr<Graph> g, shared_ptr<GraphScene> gs, double start, double goal, int up_to_step) {
    std::unordered_set<double> visited;
    std::unordered_map<double, double> costs;

    std::unordered_set<double> open_set;

    std::unordered_map<double, double> came_from;

    open_set.insert(start);
    gs->config->transition_node_color(MICRO, start, 0xffff0000);
    if(--up_to_step == 0) return;

    costs[start] = 0;
    gs->config->transition_node_label(MICRO, start, "0");
    if(--up_to_step == 0) return;

    for(auto& [hash, node] : g->nodes) {
        if(hash == start) continue;
        costs[hash] = std::numeric_limits<double>::infinity();
        gs->config->transition_node_label(MICRO, hash, "\\infty");
    }
    if(--up_to_step == 0) return;

    while(open_set.size() > 0) {
        // Find node in open set with lowest cost
        double current = -1;
        double current_cost = std::numeric_limits<double>::infinity();
        for(double hash : open_set) {
            if(costs[hash] < current_cost) {
                current_cost = costs[hash];
                current = hash;
            }
        }

        if(current == goal) {
            cout << "Reached goal!" << endl;
            break;
        }

        open_set.erase(current);
        gs->config->transition_node_color(MICRO, current, 0xff0000ff);
        if(--up_to_step == 0) return;

        unordered_set<double> neighbors = g->get_neighbors(current);

        for(double neighbor : neighbors) {
            if(visited.find(neighbor) != visited.end()) {
                continue;
            }
            double weight = length(g->nodes.find(current)->second.position - g->nodes.find(neighbor)->second.position);
            double tentative_cost = costs[current] + weight;
            gs->config->transition_edge_color(MICRO, current, neighbor, 0xffff0000);
            if(--up_to_step == 0) return;

            if(tentative_cost < costs[neighbor]) {
                came_from[neighbor] = current;
                costs[neighbor] = tentative_cost;
                gs->config->transition_node_label(MICRO, neighbor, to_string_with_precision(tentative_cost, 1));
                if(open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                    gs->config->transition_node_color(MICRO, neighbor, 0xffff0000);
                }
            }
            if(--up_to_step == 0) return;

            gs->config->fade_edge_color(MICRO, neighbor, current, 0xffffffff);
            if(--up_to_step == 0) return;
        }
        visited.insert(current);
        gs->config->transition_node_color(MICRO, current, 0xff00ff00);
        if(--up_to_step == 0) return;
    }
}

void slide8() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    gs->manager.transition(MACRO, "globe_opacity", "1");
    set_camera_to_lat_long(gs, vec2(52.5, 5.5));
    gs->manager.set({
        {"physics_multiplier","0"},
        {"d", ".07"},
    });

    // Load Netherlands map
    stage_macroblock(FileBlock("Here's a simplified graph of the Netherlands."), netherlands_cities.size());

    // Plot cities as nodes and roads as edges, expanding east->west
    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        double hash = HashableString(city).get_hash();
        g->add_node(new HashableString(city));
        g->move_node(hash, position);
        gs->render_microblock();
    }

    // Plot cities as nodes and roads as edges, expanding west->east
    stage_macroblock(FileBlock("The nodes represent the cities,"), 2);
    gs->manager.transition(MICRO, "globe_opacity", ".2");
    gs->manager.set("label_size", "0");
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), "\\text{" + city + "}");
    }
    gs->manager.set("label_size", "{microblock_fraction} {microblock_fraction} 1 - *");
    gs->render_microblock();
    gs->manager.transition(MICRO, "label_size", "1");
    gs->render_microblock();

    stage_macroblock(FileBlock("and the edges connecting them are the rail lines."), netherlands_edges_1.size());
    for(pair<string, string> edge : netherlands_edges_1) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }

    double rotterdam_hash = HashableString("Rotterdam").get_hash();
    double groningen_hash = HashableString("Groningen").get_hash();

    vector<string> path = {"Rotterdam", "Utrecht", "Arnhem", "Zwolle", "Meppel", "Groningen"};
    stage_macroblock(FileBlock("To get from Rotterdam to Groningen,"), 3);
    transition_camera_to_lat_long(MACRO, gs, netherlands_cities["Rotterdam"]);
    for(auto& [city, coords] : netherlands_cities) {
        gs->config->transition_node_label(MICRO, HashableString(city).get_hash(), "");
    }
    gs->render_microblock();
    // Label Rotterdam
    gs->config->transition_node_label(MICRO, rotterdam_hash, "\\text{Rotterdam}");
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->render_microblock();
    // Label Groningen
    gs->config->transition_node_label(MICRO, groningen_hash, "\\text{Groningen}");
    gs->config->transition_node_color(MICRO, groningen_hash, 0xffff0001);
    gs->render_microblock();

    stage_macroblock(FileBlock("this path seems obvious."), path.size());
    transition_camera_to_lat_long(MACRO, gs, netherlands_cities["Groningen"]);
    for(int i = 0; i < path.size() - 1; i++) {
        string city = path[i];
        string neighbor = path[i+1];
        gs->config->transition_edge_color(MICRO, HashableString(city).get_hash(), HashableString(neighbor).get_hash(), 0xffff0000);
        gs->render_microblock();
        gs->config->transition_node_color(MICRO, HashableString(neighbor).get_hash(), 0xffff0000);
    }
    gs->render_microblock();

    // Add more nodes
    stage_macroblock(FileBlock("And it is the shortest path, but what about now?"), netherlands_edges_2.size() + 6);
    transition_camera_to_lat_long(MACRO, gs, vec2(52.5, 5.5));
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);

    for(pair<string, string> edge : netherlands_edges_2) {
        string city = edge.first;
        string neighbor = edge.second;
        double hash = HashableString(city).get_hash();
        double neighbor_hash = HashableString(neighbor).get_hash();
        g->add_edge(hash, neighbor_hash);
        gs->render_microblock();
    }

    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    // Add even more nodes
    stage_macroblock(FileBlock("And now?"), netherlands_edges_3.size() + 4);
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
    gs->render_microblock();
    gs->render_microblock();

    // Slide 9

    stage_macroblock(FileBlock("Here's a possible method:"), 1);
    uint32_t edge_dark = 0xff808080;
    gs->config->fade_all_node_colors(MICRO, edge_dark);
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->render_microblock();

    bool simul = true;
    stage_macroblock(FileBlock("Starting from the source, we explore all of its neighboring nodes to see if the target is there."), simul?5:10);
    gs->manager.transition(MICRO, {
        {"x", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.x)},
        {"y", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.y)},
        {"z", to_string(g->nodes.find(HashableString("Rotterdam").get_hash())->second.position.z)},
    });
    gs->config->transition_node_color(MICRO, rotterdam_hash, 0xffff0000);
    gs->render_microblock();

    unordered_set<double> border;
    unordered_set<double> next_border;
    unordered_set<double> visited;
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    bfs(g, gs, border, next_border, visited, 0, simul);

    stage_macroblock(FileBlock("If it isn't, we move on to those explored nodes and then check their neighbors for the target."), simul?16:17);
    transition_camera_to_lat_long(MACRO, gs, netherlands_cities["Utrecht"]);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);

    stage_macroblock(FileBlock("If we don't find it, we expand again."), simul?12:7);
    transition_camera_to_lat_long(MACRO, gs, netherlands_cities["Arnhem"]);
    bfs(g, gs, border, next_border, visited, 2, simul);
    bfs(g, gs, border, next_border, visited, 2, simul);
    bfs(g, gs, border, next_border, visited, 2, simul);

    stage_macroblock(FileBlock("And we keep expanding out until we reach the target node."), simul?16:12);
    transition_camera_to_lat_long(MACRO, gs, netherlands_cities["Meppel"]);
    bfs(g, gs, border, next_border, visited, 3, simul);
    bfs(g, gs, border, next_border, visited, 3, simul);

    bfs(g, gs, border, next_border, visited, 4, simul);
    bfs(g, gs, border, next_border, visited, 4, simul);

    stage_macroblock(FileBlock("This algorithm is known as breadth first search."), 1);
    gs->render_microblock();
    gs->config->fade_all_edge_colors(MICRO, edge_dark);
    gs->config->fade_all_node_colors(MICRO, edge_dark);

    stage_macroblock(FileBlock("It will always find the shortest path because it checks all of the nodes at every level."), simul?4:9);
    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    bfs(g, gs, border, next_border, visited, 0, simul);

    visited.clear();
    border.clear();
    next_border.clear();
    border.insert(rotterdam_hash);
    visited.insert(rotterdam_hash);

    stage_macroblock(FileBlock("If there was a path from Rotterdam to Groningen in two steps,"), simul?4:9);
    bfs(g, gs, border, next_border, visited, 0, simul);
    stage_macroblock(FileBlock("we would’ve found it on the second iteration, so six steps must be the shortest path."), simul?22:17);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);
    bfs(g, gs, border, next_border, visited, 1, simul);

    gs->manager.set("physics_multiplier", "1");
    gs->manager.set("repel", ".01");
    gs->manager.set("decay", "0.2");
    gs->manager.transition(MICRO, "decay", "0.5");
    return;

    stage_macroblock(FileBlock("But there’s a problem. This algorithm values all of the edges the same. Just one step."), 1);
    gs->render_microblock();
    gs->manager.set("physics_multiplier", "0");

    for(auto& [city, coords] : netherlands_cities) {
        vec4 position = lat_long_to_xyz(coords);
        gs->transition_node_position(MICRO, HashableString(city).get_hash(), position);
    }
    stage_macroblock(FileBlock("So let's adjust that map, and add weights to represent distance."), 1);
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 2);
    gs->manager.transition(MICRO, "edge_weights_size", "1");
    gs->render_microblock();
    for(auto& [city, coords] : netherlands_cities) {
        double hash = HashableString(city).get_hash();
        gs->config->transition_node_label(MICRO, hash, "");
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("Now the shortest path is much harder to figure out."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra realized that it wasn’t the number of edges between a node and the source that mattered, it was their distance to the source."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So instead of exploring nodes by their level, he could explore all the nodes close to the source, and then the next closest nodes, and so on — ordered by their distance."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra’s algorithm works like this. Each node has a cost, or how far it is from the source."), 6);
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "2");
    gs->render_microblock();
    gs->manager.transition(MICRO, "points_radius_multiplier", "1");
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("At the start, the source has a cost of zero,"), 2);
    int step = 0;
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("and every other node has a cost of infinity."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("There’s two things to keep in mind:"), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("One, Dijkstra’s algorithm will always explore nodes from lowest to highest cost."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("And two, it updates costs as it finds shorter and shorter paths."), 1);
    gs->render_microblock();
    stage_macroblock(FileBlock("That’s why the other nodes start at infinity. It hasn’t explored any paths yet. Let’s see it in action."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Rotterdam is first, so the algorithm checks all of its edges."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("The current path from A to B costs 1, which is less than B’s current cost of infinity. It updates, or relaxes, B’s cost to 1."), 3);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("A also relaxes C to .7, D to .3, and E to 1.3."), 12);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("That’s all of A’s neighbors, so the algorithm marks A as explored."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("B is next since it has the lowest cost out of all the unexplored nodes."), 1);
    run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
    gs->render_microblock();

    stage_macroblock(FileBlock("B relaxes G’s cost to 6 and E’s cost to 3."), 6);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("Remember, the cost is the distance from the source, so for G and E, we have to also add the cost to get to B."), 2);
    while(remaining_microblocks_in_macroblock) {
        run_dijkstra(g, gs, rotterdam_hash, groningen_hash, ++step);
        gs->render_microblock();
    }

    return;

    stage_macroblock(FileBlock("But B can’t relax C since the path A B C costs 6, when C’s current cost is 3. The algorithm only keeps the shortest paths."), 1);
    stage_macroblock(FileBlock("This continues for the rest of the graph. If there’s any ties in the lowest cost, the algorithm can explore them in any order."), 1);
    stage_macroblock(FileBlock("And if the algorithm runs into a node it’s relaxed once, like going from E to G, it still compares the current path to the node’s current cost. In this case, the path A - B - E - G is shorter than A - B - G. So the algorithm updates G’s cost."), 1);
    stage_macroblock(FileBlock("When the target M has the next lowest cost, or it’s next up to be explored, the algorithm has built the shortest path up to M. It returns the shortest path length — 10. This is also M’s final cost."), 1);
    stage_macroblock(FileBlock("And if we mark what nodes are able to relax others, a sort of predecessor list, we can easily build directions for the shortest paths."), 1);
    stage_macroblock(FileBlock("Here’s a few more examples of Dijkstra’s running through different graphs. Just like breadth-first, there’s a search-frontier that slowly spreads through the nodes. But Dijkstra’s frontier jumps around in different directions based on the edge weights."), 1);
}

// File format:
// File starts with line NODES
// Then nodes are listed: id (integer), latitude (float), longitude (float)
// Then line EDGES
// Then edges are listed: node1 (integer), node2 (integer)
// string filename = "io_in/graph.txt";
void load_graph_from_file(shared_ptr<Graph> g, shared_ptr<GraphScene> gs) {
    ifstream file("io_in/graph.txt");
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
            g->add_node(new HashableString(to_string(id)));
            g->move_node(hash, position);
            node_count++;
        } else if(section == EDGES) {
            stringstream ss(line);
            int id1, id2;
            ss >> id1 >> id2;
            double hash1 = HashableString(to_string(id1)).get_hash();
            double hash2 = HashableString(to_string(id2)).get_hash();
            g->add_edge(hash1, hash2);
            edge_count++;
        }
    }
    cout << "Loaded graph with " << node_count << " nodes and " << edge_count << " edges." << endl;
}

void slide20() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    set_camera_to_lat_long(gs, vec2(52.5, 5.5));
    gs->manager.set({
        {"globe_opacity", "0.2"},
        {"d", ".07"},
        {"points_opacity", "0"},
    });
    // Fade globe to opacity 1
    gs->manager.transition(MICRO, "globe_opacity", "1");
    stage_macroblock(FileBlock("But Dijkstra alone isn’t good enough for Google Maps."), 2);
    gs->render_microblock();
    gs->manager.transition(MICRO, "d", "1");
    gs->render_microblock();

    // Transition to NYC
    stage_macroblock(SilenceBlock(1), 1);
    transition_camera_to_lat_long(MICRO, gs, vec2(40.7128, -74.0060));
    gs->render_microblock();

    gs->manager.transition(MICRO, "d", ".02");
    stage_macroblock(FileBlock("Let’s say I want to get from Newark Airport in New Jersey over to the Central Park Zoo."), 2);
    if(rendering_on()) load_graph_from_file(g, gs);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Dijkstra’s algorithm checks all the ten minute journeys,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("and then all the twenty minute journeys,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("and so on until it reaches all the forty minute journeys, including the Zoo."), 1);
    gs->render_microblock();
}

void slide23() {
    // A star
    stage_macroblock(FileBlock("We'd like to prioritize nodes that are closer to the Central Park Zoo."), 1);
    stage_macroblock(FileBlock("Using longitudes and latitudes, we can easily calculate the straight line distance between any node and the zoo."), 1);
    stage_macroblock(FileBlock("We'll order nodes by their cost plus this straight line distance."), 1);
    stage_macroblock(FileBlock("Nodes in the opposite direction won't be explored early on."), 1);

    stage_macroblock(FileBlock("Let’s see these two algorithms side by side."), 1);
    stage_macroblock(FileBlock("Dijkstra’s search frontier spreads out in all directions."), 1);
    stage_macroblock(FileBlock("But this modified Dijkstra's, also called A* [A-star] immediately heads towards Manhattan."), 1);
    stage_macroblock(FileBlock("It only checks around 7,000 nodes — that’s almost a 10x improvement!"), 1);
}

unordered_map<string, vec2> graph_nodes = {
    {"a", vec2(-1.5, 0)},
    {"b", vec2(-.5, .5)},
    {"c", vec2(.5, .5)},
    {"d", vec2(1.5, .5)},
    {"e", vec2(-.5, -.5)},
    {"f", vec2(.5, -.5)},
    {"h", vec2(1.5, -.5)},
};

unordered_map<string, int> graph_edges_with_weights = {
    {"ab", 2},
    {"bc", 3},
    {"cd", 2},
    {"ae", 10},
    {"ef", 1},
    {"fh", 7},
    {"ec", 5},
    {"cf", 1},
    {"fd", 8},
};

void slide42() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "4"},
        {"edge_weights_size", "1"},
    });

    stage_macroblock(FileBlock("Let's zoom in on a smaller part of the graph to see how this really works."), 1);
    for(auto& [node, coords] : graph_nodes) {
        double hash = HashableString(node).get_hash();
        vec4 position = vec4(coords.x, coords.y, 0, 1);
        g->add_node(new HashableString(node));
        g->move_node(hash, position);
    }
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        g->add_edge(hash1, hash2);
        gs->config->transition_edge_label(MICRO, hash1, hash2, to_string(weight));
    }
    gs->render_microblock();

    gs->manager.transition(MICRO, "edge_weights_size", "0");
    stage_macroblock(FileBlock("We can ignore the edge weights for the first step."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("These two nodes split the graph roughly in half,"), 4);
    // Highlight c and e
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_edge_color(MICRO, HashableString("c").get_hash(), HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();

    unordered_map<string, int> node_ranks = {
        {"a", 2},
        {"b", 1},
        {"c", 6},
        {"d", 4},
        {"e", 7},
        {"f", 5},
        {"h", 3},
    };

    stage_macroblock(FileBlock("so they will have the highest rank."), 2);
    gs->render_microblock();
    gs->config->transition_node_label(MICRO, HashableString("c").get_hash(), to_string(node_ranks["c"]));
    gs->config->transition_node_label(MICRO, HashableString("e").get_hash(), to_string(node_ranks["e"]));
    gs->render_microblock();

    // Fade all nodes and edges gray except for d, f and h
    stage_macroblock(FileBlock("We can split one side a little further, so that node gets the next highest rank."), 4);
    for(auto& [node, coords] : graph_nodes) {
        if(node == "d" || node == "f" || node == "h") continue;
        gs->config->fade_node_color(MICRO, HashableString(node).get_hash(), 0xff505050);
    }
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        if((node1 == "d" || node1 == "f" || node1 == "h") && (node2 == "d" || node2 == "f" || node2 == "h")) continue;
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        gs->config->fade_edge_color(MICRO, hash1, hash2, 0xff505050);
    }
    gs->render_microblock();

    // Highlight f
    gs->config->transition_node_color(MICRO, HashableString("f").get_hash(), 0xff00ff00);
    gs->config->transition_node_label(MICRO, HashableString("f").get_hash(), to_string(node_ranks["f"]));
    gs->render_microblock();

    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("We then rank the rest of the nodes in any order with the remaining numbers."), 6);
    gs->render_microblock();
    string remaining_nodes = "abdh";
    double height = 0;
    for(char node : remaining_nodes) {
        string node_str(1, node);
        vec2 position = graph_nodes[node_str];
        gs->config->transition_node_label(MICRO, HashableString(node_str).get_hash(), to_string(node_ranks[node_str]));
        gs->config->splash_node(HashableString(node_str).get_hash());
        gs->render_microblock();
        height -= .5;
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("It’s easier to see this hierarchy in 3D."), 1);
    gs->manager.transition(MICRO, "qi", "-.9");
    gs->render_microblock();

    // Duplicate the whole graph
    for(auto& [node, coords] : graph_nodes) {
        double hash = HashableString(node).get_hash();
        vec4 position = vec4(coords.x, coords.y, 0, 1);
        g->add_node(new HashableString(node + "'"));
        g->move_node(HashableString(node + "'").get_hash(), position);
    }
    // Add edges
    for(auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        double hash1 = HashableString(node1).get_hash();
        double hash2 = HashableString(node2).get_hash();
        g->add_edge(HashableString(node1 + "'").get_hash(), HashableString(node2 + "'").get_hash());
    }

    stage_macroblock(FileBlock("The original graph is in white, and then the nodes are pulled down, or contracted, by their rank."), 1);
    for(auto& [node, coords] : graph_nodes) {
        string node_str = node;
        vec2 position = graph_nodes[node_str];
        gs->transition_node_position(MICRO, HashableString(node_str).get_hash(), vec4(position.x, position.y, -.1, 0));
    }
    gs->render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("1 is the furthest down, 2 is next up and so on."), SilenceBlock(3)), 10);
    for (int rank = 1; rank <= 7; rank++) {
        // Find node with this rank
        string node;
        for(auto& [n, r] : node_ranks) {
            if(r == rank) {
                node = n;
                break;
            }
        }
        vec2 position = graph_nodes[node];
        gs->transition_node_position(MICRO, HashableString(node).get_hash(), vec4(position.x, position.y, node_ranks[node]/7.-1.2, 0));
        if(node == "b" || node == "a" || node == "h") {
            gs->render_microblock();
        }
        gs->render_microblock();
    }

    stage_macroblock(FileBlock("Eventually, our algorithm will use a bidirectional search,"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("but it will only ever search from low ranks to higher ranks."), 1);
    // Transition all edge colors from side of low rank to side of high rank
    for (auto& [edge, weight] : graph_edges_with_weights) {
        string node1 = edge.substr(0, 1);
        string node2 = edge.substr(1, 1);
        string lower_rank_node = node_ranks[node1] < node_ranks[node2] ? node1 : node2;
        string higher_rank_node = node_ranks[node1] < node_ranks[node2] ? node2 : node1;
        double hash1 = HashableString(lower_rank_node).get_hash();
        double hash2 = HashableString(higher_rank_node).get_hash();
        gs->config->transition_edge_color(MICRO, hash1, hash2, 0xffff0000);
    }
    gs->render_microblock();

    stage_macroblock(FileBlock("But if we search like that, this graph has an obvious problem."), 1);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Let’s add the edge weights back."), 1);
    gs->manager.transition(MICRO, "edge_weights_size", "1");
    gs->render_microblock();

    stage_macroblock(FileBlock("What’s the shortest path from A to C using this hierarchy search?"), 5);
    // Highlight A and C
    gs->render_microblock();
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xffff0000);
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Well, starting from A, we search up and reach E with a cost of 10."), 1);
    // Add edge A to E
    gs->config->transition_edge_color(MICRO, HashableString("a").get_hash(), HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("From C, we also search up and reach E with a cost of 5."), 1);
    // Add edge C to E
    gs->config->transition_edge_color(MICRO, HashableString("c").get_hash(), HashableString("e").get_hash(), 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("Shortest path is A, E, C with a cost of 15. Done."), 3);
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("a").get_hash(), HashableString("e").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("e").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("e").get_hash(), HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();

    stage_macroblock(FileBlock("Except that’s not the shortest path."), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Using the path A, B, C only costs 5."), 3);
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("a").get_hash(), HashableString("b").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("b").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();

    stage_macroblock(FileBlock("What went wrong?"), 1);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("B has edges to two higher ranked nodes."), 1);
    gs->config->transition_node_color(MICRO, HashableString("b").get_hash(), 0xffff0000);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("a").get_hash(), 0xffff0000);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("c").get_hash(), 0xffff0000);
    gs->render_microblock();

    stage_macroblock(FileBlock("But since it’s the lowest of the three, there’s no way to consider that path."), 2);
    // Pop B
    gs->config->splash_node(HashableString("b").get_hash());
    gs->render_microblock();
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();

    stage_macroblock(FileBlock("Remember, our search never follows a path that’s up, down, up."), 3);
    // Highlight path A, B, C
    gs->config->transition_node_color(MICRO, HashableString("a").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("a").get_hash(), HashableString("b").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("b").get_hash(), 0xff00ff00);
    gs->config->transition_edge_color(MICRO, HashableString("b").get_hash(), HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();
    gs->config->transition_node_color(MICRO, HashableString("c").get_hash(), 0xff00ff00);
    gs->render_microblock();

    stage_macroblock(FileBlock("To fix it, we’ll add a shortcut between A and C."), 2);
    gs->config->fade_all_node_colors(MICRO, 0xffffffff);
    gs->config->fade_all_edge_colors(MICRO, 0xffffffff);
    gs->render_microblock();
    g->add_edge(HashableString("a").get_hash(), HashableString("c").get_hash());
    gs->render_microblock();

    stage_macroblock(FileBlock("Its weight is just the cost of the path A, B, C and it has a midpoint marker to B."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("The bidirectional search now checks both the upper path [AEC] and the shortcut to find the best path."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But there's three more problems if we want this to work on ANY graph."), 1);
    gs->render_microblock();
}

void render_video() {
    //slide3();
    //slide8();
    slide20();
    //slide42();
}
