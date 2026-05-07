#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"
#include "GraphAlgs_common.cpp"

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

void render_video() {
    find_closest_pair();
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);
    vec2 center_coords = us_cities["Wichita"];
    set_camera_to_lat_long(gs, center_coords, true, MACRO);
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
