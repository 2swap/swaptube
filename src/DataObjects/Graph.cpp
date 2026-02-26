#include "Graph.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <map>
#include <deque>
#include <limits.h>
#include <queue>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>

using json = nlohmann::json;

extern "C" void compute_repulsion_cuda(glm::vec4* h_positions, glm::vec4* h_velocities, const int* h_adjacency_matrix, const int* h_mirrors, const int* h_mirror2s, int num_nodes, int max_degree, float attract, float repel, float mirror_force, const float decay, const float dimension, const int iterations);

glm::vec4 random_unit_cube_vector(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    return glm::vec4(
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f
    );
}

Node::Node(GenericBoard* t, double hash, glm::vec4 position, glm::vec4 velocity) :
    data(t), hash(hash), velocity(velocity), position(position) {}

float Node::weight() const { return sigmoid(age*.2f + 0.01f); }
double Node::radius() const { return size * (((3*age - 1) * exp(-.5*age)) + 1); }
double Node::splash_opacity() const { return 1-square(age/12.); }
double Node::splash_radius() const { return size * age * .4; }

Graph::Graph() : root_node_hash(0), dist(0.0f, 1.0f), rng(0) {}

Graph::~Graph() {
    clear();
}

int Graph::size() const {
    return nodes.size();
}

void Graph::clear_queue() {
    traverse_deque.clear();
}

void Graph::clear() {
    traverse_deque.clear();
    while (nodes.size()>0) {
        auto i = nodes.begin();
        delete i->second.data;
        nodes.erase(i->first);
    }
}

void Graph::add_to_stack(GenericBoard* t){
    double hash = t->get_hash();
    add_node_without_edges(t);
    traverse_deque.push_front(hash);
}

double Graph::add_node(GenericBoard* t){
    double x = add_node_without_edges(t);
    add_missing_edges();
    return x;
}
double Graph::add_node_without_edges(GenericBoard* t){
    double hash = t->get_hash();
    if (node_exists(hash)) {
        delete t;
        return hash;
    }
    Node new_node(t, hash, random_unit_cube_vector(rng, dist), random_unit_cube_vector(rng, dist));
    if (size() == 0) {
        root_node_hash = hash;
    }
    nodes.emplace(hash, new_node);
    return hash;
}

int Graph::expand(int n) {
    if(n == 0) return 0;
    int added = 0;
    while (!traverse_deque.empty()) {
        double id = traverse_deque.front();
        traverse_deque.pop_front();

        std::unordered_set<GenericBoard*> child_nodes = nodes.at(id).data->get_children();
        bool done = false;
        for (const auto& child : child_nodes) {
            double child_hash = child->get_hash();
            if (done || node_exists(child_hash)) delete child;
            else {
                add_node_without_edges(child);
                if(n>0) traverse_deque.push_front(id); // No need to save progress if expanding whole graph
                traverse_deque.push_back(child_hash); // push_back: bfs // push_front: dfs
                added++;
                std::cout << "." << std::flush;
                if(added >= n && n > 0) done = true;
            }
        }
        if(done) {
            add_missing_edges();
            mark_updated();
            return added;
        }
    }
    add_missing_edges();
    if(added > 0) mark_updated();
    return added;
}

void Graph::add_node_with_position(GenericBoard* t, double x, double y, double z) {
    double hash = add_node(t);
    move_node(hash, x, y, z);
}

void Graph::move_node(double hash, double x, double y, double z, double w) {
    auto it = nodes.find(hash);
    if (it == nodes.end()) return;
    Node& node = it->second;
    node.position = glm::vec4(x,y,z,w);
    mark_updated();
}

void Graph::add_directed_edge(double from, double to, double opacity) {
    if (!node_exists(from) || !node_exists(to)) return;
    nodes.at(from).neighbors.insert(Edge(from, to, opacity));
    mark_updated();
}

void Graph::remove_edge(double from, double to) {
    if (!node_exists(from) || !node_exists(to)) return;
    
    Node& from_node = nodes.at(from);
    Edge edge_to_remove(from, to);
    
    from_node.neighbors.erase(edge_to_remove);
    mark_updated();
}

const Edge* Graph::get_edge(double from, double to) {
    if (!node_exists(from)) return nullptr;
    Node& from_node = nodes.at(from);
    for (const Edge& edge : from_node.neighbors) {
        if (edge.to == to) {
            return &edge;
        }
    }
    return nullptr;
}

bool Graph::does_edge_exist(double from, double to){
    if (!node_exists(from)) return false;

    EdgeSet& children = nodes.at(from).neighbors;

    for (const Edge& edge : children) {
        if (edge.to == to) {
            return true;
        }
    }

    return false;
}

void Graph::add_missing_edges() {
    for (auto& pair : nodes) {
        Node& parent = pair.second;
        if(parent.expected_children_hashes.size() == 0)
            parent.expected_children_hashes = parent.data->get_children_hashes();

        for (double child_hash : parent.expected_children_hashes) {
            if(!node_exists(child_hash)) continue;
            Node& child = nodes.find(child_hash)->second;
            if(child.age == 0 && parent.age != 0 && !does_edge_exist(parent.hash, child_hash)){
                child.position = parent.position + random_unit_cube_vector(rng, dist);
                child.velocity = parent.velocity;
            }
            add_directed_edge(parent.hash, child_hash);
        }
    }
}

bool Graph::node_exists(double id) const {
    return nodes.find(id) != nodes.end();
}

void Graph::remove_node(double id) {
    if (!node_exists(id)) return;
    Node& node = nodes.at(id);
    for (const auto& neighbor_edge : node.neighbors) {
        double neighbor_id = neighbor_edge.to;
        nodes.at(neighbor_id).neighbors.erase(Edge(neighbor_id, id));
    }
    delete node.data;
    nodes.erase(id);
    mark_updated();
}

int Graph::measure_distance(double start, double end) {
    return shortest_path(start, end).first.size();
}

std::pair<std::list<double>, std::list<Edge*>> Graph::shortest_path(double start, double end) {
    std::unordered_map<double, double> distances;
    std::unordered_map<double, double> predecessors;
    std::unordered_set<double> visited;
    auto compare = [&](double lhs, double rhs) { return distances[lhs] > distances[rhs]; };
    std::priority_queue<double, std::vector<double>, decltype(compare)> priority_queue(compare);

    for (const auto& node_pair : nodes) {
        distances[node_pair.first] = std::numeric_limits<double>::max();
    }
    distances[start] = 0;
    priority_queue.push(start);

    while (!priority_queue.empty()) {
        double current = priority_queue.top();
        priority_queue.pop();

        if (visited.find(current) != visited.end()) continue;
        visited.insert(current);

        if (current == end) break;

        for (const auto& edge : nodes.at(current).neighbors) {
            double neighbor = edge.to;
            double new_distance = distances[current] + 1;
            if (new_distance < distances[neighbor]) {
                distances[neighbor] = new_distance;
                predecessors[neighbor] = current;
                priority_queue.push(neighbor);
            }
        }
    }

    std::list<double> path;
    std::list<Edge*> edges;
    for (double at = end; at != start; at = predecessors[at]) {
        if (predecessors.find(at) == predecessors.end()) {
            return {std::list<double>(), std::list<Edge*>()};
        }
        path.push_front(at);
        for (auto& edge : nodes.at(predecessors[at]).neighbors) {
            if (edge.to == at) {
                edges.push_front(const_cast<Edge*>(&edge));
                break;
            }
        }
    }
    path.push_front(start);

    return {path, edges};
}

void Graph::collapse_two_nodes(double hash_keep, double hash_remove) {
    if (!node_exists(hash_keep) || !node_exists(hash_remove)) return;
    if (hash_keep == hash_remove) return;

    Node& node_keep = nodes.at(hash_keep);
    Node& node_remove = nodes.at(hash_remove);

    for (const auto& edge : node_remove.neighbors) {
        if (edge.to != hash_keep) {
            add_directed_edge(hash_keep, edge.to, edge.opacity);
            add_directed_edge(edge.to, hash_keep, edge.opacity);
        }
    }

    remove_node(hash_remove);
    mark_updated();
}

void Graph::delete_isolated() {
    std::unordered_set<double> non_isolated;

    for (const auto& node_pair : nodes) {
        const Node& node = node_pair.second;
        for (const auto& edge : node.neighbors) {
            non_isolated.insert(edge.to);
            non_isolated.insert(edge.from);
        }
    }

    for (auto it = nodes.begin(); it != nodes.end(); ) {
        if (non_isolated.find(it->first) == non_isolated.end() && it->first != root_node_hash) {
            delete it->second.data;
            it = nodes.erase(it);
        } else {
            ++it;
        }
    }
    mark_updated();
}

void Graph::delete_orphans() {
    bool orphan_found;
    do {
        orphan_found = false;
        std::unordered_set<double> non_orphans;

        for (const auto& node_pair : nodes) {
            const Node& node = node_pair.second;
            for (const auto& edge : node.neighbors) {
                non_orphans.insert(edge.to);
            }
        }

        for (auto it = nodes.begin(); it != nodes.end(); ) {
            if (non_orphans.find(it->first) == non_orphans.end() && it->first != root_node_hash) {
                delete it->second.data;
                it = nodes.erase(it);
                orphan_found = true;
            } else {
                ++it;
            }
        }
    } while (orphan_found);
    mark_updated();
}

std::vector<int> Graph::make_adjacency_matrix(const std::vector<Node*>& node_vector, int &max_degree) {
    int n = node_vector.size();
    max_degree = 0;
    for (int i = 0; i < n; ++i) {
        int degree = node_vector[i]->neighbors.size();
        if (degree > max_degree) max_degree = degree;
    }

    std::vector<int> adjacency_matrix(n * max_degree, -1);

    std::unordered_map<double, int> node_index_map;
    for (int i = 0; i < n; ++i) {
        node_index_map[node_vector[i]->hash] = i;
    }

    for (int i = 0; i < n; ++i) {
        int col = 0;
        for (const Edge& edge : node_vector[i]->neighbors) {
            if (col >= max_degree) break;
            auto it = node_index_map.find(edge.to);
            if (it != node_index_map.end()) {
                adjacency_matrix[i * max_degree + col] = it->second;
                col++;
            }
        }
    }
    return adjacency_matrix;
}

void Graph::iterate_physics(const int iterations, const float repel, const float attract, const float decay, const float centering_strength, const double dimension, const float mirror_force, const bool flip_by_symmetry) {
    std::vector<Node*> node_vector;

    for (auto& node_pair : nodes) node_vector.push_back(&node_pair.second);
    for (int i = 0; i < node_vector.size(); ++i) { node_vector[i]->age++; }

    int s = node_vector.size();
    std::vector<glm::vec4> positions(s);
    std::vector<glm::vec4> velocities(s);

    glm::vec4 com = center_of_mass() * centering_strength;
    for (int i = 0; i < s; ++i) {
         positions[i] = node_vector[i]->position - com;
        velocities[i] = node_vector[i]->velocity;
        node_vector[i]->index = i;
    }
    int max_degree = 0;
    std::vector<int> adjacency_matrix = make_adjacency_matrix(node_vector, max_degree);
    std::vector<int> mirrors(s, -1);
    std::vector<int> mirror2s(s, -1);

    for (int i = 0; i < s; ++i) {
        const auto& node = node_vector[i];

        {
            double rev_hash = node->data->get_reverse_hash();
            auto it_mirror = nodes.find(rev_hash);
            if (it_mirror != nodes.end()) {
                mirrors[i] = it_mirror->second.index;
            }
        }
        {
            double rev_hash_2 = node->data->get_reverse_hash_2();
            auto it_mirror_2 = nodes.find(rev_hash_2);
            if (it_mirror_2 != nodes.end()) {
                mirror2s[i] = it_mirror_2->second.index;
            }
        }
    }

    compute_repulsion_cuda(positions.data(), velocities.data(), adjacency_matrix.data(), mirrors.data(), mirror2s.data(), s, max_degree, attract, repel, mirror_force, decay, dimension, iterations);

    for (int i = 0; i < s; ++i) {
        int flip = 1;
        if(flip_by_symmetry) flip = signum(node_vector[i]->data->which_side() * node_vector[i]->position.x);
        node_vector[i]->position = positions[i];
        node_vector[i]->position.x *= flip;
        node_vector[i]->velocity = velocities[i];
    }

    mark_updated();
}

glm::vec4 Graph::center_of_mass() const {
    glm::vec4 sum_position(0.0f);
    float mass = 0.1;

    for (const auto& node_pair : nodes) {
        const Node& node = node_pair.second;
        float sig = node.weight();
        glm::vec4 addy = sig*node.position;
        sum_position += addy;
        mass += sig;
    }

    glm::vec4 ret = sum_position / mass;
    return ret;
}

float Graph::af_dist() const {
    float sum_distance_sq = 0.0;
    float ct = 0.1;

    glm::vec4 com = center_of_mass();

    for (const auto& node_pair : nodes) {
        const Node& node = node_pair.second;
        float sig = node.weight();
        glm::vec4 pos_com = node.position - com;
        sum_distance_sq += sig * glm::dot(pos_com, pos_com);
        ct += sig;
    }

    float ans = 6 + 4.8*pow(sum_distance_sq / ct, .5);
    return ans;
}

void Graph::render_json(std::string json_out_filename) {
    std::ofstream myfile;
    myfile.open(json_out_filename);

    json json_data;

    json nodes_to_use;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        const Node& node = it->second;
        json node_info;
        node_info["x"] = node.position.x;
        node_info["y"] = node.position.y;
        node_info["z"] = node.position.z;
        node_info["rep"] = node.data->representation;
        node_info["data"] = node.data->get_data();

        json neighbors;
        for (const auto& neighbor : node.neighbors) {
            std::string neighbor_representation = nodes.at(neighbor.to).data->representation;
            if(neighbor_representation.size() < node.data->representation.size()) continue;
            std::ostringstream oss;
            oss << std::setprecision(17) << neighbor.to;
            neighbors.push_back(oss.str());
        }
        node_info["neighbors"] = neighbors;

        std::ostringstream oss;
        oss << std::setprecision(17) << it->first;
        nodes_to_use[oss.str()] = node_info;
    }

    json_data["nodes_to_use"] = nodes_to_use;
    json_data["nodes_to_use"].dump(4, ' ', false, json::error_handler_t::ignore);

    std::ostringstream oss;
    oss << std::setprecision(17) << root_node_hash;
    json_data["root_node_hash"] = oss.str();
    json_data["board_w"  ] = 7;
    json_data["board_h"  ] = 6;
    json_data["game_name"] = "c4";

    myfile.seekp(0, ios::beg);
    myfile << "var dataset = ";
    myfile << json_data.dump();

    myfile.close();

    std::cout << "Rendered json!" << std::endl;
}

std::unordered_set<double> Graph::get_neighborhood(double hash, int dist) {
    std::unordered_set<double> neighborhood;
    if (!node_exists(hash) || dist < 0) return neighborhood;
    neighborhood.insert(hash);

    std::unordered_set<double> current_level_nodes;
    current_level_nodes.insert(hash);

    for (int d = 0; d < dist; ++d) {
        std::unordered_set<double> next_level_nodes;
        for (double node_hash : current_level_nodes) {
            const Node& node = nodes.at(node_hash);
            for (const Edge& edge : node.neighbors) {
                if (neighborhood.find(edge.to) == neighborhood.end()) {
                    neighborhood.insert(edge.to);
                    next_level_nodes.insert(edge.to);
                }
            }
        }
        if (next_level_nodes.empty()) break;
        current_level_nodes = std::move(next_level_nodes);
    }

    return neighborhood;
}

void Graph::make_bidirectional() {
    std::vector<std::pair<double, double>> edges_to_add;
    for (const auto& pair : nodes) {
        double from = pair.first;
        const EdgeSet& neighbors = pair.second.neighbors;
        for (const Edge& edge : neighbors) {
            double to = edge.to;
            if (!does_edge_exist(to, from)) {
                edges_to_add.emplace_back(to, from);
            }
        }
    }
    for (const auto& edge : edges_to_add) {
        add_directed_edge(edge.first, edge.second);
    }
}
