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
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Core/Smoketest.h"
#include "../IO/SFX.h"

using json = nlohmann::json;

extern "C" void compute_repulsion_cuda(vec4* h_positions, vec4* h_velocities, const int* h_adjacency_matrix, const int* h_mirrors, const int* h_mirror2s, int num_nodes, int max_degree, float attract, float repel, float mirror_force, const float decay, const float dimension, const int iterations);

vector<int> tones = {0,4,7};
int tone_incr = 0;
void node_pop(double subdiv, bool added_not_deleted) {
    int tone_number = added_not_deleted?tones[tone_incr%tones.size()]:-6;
    double tone = pow(2,tone_number/12.);
    tone_incr++;
    sfx_boink(get_global_state("t") + subdiv, tone * 440, 1/80., 1);
}

vec4 random_unit_cube_vector(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    return {
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f,
        2.0f * dist(rng) - 1.0f
    };
}

Node::Node(GenericBoard* t, double hash, vec4 position, vec4 velocity) :
    data(t), hash(hash), velocity(velocity), position(position) {}

Graph::Graph() : dist(0.0f, 1.0f), rng(0) {}

Graph::~Graph() {
    clear();
}

int Graph::size() const {
    return nodes.size();
}

void Graph::tick(const StateReturn& state) {
    int nodes_to_add = state["desired_nodes"] - size();

    // SFX
    if(last_node_count > -1){
        int diff = size() - last_node_count;
        for(int i = 0; i < abs(diff); i++) {
            node_pop(static_cast<double>(i)/abs(diff), diff>0);
        }
    }

    last_node_count = size();
    int amount_to_iterate = state["physics_multiplier"];
    if(!rendering_on()) amount_to_iterate = min(amount_to_iterate, 1); // No need to spread graphs out in smoketest
    iterate_physics(
        amount_to_iterate,
        state["repel"],
        state["attract"],
        state["decay"],
        state["dimensions"],
        state["mirror_force"]
    );
    if(has_been_updated_since_last_scene_query()) {
        //graph_to_3d();
        //clear_surfaces();
        //update_surfaces();
    }
}

void Graph::clear() {
    while (nodes.size()>0) {
        auto i = nodes.begin();
        delete i->second.data;
        nodes.erase(i->first);
    }
}

double Graph::add_node(GenericBoard* t){
    double hash = t->get_hash();
    if (node_exists(hash)) {
        delete t;
        return hash;
    }
    Node new_node(t, hash, random_unit_cube_vector(rng, dist), random_unit_cube_vector(rng, dist));
    nodes.emplace(hash, new_node);
    return hash;
}

void Graph::add_node_with_neighbors(GenericBoard* t, std::vector<double> neighbor_hashes) {
    double hash = add_node(t);
    if (neighbor_hashes.empty()) return;
    vec4 avg_position(0.0f);
    vec4 avg_velocity(0.0f);
    for (double neighbor_hash : neighbor_hashes) {
        if (!node_exists(neighbor_hash)) continue;
        const Node& neighbor = nodes.at(neighbor_hash);
        avg_position += neighbor.position + 0.01f * random_unit_cube_vector(rng, dist);
        avg_velocity += neighbor.velocity;
        add_edge(hash, neighbor_hash);
    }
    int count = neighbor_hashes.size();
    avg_position /= count;
    avg_velocity /= count;
    nodes.at(hash).position = avg_position;
    nodes.at(hash).velocity = avg_velocity;
}

void Graph::move_node(double hash, vec4 pos) {
    auto it = nodes.find(hash);
    if (it == nodes.end()) return;
    Node& node = it->second;
    node.position = pos;
    mark_updated();
}

void Graph::add_edge(double from, double to, double opacity) {
    if (!node_exists(from) || !node_exists(to)) return;
    nodes.at(from).neighbors.insert(Edge(from, to));
    nodes.at(to  ).neighbors.insert(Edge(to, from));
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

void Graph::iterate_physics(const int iterations, const float repel, const float attract, const float decay, const double dimension, const float mirror_force) {
    std::vector<Node*> node_vector;
    std::unordered_map<double, int> node_indices;

    for (auto& node_pair : nodes) node_vector.push_back(&node_pair.second);

    int s = node_vector.size();
    std::vector<vec4> positions(s);
    std::vector<vec4> velocities(s);

    for (int i = 0; i < s; ++i) {
         positions[i] = node_vector[i]->position;
        velocities[i] = node_vector[i]->velocity;
        node_indices[node_vector[i]->hash] = i;
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
                mirrors[i] = node_indices[rev_hash];
            }
        }
        {
            double rev_hash_2 = node->data->get_reverse_hash_2();
            auto it_mirror_2 = nodes.find(rev_hash_2);
            if (it_mirror_2 != nodes.end()) {
                mirror2s[i] = node_indices[rev_hash_2];
            }
        }
    }

    compute_repulsion_cuda(positions.data(), velocities.data(), adjacency_matrix.data(), mirrors.data(), mirror2s.data(), s, max_degree, attract, repel, mirror_force, decay, dimension, iterations);

    // TODO we should just permanently store the graph on the GPU, unless it is modified often?
    for (int i = 0; i < s; ++i) {
        node_vector[i]->position = positions[i];
        node_vector[i]->velocity = velocities[i];
    }

    mark_updated();
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

std::unordered_set<double> Graph::get_neighbors(double hash) {
    std::unordered_set<double> neighbors;
    if (!node_exists(hash)) {
        throw std::runtime_error("Node with hash " + std::to_string(hash) + " does not exist.");
        return neighbors;
    }
    const Node& node = nodes.at(hash);
    for (const Edge& edge : node.neighbors) {
        neighbors.insert(edge.to);
    }
    return neighbors;
}
