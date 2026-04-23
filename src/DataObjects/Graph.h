#pragma once

#include "../Host_Device_Shared/vec.h"
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <vector>
#include <list>
#include <random>
#include <string>
#include <utility>
#include "DataObject.h"
#include "GenericBoard.h"

class Edge {
public:
    Edge(double f, double t) : from(f), to(t) {}
    double from;
    double to;

    bool operator==(const Edge& other) const {
        return (this->to == other.to && this->from == other.from);
    }
    struct HashFunction {
        size_t operator()(const Edge& edge) const {
            size_t toHash = std::hash<int>()(edge.to);
            size_t fromHash = std::hash<int>()(edge.from) << 1;
            return toHash ^ fromHash;
        }
    };
};
using EdgeSet = std::unordered_set<Edge, Edge::HashFunction, std::equal_to<Edge>>;

vec4 random_unit_cube_vector(std::mt19937& rng, std::uniform_real_distribution<float>& dist);

class Node {
public:
    Node(GenericBoard* t, double hash, vec4 position, vec4 velocity);

    GenericBoard* data;
    double hash = 0;
    EdgeSet neighbors;
    vec4 velocity;
    vec4 position;
};

class Graph : public DataObject {
public:
    Graph();
    ~Graph();

    int size() const;

    std::unordered_map<double, Node> nodes;
    std::uniform_real_distribution<float> dist;
    std::mt19937 rng;

    void clear_queue();
    void clear();

    double add_node(GenericBoard* t);
    double add_node_without_edges(GenericBoard* t);

    int expand(int n = -1);

    void add_node_with_neighbors(GenericBoard* t, std::vector<double> neighbor_hashes);
    void move_node(double hash, vec4 pos);

    void add_edge(double from, double to, double opacity = 1);
    void remove_edge(double from, double to);
    const Edge* get_edge(double from, double to);
    bool does_edge_exist(double from, double to);

    void add_missing_edges();

    bool node_exists(double id) const;

    void remove_node(double id);

    int measure_distance(double start, double end);
    std::pair<std::list<double>, std::list<Edge*>> shortest_path(double start, double end);

    std::vector<int> make_adjacency_matrix(const std::vector<Node*>& node_vector, int &max_degree);

    void iterate_physics(const int iterations, const float repel, const float attract, const float decay, const double dimension, const float mirror_force);

    std::unordered_set<double> get_neighborhood(double hash, int dist);
    std::unordered_set<double> get_neighbors(double hash);

    void tick(const StateReturn& state);

private:
    int last_node_count = -1;
};
