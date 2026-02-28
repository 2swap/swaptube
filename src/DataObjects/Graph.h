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
    Edge(double f, double t, double opa = 1) : from(f), to(t), opacity(opa) {}
    double from;
    double to;
    double opacity;

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
    int index = -1;
    std::unordered_set<double> expected_children_hashes;
    EdgeSet neighbors;
    bool draw_point = false;
    double opacity = 1;
    int color = 0xffffffff;
    float size = 1;
    double age = 0;
    vec4 velocity;
    vec4 position;
    float weight() const;
    double radius() const;
    double splash_opacity() const;
    double splash_radius() const;
};

class Graph : public DataObject {
public:
    Graph();
    ~Graph();

    int size() const;

    std::deque<double> traverse_deque;
    std::unordered_map<double, Node> nodes;
    double root_node_hash;
    std::uniform_real_distribution<float> dist;
    std::mt19937 rng;

    void clear_queue();
    void clear();

    void add_to_stack(GenericBoard* t);

    double add_node(GenericBoard* t);
    double add_node_without_edges(GenericBoard* t);

    int expand(int n = -1);

    void add_node_with_position(GenericBoard* t, double x, double y, double z);
    void move_node(double hash, float x, float y, float z, float w = 0);

    void add_directed_edge(double from, double to, double opacity = 1);
    void remove_edge(double from, double to);
    const Edge* get_edge(double from, double to);
    bool does_edge_exist(double from, double to);

    void add_missing_edges();

    bool node_exists(double id) const;

    void remove_node(double id);

    int measure_distance(double start, double end);
    std::pair<std::list<double>, std::list<Edge*>> shortest_path(double start, double end);

    void collapse_two_nodes(double hash_keep, double hash_remove);

    void delete_isolated();
    void delete_orphans();

    std::vector<int> make_adjacency_matrix(const std::vector<Node*>& node_vector, int &max_degree);

    void iterate_physics(const int iterations, const float repel, const float attract, const float decay, const float centering_strength, const double dimension, const float mirror_force, const bool flip_by_symmetry);

    vec4 center_of_mass() const;

    float af_dist() const;

    void render_json(std::string json_out_filename);

    std::unordered_set<double> get_neighborhood(double hash, int dist);

    void make_bidirectional();
};
