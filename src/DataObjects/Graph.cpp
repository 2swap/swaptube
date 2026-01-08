#pragma once

#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <list>
#include <deque>
#include <random>
#include <limits.h>
#include <glm/glm.hpp>
#include <queue>
#include <cstdlib>
#include "DataObject.cpp"
#include "GenericBoard.cpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

extern "C" void compute_repulsion_cuda(glm::vec4* h_positions, glm::vec4* h_velocities, const int* h_adjacency_matrix, const int* h_mirrors, const int* h_mirror2s, int num_nodes, int max_degree, float attract, float repel, float mirror_force, const float decay, const float dimension, const int iterations);

glm::vec4 random_unit_cube_vector() {
    return glm::vec4(
        2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f,
        2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f,
        2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f,
        2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f
    );
}

class Edge {
public:
    Edge(double f, double t, double opa = 1) : from(f), to(t), opacity(opa) {}
    double from;
    double to;
    double opacity;

    //Stuff needed to slap it in an unordered set
    bool operator==(const Edge& other) const {
        return (this->to == other.to && this->from == other.from);
    }
    struct HashFunction {
        size_t operator()(const Edge& edge) const {
            size_t toHash = hash<int>()(edge.to);
            size_t fromHash = hash<int>()(edge.from) << 1;
            return toHash ^ fromHash;
        }
    };
};
using EdgeSet = unordered_set<Edge, Edge::HashFunction, equal_to<Edge>>;

class Node {
public:
    /**
     * Constructor to create a new node.
     * @param t The data associated with the node.
     */
    Node(GenericBoard* t, double hash) : data(t), hash(hash),
        velocity(random_unit_cube_vector()),
        position(random_unit_cube_vector()) {}

    GenericBoard* data;
    double hash = 0;
    int index = -1;
    unordered_set<double> expected_children_hashes;
    EdgeSet neighbors;
    double opacity = 1;
    int color = 0xffffffff;
    float size = 1;
    double age = 0;
    glm::vec4 velocity;
    glm::vec4 position;
    float weight() const { return sigmoid(age*.2f + 0.01f); }
    double radius() const { return size * (((3*age - 1) * exp(-.5*age)) + 1); }
    double splash_opacity() const { return 1-square(age/12.); }
    double splash_radius() const { return size * age * .8; }
};

/**
 * A template class representing a graph.
 * @tparam T The type of data stored in the nodes of the graph.
 */
class Graph : public DataObject {
public:
    int size() const {
        return nodes.size();
    }

    deque<double> traverse_deque;
    unordered_map<double, Node> nodes;
    double root_node_hash = 0;

    Graph(){}

    ~Graph() {
        clear();
    }

    void clear_queue() {
        traverse_deque.clear();
    }

    void clear() {
        traverse_deque.clear();
        while (nodes.size()>0) {
            auto i = nodes.begin();
            delete i->second.data;
            nodes.erase(i->first);
        }
    }

    void add_to_stack(GenericBoard* t){
        double hash = t->get_hash();
        add_node_without_edges(t);
        traverse_deque.push_front(hash);
    }

    /**
     * Add a node to the graph.
     * @param t The data associated with the node.
     * @return hash The hash/id of the node which was added
     */
    double add_node(GenericBoard* t){
        double x = add_node_without_edges(t);
        add_missing_edges();
        return x;
    }
    double add_node_without_edges(GenericBoard* t){
        double hash = t->get_hash();
        if (node_exists(hash)) {
            delete t;
            return hash;
        }
        Node new_node(t, hash);
        if (size() == 0) {
            root_node_hash = hash;
        }
        nodes.emplace(hash, new_node);
        return hash;
    }

    /**
     * Expand the graph by adding neighboring nodes.
     * Return amount of new nodes that were added.
     */
    int expand(int n = -1) {
        if(n == 0) return 0;
        int added = 0;
        while (!traverse_deque.empty()) {
            double id = traverse_deque.front();
            traverse_deque.pop_front();

            unordered_set<GenericBoard*> child_nodes = nodes.at(id).data->get_children();
            bool done = false;
            for (const auto& child : child_nodes) {
                double child_hash = child->get_hash();
                if (done || node_exists(child_hash)) delete child;
                else {
                    add_node_without_edges(child);
                    if(n>0) traverse_deque.push_front(id); // No need to save progress if expanding whole graph
                    traverse_deque.push_back(child_hash); // push_back: bfs // push_front: dfs
                    added++;
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

    void add_node_with_position(GenericBoard* t, double x, double y, double z) {
        double hash = add_node(t);
        move_node(hash, x, y, z);
    }

    void move_node(double hash, double x, double y, double z, double w=0) {
        auto it = nodes.find(hash);
        if (it == nodes.end()) return;
        Node& node = it->second;
        node.position = glm::vec4(x,y,z,w);
        mark_updated();
    }

    /**
     * Connect two nodes in the graph.
     * @param node1 The hash of the first node.
     * @param node2 The hash of the second node.
     */
    void add_directed_edge(double from, double to, double opacity = 1) {
        // Check if both nodes exist in the graph
        if (!node_exists(from) || !node_exists(to)) return;
        nodes.at(from).neighbors.insert(Edge(from, to, opacity));
        mark_updated();
    }

    /**
     * Remove an edge between two nodes.
     * @param from The hash of the source node.
     * @param to The hash of the destination node.
     */
    void remove_edge(double from, double to) {
        if (!node_exists(from) || !node_exists(to)) return;
        
        Node& from_node = nodes.at(from);
        Edge edge_to_remove(from, to);
        
        from_node.neighbors.erase(edge_to_remove);
        mark_updated();
    }

    const Edge* get_edge(double from, double to) {
        if (!node_exists(from)) return nullptr;
        Node& from_node = nodes.at(from);
        for (const Edge& edge : from_node.neighbors) {
            if (edge.to == to) {
                return &edge;
            }
        }
        return nullptr;
    }

    bool does_edge_exist(double from, double to){
        if (!node_exists(from)) return false;

        // Get the set of edges of the from node
        EdgeSet& children = nodes.at(from).neighbors;

        // Iterate through the edges to check if any points to the "to" node
        for (const Edge& edge : children) {
            if (edge.to == to) {
                return true;
            }
        }

        return false;
    }

    /**
     * Sanitize the graph by adding edges which should be present but are not.
     */
    void add_missing_edges() {
        for (auto& pair : nodes) {
            Node& parent = pair.second;
            if(parent.expected_children_hashes.size() == 0)
                parent.expected_children_hashes = parent.data->get_children_hashes();

            for (double child_hash : parent.expected_children_hashes) {
                // this theoretical child isn't guaranteed to be in the graph
                if(!node_exists(child_hash)) continue;
                Node& child = nodes.find(child_hash)->second;
                if(child.age == 0 && parent.age != 0 && !does_edge_exist(parent.hash, child_hash)){//if child is orphaned
                    // Teleport children to parents
                    child.position = parent.position + random_unit_cube_vector();
                    child.velocity = parent.velocity;
                }
                add_directed_edge(parent.hash, child_hash);
            }
        }
    }

    /**
     * Check if a node with the given hash exists in the graph.
     * @param id The hash of the node to check.
     * @return True if the node exists, false otherwise.
     */
    bool node_exists(double id) const {
        return nodes.find(id) != nodes.end();
    }

    /**
     * Remove a node from the graph, along with all of its incident edges.
     * @param id The hash of the node to be removed.
     */
    void remove_node(double id) {
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

    int dist(double start, double end) {
        return shortest_path(start, end).first.size();
    }

    // Function to find the shortest path between two nodes using Dijkstra's algorithm
    pair<list<double>, list<Edge*>> shortest_path(double start, double end) {
        unordered_map<double, double> distances;
        unordered_map<double, double> predecessors;
        unordered_set<double> visited;
        auto compare = [&](double lhs, double rhs) { return distances[lhs] > distances[rhs]; };
        priority_queue<double, vector<double>, decltype(compare)> priority_queue(compare);

        // Initialize distances
        for (const auto& node_pair : nodes) {
            distances[node_pair.first] = numeric_limits<double>::max();
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
                double new_distance = distances[current] + 1; // assuming all edges have equal weight
                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    predecessors[neighbor] = current;
                    priority_queue.push(neighbor);
                }
            }
        }

        // Reconstruct the shortest path
        list<double> path;
        list<Edge*> edges;
        for (double at = end; at != start; at = predecessors[at]) {
            if (predecessors.find(at) == predecessors.end()) {
                // If there's no path from start to end, return empty lists
                return {list<double>(), list<Edge*>()};
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

    void collapse_two_nodes(double hash_keep, double hash_remove) {
        if (!node_exists(hash_keep) || !node_exists(hash_remove)) return;
        if (hash_keep == hash_remove) return;

        Node& node_keep = nodes.at(hash_keep);
        Node& node_remove = nodes.at(hash_remove);

        // Transfer edges from node_remove to node_keep
        for (const auto& edge : node_remove.neighbors) {
            if (edge.to != hash_keep) { // Avoid self-loop
                add_directed_edge(hash_keep, edge.to, edge.opacity);
                add_directed_edge(edge.to, hash_keep, edge.opacity);
            }
        }

        // Remove the node to be removed
        remove_node(hash_remove);
        mark_updated();
    }

    void delete_isolated() {
        unordered_set<double> non_isolated;

        // Mark all nodes that are in the "to" position of any edge
        for (const auto& node_pair : nodes) {
            const Node& node = node_pair.second;
            for (const auto& edge : node.neighbors) {
                non_isolated.insert(edge.to);
                non_isolated.insert(edge.from);
            }
        }

        // Iterate through the nodes and remove those that are not in the non_isolated set
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

    void delete_orphans() {
        bool orphan_found;
        do {
            orphan_found = false;
            unordered_set<double> non_orphans;

            // Mark all nodes that are in the "to" position of any edge
            for (const auto& node_pair : nodes) {
                const Node& node = node_pair.second;
                for (const auto& edge : node.neighbors) {
                    non_orphans.insert(edge.to);
                }
            }

            // Iterate through the nodes and remove those that are not in the non_orphans set
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

    // New function to create adjacency matrix for GPU attractive forces
    vector<int> make_adjacency_matrix(const vector<Node*>& node_vector, int &max_degree) {
        int n = node_vector.size();
        max_degree = 0;
        // We'll find the maximum outgoing degree (neighbors size)
        for (int i = 0; i < n; ++i) {
            int degree = node_vector[i]->neighbors.size();
            if (degree > max_degree) max_degree = degree;
        }

        // Allocate matrix n * max_degree, initialized to -1 (no neighbor)
        vector<int> adjacency_matrix(n * max_degree, -1);

        unordered_map<double, int> node_index_map;
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

    /**
     * Iterate the physics engine to spread out graph nodes.
     * @param iterations The number of iterations to perform.
     */
    void iterate_physics(const int iterations, const float repel, const float attract, const float decay, const float centering_strength, const double dimension, const float mirror_force, const bool flip_by_symmetry) {
        vector<Node*> node_vector;

        for (auto& node_pair : nodes) node_vector.push_back(&node_pair.second);
        for (int i = 0; i < node_vector.size(); ++i) { node_vector[i]->age++; }

        int s = node_vector.size();
        vector<glm::vec4> positions(s);
        vector<glm::vec4> velocities(s);

        // Populate positions array
        glm::vec4 com = center_of_mass() * centering_strength;
        for (int i = 0; i < s; ++i) {
             positions[i] = node_vector[i]->position - com;
            velocities[i] = node_vector[i]->velocity;
            node_vector[i]->index = i;
        }
        int max_degree = 0;
        vector<int> adjacency_matrix = make_adjacency_matrix(node_vector, max_degree);
        // Construct the mirrors and mirror2s vectors containing indices of reverse hashes
        vector<int> mirrors(s, -1);
        vector<int> mirror2s(s, -1);

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

    glm::vec4 center_of_mass() const {
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

    float af_dist() const {
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

    void render_json(string json_out_filename) {
        ofstream myfile;
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
                string neighbor_representation = nodes.at(neighbor.to).data->representation;
                if(neighbor_representation.size() < node.data->representation.size()) continue;
                ostringstream oss;
                oss << setprecision(17) << neighbor.to;
                neighbors.push_back(oss.str());
            }
            node_info["neighbors"] = neighbors;

            ostringstream oss;
            oss << setprecision(17) << it->first;
            nodes_to_use[oss.str()] = node_info;
        }

        json_data["nodes_to_use"] = nodes_to_use;
        json_data["nodes_to_use"].dump(4, ' ', false, json::error_handler_t::ignore);

        ostringstream oss;
        oss << setprecision(17) << root_node_hash;
        json_data["root_node_hash"] = oss.str();
        json_data["board_w"  ] = 7;
        json_data["board_h"  ] = 6;
        json_data["game_name"] = "c4";

        // Prepend "var dataset = " to the file
        myfile.seekp(0, ios::beg);
        myfile << "var dataset = ";
        myfile << json_data.dump();

        myfile.close();

        cout << "Rendered json!" << endl;
    }

    unordered_set<double> get_neighborhood(double hash, int dist) {
        unordered_set<double> neighborhood;
        if (!node_exists(hash) || dist < 0) return neighborhood;
        neighborhood.insert(hash);

        unordered_set<double> current_level_nodes;
        current_level_nodes.insert(hash);

        for (int d = 0; d < dist; ++d) {
            unordered_set<double> next_level_nodes;
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

    // Added function to make graph bidirectional by adding reverse edges
    void make_bidirectional() {
        // Collect all edges first to avoid modifying the container while iterating
        vector<pair<double, double>> edges_to_add;
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
        // Add reverse edges
        for (const auto& edge : edges_to_add) {
            add_directed_edge(edge.first, edge.second);
        }
    }
};
