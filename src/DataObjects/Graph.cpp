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
#include "../misc/json.hpp"
using json = nlohmann::json;

extern "C" void compute_repulsion_cuda(const glm::vec4* h_positions, glm::vec4* h_velocity_deltas, int num_nodes);

glm::vec4 random_unit_cube_vector() {
    return glm::vec4(1 * static_cast<float>(rand()) / RAND_MAX,
                     1 * static_cast<float>(rand()) / RAND_MAX,
                     1 * static_cast<float>(rand()) / RAND_MAX,
                     1 * static_cast<float>(rand()) / RAND_MAX);
}

class Edge {
public:
    Edge(double f, double t) : from(f), to(t), opacity(1) {}
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
    unordered_set<double> expected_children_hashes;
    EdgeSet neighbors;
    double opacity = 1;
    int color = 0;
    float radius_multiplier = 1;
    double age = 0;
    glm::vec4 velocity;
    glm::vec4 position;
    float weight() const { return sigmoid(age*.2f + 0.01f); }
    double radius() const { return radius_multiplier * (((3*age - 1) * exp(-.5*age)) + 1); }
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

    double gravity_strength = 0;
    double speedlimit = 20;
    int dimensions = 3;

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
        //cout << "Adding to stack... length before adding is " << traverse_deque.size() << endl;
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
        //cout << "Manually adding missing edges cause a human manually added a node" << endl;
        add_missing_edges();
        return x;
    }
    double add_node_without_edges(GenericBoard* t){
        //cout << "total nodes: " << size() << endl;
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
        if(n == 0) throw runtime_error("expand argument was 0. positive: finite node count. negative: full graph.");
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
    void add_directed_edge(double from, double to) {
        // Check if both nodes exist in the graph
        if (!node_exists(from) || !node_exists(to)) return;
        nodes.at(from).neighbors.insert(Edge(from, to));
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

    /**
     * Iterate the physics engine to spread out graph nodes.
     * @param iterations The number of iterations to perform.
     */
    void iterate_physics(const int iterations, const float repel, const float attract, const float decay, const float centering_strength, const double z_dilation, const float mirror_force) {
        vector<Node*> node_vector;

        for (auto& node_pair : nodes) node_vector.push_back(&node_pair.second);
        for (int n = 0; n < iterations; n++) {
            for (int i = 0; i < node_vector.size(); ++i) { node_vector[i]->age += 1./iterations; }
            cout << "." << flush;
            perform_single_physics_iteration(node_vector, repel, attract, decay, centering_strength, z_dilation, mirror_force);
        }
        glm::vec4 com = center_of_mass();
        for (int n = 0; n < size(); n++) {
            node_vector[n]->position -= com*centering_strength;
        }
        mark_updated();
    }

    void perform_single_physics_iteration(const vector<Node*>& node_vector, const float repel, const float attract, const float decay, const float centering_strength, const double z_dilation, const float mirror_force) {
        int s = node_vector.size();

        vector<glm::vec4> positions(s);
        vector<glm::vec4> velocity_deltas(s, glm::vec4(0.0f));

        // Populate positions array
        for (int i = 0; i < s; ++i) {
            positions[i] = node_vector[i]->position;
        }

        // Use CUDA to compute repulsion velocity deltas
        compute_repulsion_cuda(positions.data(), velocity_deltas.data(), s);

        // Apply velocity deltas from CUDA and calculate attraction forces on the CPU
        for (int i = 0; i < s; ++i) {
            Node* node = node_vector[i];
            if(glm::any(glm::isnan(velocity_deltas[i]))) velocity_deltas[i] = glm::vec4(0, 0, 0, 0);
            if(glm::any(glm::isnan(node->position))) node->position = glm::vec4(0, 0, 0, 0);
            node->velocity += repel * velocity_deltas[i]; // Repulsion forces from CUDA

            // Add symmetry forces
            if (mirror_force > 0.001) {
                const auto& mirror = nodes.find(node->data->get_reverse_hash());
                if(mirror != nodes.end()) {
                    glm::vec4 mirror_pos = mirror->second.position;
                    mirror_pos.x *= -1;
                    node->velocity += mirror_force*(mirror_pos - node->position);
                }
               //else {cout << "Mirror not found!" << endl;}
            }

            // Calculate attraction forces (CPU)
            const EdgeSet& neighbor_nodes = node->neighbors;
            for (const Edge& neighbor_edge : neighbor_nodes) {
                double neighbor_id = neighbor_edge.to;
                Node* neighbor = &nodes.at(neighbor_id);
                glm::vec4 diff = node->position - neighbor->position;
                float dist_sq = glm::dot(diff, diff) + 1;
                glm::vec4 force = diff * attract * get_attraction_force(dist_sq);

                node->velocity -= force; // Apply attraction forces
                neighbor->velocity += force; // Apply attraction forces
            }
        }

        // Second loop: scale node positions and apply physics
        for (size_t i = 0; i < s; ++i) {
            Node* node = node_vector[i];

            double magnitude = glm::length(node->velocity);
            if (magnitude > speedlimit) {
                double scale = speedlimit / magnitude;
                node->velocity *= scale;
            }

            node->velocity.y += gravity_strength / size();
            node->velocity *= decay;
            node->position += node->velocity;

            // Slight force which tries to flatten the thinnest axis onto the view plane
            node->position.z *= z_dilation;
            node->position.w *= 0.99;

            // Dimensional constraints
            if (dimensions < 3) {
                node->velocity.z = 0;
                node->position.z = 0;
            }
            if (dimensions < 4) {
                node->velocity.w = 0;
                node->position.w = 0;
            }
        }
        mark_updated();
    }

    float get_attraction_force(float dist_sq){
        float dist_6th = dist_sq*dist_sq*dist_sq*.05f;
        return (dist_6th-1)/(dist_6th+1)*.2f-.1f;
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

        for (const auto& node_pair : nodes) {
            const Node& node = node_pair.second;
            float sig = node.weight();
            sum_distance_sq += sig * glm::dot(node.position, node.position);
            ct += sig;
        }

        float ans = 6 + 4.8*pow(sum_distance_sq / ct, .5);
        return ans;
    }

    /*
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
                ostringstream oss;
                oss << setprecision(numeric_limits<double>::digits10 + 2) << neighbor.to;
                neighbors.push_back(oss.str());
            }
            node_info["neighbors"] = neighbors;

            ostringstream oss;
            oss << setprecision(numeric_limits<double>::digits10 + 2) << it->first;
            nodes_to_use[oss.str()] = node_info;
        }

        json_data["nodes_to_use"] = nodes_to_use;
        json_data["nodes_to_use"].dump(4, ' ', false, json::error_handler_t::ignore);

        ostringstream oss;
        oss << setprecision(numeric_limits<double>::digits10 + 2) << root_node_hash;
        json_data["root_node_hash"] = oss.str();
        json_data["board_w"] = nodes.find(root_node_hash)->second.data->BOARD_WIDTH;
        json_data["board_h"] = nodes.find(root_node_hash)->second.data->BOARD_HEIGHT;
        json_data["game_name"] = nodes.find(root_node_hash)->second.data->game_name;

        myfile << json_data.dump();

        myfile.close();
        cout << "Rendered json!" << endl;
    }
    */
};
