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

template <class T>
class Node {
public:
    /**
     * Constructor to create a new node.
     * @param t The data associated with the node.
     */
    Node(T* t, double hash) : data(t), hash(hash),
        velocity(10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX), 
        position(10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX,
                 10 * static_cast<double>(rand()) / RAND_MAX) {}

    T* data;
    double hash = 0;
    bool highlight = false;
    EdgeSet neighbors;
    double opacity = 1;
    int color = 0xffffffff;
    bool flooded = false;
    bool immobile = false;
    glm::dvec4 velocity;
    glm::dvec4 position;
};

/**
 * A template class representing a graph.
 * @tparam T The type of data stored in the nodes of the graph.
 */
template <class T>
class Graph : public DataObject {
public:
    int size() const {
        return nodes.size();
    }

    deque<double> traverse_deque;
    unordered_map<double, Node<T>> nodes;
    double root_node_hash = 0;

    double gravity_strength = 0;
    double decay = .90;
    double speedlimit = 10;
    double repel_force = .4;
    double attract_force = .4;
    int dimensions = 2;

    Graph(){}

    ~Graph() {
        clear();
    }

    void clear() {
        while (nodes.size()>0) {
            auto i = nodes.begin();
            delete i->second.data;
            nodes.erase(i->first);
        }
    }

    void add_to_stack(T* t){
        //cout << "Adding to stack... length before adding is " << traverse_deque.size() << endl;
        double hash = t->get_hash();
        root_node_hash = hash;
        add_node(t);
        traverse_deque.push_front(hash);
    }

    /**
     * Add a node to the graph.
     * @param t The data associated with the node.
     * @return hash The hash/id of the node which was added
     */
    double add_node(T* t){
        double x = add_node_without_edges(t);
        //cout << "Manually adding missing edges cause a human manually added a node" << endl;
        add_missing_edges(true);
        return x;
    }
    double add_node_without_edges(T* t){
        double hash = t->get_hash();
        if(size()%500 == 999) cout << "Node count: " << size() << endl;
        if (node_exists(hash)) {
            delete t;
            return hash;
        }
        Node<T> new_node(t, hash);
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
    bool expand_graph(bool only_one = false) {
        int new_nodes_added = 0;
        while (!traverse_deque.empty()) {
            double id = traverse_deque.front();
            traverse_deque.pop_front();

            unordered_set<T*> child_nodes = nodes.at(id).data->get_children();
            for (const auto& child : child_nodes) {
                double child_hash = child->get_hash();
                if (!node_exists(child_hash)) {
                    add_node_without_edges(child);
                    new_nodes_added++;
                    if (only_one) traverse_deque.push_front(id);

                    traverse_deque.push_back(child_hash); // This is bfs. To change to dfs, push_front here.

                    if (only_one) {add_missing_edges(true); return new_nodes_added;}
                }
            }
        }
        add_missing_edges(true);
        return new_nodes_added;
    }

    void add_node_with_position(T* t, double x, double y, double z) {
        double hash = add_node(t);
        move_node(hash, x, y, z);
    }

    void move_node(double hash, double x, double y, double z) {
        auto it = nodes.find(hash);
        if (it == nodes.end()) return;
        Node<T>& node = it->second;
        node.x = x;
        node.y = y;
        node.z = z;
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
    }
    
    /**
     * Remove an edge between two nodes.
     * @param from The hash of the source node.
     * @param to The hash of the destination node.
     */
    void remove_edge(double from, double to) {
        if (!node_exists(from) || !node_exists(to)) return;
        
        Node<T>& from_node = nodes.at(from);
        Edge edge_to_remove(from, to);
        
        from_node.neighbors.erase(edge_to_remove);
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
    void add_missing_edges(bool teleport_orphans_to_parents) {
        for (auto& pair : nodes) {
            Node<T>& parent = pair.second;
            unordered_set<double> child_hashes = parent.data->get_children_hashes();

            for (double child_hash : child_hashes) {
                // this theoretical child isn't guaranteed to be in the graph
                if(!node_exists(child_hash)) continue;
                Node<T>& child = nodes.find(child_hash)->second;
                if(teleport_orphans_to_parents && !does_edge_exist(parent.hash, child.hash)/*child is orphan*/){
                    child.position = parent.position;
                }
                add_directed_edge(parent.hash, child_hash);
            }
        }
    }

    /**
     * Mark all nodes presently in the graph as mobile / immobile.
     */
    void mobilize_all_nodes() {
        for (auto& pair : nodes) {
            pair.second.immobile = false;
        }
    }
    void immobilize_all_nodes() {
        for (auto& pair : nodes) {
            pair.second.immobile = true;
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
        Node<T>& node = nodes.at(id);
        for (const auto& neighbor_edge : node.neighbors) {
            double neighbor_id = neighbor_edge.to;
            nodes.at(neighbor_id).neighbors.erase(Edge(neighbor_id, id));
        }
        delete node->data;
        nodes.erase(id);
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

    void delete_orphans() {
        bool orphan_found;
        do {
            orphan_found = false;
            unordered_set<double> non_orphans;

            // Mark all nodes that are in the "to" position of any edge
            for (const auto& node_pair : nodes) {
                const Node<T>& node = node_pair.second;
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
    }

    /**
     * Iterate the physics engine to spread out graph nodes.
     * @param iterations The number of iterations to perform.
     */
    void iterate_physics(int iterations){
        vector<Node<T>*> node_vector;

        for (auto& node_pair : nodes) {
            node_vector.push_back(&node_pair.second);
        }

        for (int n = 0; n < iterations; n++) {
            if(n%10==9) cout << "Spreading out graph, iteration " << n << "." << endl;
                perform_single_physics_iteration(node_vector);
        }
    }

    void perform_single_physics_iteration(const vector<Node<T>*>& node_vector){
        int s = node_vector.size();
        glm::dvec4 center_of_mass(0,0,0,0);

        for (size_t i = 0; i < s; ++i) {
            Node<T>* node = node_vector[i];
            center_of_mass += node->position;
            for (size_t j = i+1; j < s; ++j) {
                Node<T>* node2 = node_vector[j];
                perform_pairwise_node_motion(node, node2, true);
            }
            const EdgeSet& neighbor_nodes = node->neighbors;
            for (const Edge& neighbor_edge : neighbor_nodes) {
                double neighbor_id = neighbor_edge.to;
                Node<T>* neighbor = &nodes.at(neighbor_id);
                perform_pairwise_node_motion(node, neighbor, false);
            }
        }
        center_of_mass /= s;

        for (size_t i = 0; i < s; ++i) {
            Node<T>* node = node_vector[i];
            if(node->immobile) continue;

            double magnitude = glm::length(node->velocity);
            if(magnitude > speedlimit) {
                double scale = speedlimit / magnitude;
                node->velocity *= scale;
            }

            node->velocity.y += gravity_strength;
            node->velocity *= decay;
            node->position += node->velocity - center_of_mass;
            if(dimensions < 3) {node->velocity.z = 0; node->position.z = 0;}
            if(dimensions < 4) {node->velocity.w = 0; node->position.w = 0;}
        }
    }

    double get_attraction_force(double dist_sq){
        return attract_force * (dist_sq-1)/dist_sq;
    }

    double get_repulsion_force(double dist_sq){
        return -repel_force / dist_sq;
    }

    void perform_pairwise_node_motion(Node<T>* node1, Node<T>* node2, bool repulsion_mode) {
        glm::dvec4 delta = node1->position - node2->position;
        double dist_sq = square(delta.x) + square(delta.y) + square(delta.z) + square(delta.w) + 1;
        double force = repulsion_mode?get_repulsion_force(dist_sq):get_attraction_force(dist_sq);
        glm::dvec4 change = delta * force;
        node2->velocity += change;
        node1->velocity -= change;
    }

    void render_json(string json_out_filename) {
        ofstream myfile;
        myfile.open(json_out_filename);

        json json_data;

        json nodes_to_use;
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            const Node<T>& node = it->second;
            json node_info;
            node_info["x"] = node.position.x;
            node_info["y"] = node.position.y;
            node_info["z"] = node.position.z;
            node_info["representation"] = node.data->representation;
            node_info["highlight"] = node.highlight;
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

        myfile << setw(4) << json_data;

        myfile.close();
        cout << "Rendered json!" << endl;
    }
};

