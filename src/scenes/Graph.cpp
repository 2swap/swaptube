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
#include "GenericBoard.cpp"
#include "json.hpp"
using json = nlohmann::json;

class Edge {
public:
    Edge(double f, double t) : from(f), to(t), opacity(1) {}
    double to;
    double from;
    double opacity;

    //Stuff needed to slap it in an unordered set
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

template <class T>
class Node {
public:
    /**
     * Constructor to create a new node.
     * @param t The data associated with the node.
     */
    Node(T* t, double hash) : data(t), hash(hash) {}
    double hash = 0;
    bool highlight = false;
    T* data;
    EdgeSet neighbors;
    double opacity = 1;
    int color = 0xffffffff;
    bool flooded = false;
    bool immobile = false;
    double vx = (double)rand() / (RAND_MAX), vy = (double)rand() / (RAND_MAX), vz = (double)rand() / (RAND_MAX), vw = (double)rand() / (RAND_MAX);
    double x = (double)rand() / (RAND_MAX), y = (double)rand() / (RAND_MAX), z = (double)rand() / (RAND_MAX), w = (double)rand() / (RAND_MAX);
};

/**
 * A template class representing a graph.
 * @tparam T The type of data stored in the nodes of the graph.
 */
template <class T>
class Graph {
public:
    int size() const {
        return nodes.size();
    }

    std::deque<double> traverse_deque;
    std::unordered_map<double, Node<T>> nodes;
    double root_node_hash = 0;

    double gravity_strength = 0;
    double decay = .9;
    double speedlimit = 3;
    double repel_force = 1;
    double attract_force = 1;
    int dimensions = 2;
    bool lock_root_at_origin = true;
    bool sqrty = true;

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
        double hash = t->get_hash();
        std::cout << "adding node with hash: " << hash << " and representation " << t->representation << std::endl;
        if (node_exists(hash)) {
            delete t;
            return hash;
        }
        nodes.emplace(hash, Node<T>(t, hash));
        int s = size();
        if (s == 1) root_node_hash = hash;
        add_missing_edges(true);
        if (hash == root_node_hash && lock_root_at_origin) {
            Node<T>& just_inserted = nodes.at(hash);
            just_inserted.x = just_inserted.y = just_inserted.z = 0;
        }
        return hash;
    }

    void add_node_with_position(T* t, double x, double y, double z) {
        double hash = add_node(t);
        auto it = nodes.find(hash);
        if (it == nodes.end()) return;
        Node<T>& node = it->second;
        node.x = x;
        node.y = y;
        node.z = z;
    }

    /**
     * Expand the graph by adding neighboring nodes.
     */
    void expand_graph(bool is_dfs, bool only_one = false) {
        while (!traverse_deque.empty()) {
            double id = is_dfs ? traverse_deque.front() : traverse_deque.back();

            std::unordered_set<T*> neighbor_nodes = nodes.at(id).data->get_children();
            for (const auto& neighbor : neighbor_nodes) {
                double child_hash = neighbor->get_hash();
                add_directed_edge(id, child_hash);
                if (!node_exists(child_hash)) {
                    add_node(neighbor);
                    if (is_dfs) traverse_deque.push_front(child_hash);
                    else        traverse_deque.push_back (child_hash);
                    if (only_one) return;
                }
            }
            if (is_dfs) traverse_deque.pop_front();
            else        traverse_deque.pop_back();
        }
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
            std::unordered_set<double> child_hashes;

            std::unordered_set<T*> children_nodes = parent.data->get_children();
            for (const auto& child : children_nodes) {
                child_hashes.insert(child->get_hash());
            }

            for (double child_hash : child_hashes) {
                // this theoretical child isn't guaranteed to be in the graph
                if(!node_exists(child_hash)) continue;
                Node<T>& child = nodes.find(child_hash)->second;
                bool child_is_orphan = !does_edge_exist(parent.hash, child.hash);
                if(teleport_orphans_to_parents && child_is_orphan){
                    child.x = parent.x;
                    child.y = parent.y;
                    child.z = parent.z;
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
    std::pair<std::list<double>, std::list<Edge*>> shortest_path(double start, double end) {
        std::unordered_map<double, double> distances;
        std::unordered_map<double, double> predecessors;
        std::unordered_set<double> visited;
        auto compare = [&](double lhs, double rhs) { return distances[lhs] > distances[rhs]; };
        std::priority_queue<double, std::vector<double>, decltype(compare)> priority_queue(compare);

        // Initialize distances
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
                double new_distance = distances[current] + 1; // assuming all edges have equal weight
                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    predecessors[neighbor] = current;
                    priority_queue.push(neighbor);
                }
            }
        }

        // Reconstruct the shortest path
        std::list<double> path;
        std::list<Edge*> edges;
        for (double at = end; at != start; at = predecessors[at]) {
            if (predecessors.find(at) == predecessors.end()) {
                // If there's no path from start to end, return empty lists
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

    void delete_orphans() {
        bool orphan_found;
        do {
            orphan_found = false;
            std::unordered_set<double> non_orphans;

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
        std::vector<Node<T>*> node_vector; // Change from list to vector

        for (auto& node_pair : nodes) {
            node_vector.push_back(&node_pair.second); // Add it to the vector
        }
        int s = node_vector.size();

        for (int n = 0; n < iterations; n++) {
            if(n%10==0) std::cout << "Spreading out graph, iteration " << n << ". Node count = " << s << std::endl;

            for (size_t i = 0; i < s; ++i) {
                Node<T>* node = node_vector[i];
                for (size_t j = i+1; j < s; ++j) {
                    Node<T>* node2 = node_vector[j];
                    
                    double dx = node2->x - node->x;
                    double dy = node2->y - node->y;
                    double dz = node2->z - node->z;
                    double dw = node2->w - node->w;
                    double force = repel_force;
                    double dist_sq = dx * dx + dy * dy + dz * dz + dw * dw + .1;
                    if(sqrty){
                        force *= .5/dist_sq;
                    } else {
                        force *= .0025 / (dist_sq*dist_sq);
                    }
                    double nx = force * dx;
                    double ny = force * dy;
                    double nz = force * dz;
                    double nw = force * dw;

                    node2->vx += nx;
                    node2->vy += ny;
                    node2->vz += nz;
                    node2->vw += nw;
                    node->vx -= nx;
                    node->vy -= ny;
                    node->vz -= nz;
                    node->vw -= nw;
                }
                const EdgeSet& neighbor_nodes = node->neighbors;
                
                for (const Edge& neighbor_edge : neighbor_nodes) {
                    double neighbor_id = neighbor_edge.to;
                    Node<T>& neighbor = nodes.at(neighbor_id);
                    
                    double dx = node->x - neighbor.x;
                    double dy = node->y - neighbor.y;
                    double dz = node->z - neighbor.z;
                    double dw = node->w - neighbor.w;
                    double force = attract_force;
                    double dist_sq = dx * dx + dy * dy + dz * dz + dw * dw + 1;
                    if(sqrty){
                        force *= (dist_sq-1)/dist_sq;
                    } else {
                        force *= 1/dist_sq + dist_sq - 2;
                    }
                    double nx = force * dx;
                    double ny = force * dy;
                    double nz = force * dz;
                    double nw = force * dw;
                    
                    neighbor.vx += nx;
                    neighbor.vy += ny;
                    neighbor.vz += nz;
                    neighbor.vw += nw;
                    node->vx -= nx;
                    node->vy -= ny;
                    node->vz -= nz;
                    node->vw -= nw;
                }
            }

            for (size_t i = 0; i < s; ++i) {
                Node<T>* node = node_vector[i];
                if((lock_root_at_origin && node->hash == root_node_hash) || node->immobile){
                    continue;
                }
                double magnitude = std::sqrt(node->vx * node->vx + node->vy * node->vy + node->vz * node->vz + node->vw * node->vw);
                if(magnitude > speedlimit) {
                    double scale = speedlimit / magnitude;

                    node->vx *= scale;
                    node->vy *= scale;
                    node->vz *= scale;
                    node->vw *= scale;
                }
                node->vy += gravity_strength;
                node->vx *= decay;
                node->vy *= decay;
                node->vz *= decay;
                node->vw *= decay;
                node->x += node->vx;
                node->y += node->vy;
                if(dimensions>=3)
                    node->z += node->vz;
                else
                    node->z = 0;
                if(dimensions>=4)
                    node->w = (node->w + node->vw)*.9;
                else
                    node->w = 0;
            }
        }
    }
};

