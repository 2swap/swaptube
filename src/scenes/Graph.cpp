#pragma once

#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <list>
#include <queue>
#include <stack>
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
     * @param dist The distance of the node from the root.
     */
    Node(T* t, double hash) : data(t), hash(hash) {}
    int dist = 0;
    double hash = 0;
    bool highlight = false;
    T* data;
    EdgeSet neighbors;
    bool flooded = false;
    bool immobile = false;
    double vx = (double) rand() / (RAND_MAX), vy = (double) rand() / (RAND_MAX), vz = (double) rand() / (RAND_MAX), vw = (double) rand() / (RAND_MAX);
    double x = (double) rand() / (RAND_MAX), y = (double) rand() / (RAND_MAX), z = (double) rand() / (RAND_MAX), w = (double) rand() / (RAND_MAX);
};

/**
 * A template class representing a graph.
 * @tparam T The type of data stored in the nodes of the graph.
 */
template <class T>
class Graph{
public:
    /**
     * Get the number of nodes in the graph.
     * @return The number of nodes in the graph.
     */
    int size(){
        return nodes.size();
    }

    std::stack<double> dfs_stack;
    std::queue<double> bfs_queue;
    std::unordered_map<double, Node<T>> nodes;
    std::map<int, std::pair<int, int>> dist_count;
    double root_node_hash = 0;




    double gravity_strength = 0;
    double decay = .9;
    double speedlimit = 3;
    double repel_force = 1;
    double attract_force = 1;
    int dimensions = 2;
    bool lock_root_at_origin = false;
    bool sqrty = true;
    bool iterate_and_render_on_add = false;
    std::string json_out_filename = "thisshouldntappear";

    Graph(){}

    ~Graph() {
        clear();
    }

    void clear(){
        while(nodes.size()>0){
            auto i = nodes.begin();
            delete i->second.data;
            nodes.erase(i->first);
        }
    }

    void add_to_stack(T* t){
        double hash = t->get_hash();
        root_node_hash = hash;
        Node<T> n(t, hash);
        add_node(t);
        dfs_stack.push(hash);
        bfs_queue.push(hash);
    }

    /**
     * Add a node to the graph.
     * @param t The data associated with the node.
     * @return hash The hash/id of the node which was added
     */
    double add_node(T* t){
        double hash = t->get_hash();
        std::cout << "adding node with hash: " << hash << " and representation " << t->representation << std::endl;
        if(node_exists(hash)) {
            //std::cout << t->representation << "deleted since it was already in the graph" << std::endl;
            delete t;
            return hash;
        }
        Node<T> n(t, hash);
        nodes.insert(std::make_pair(hash,n));
        int s = size();
        if(s == 1){
            root_node_hash = hash;
        }
        if(s%100 == 0) std::cout << s << " nodes and counting..." << std::endl;
        make_edges_bidirectional();
        add_edges_that_are_missing_for_specific_node_and_teleport_to_parent(hash);
        if(hash == root_node_hash && lock_root_at_origin){
            Node<T>& just_inserted = nodes.find(hash)->second;
            just_inserted.x = just_inserted.y = just_inserted.z = 0;
        }
        if(iterate_and_render_on_add){
            iterate_physics(3);
        }
        return hash;
    }

    void add_node_with_position(T* t, double x, double y, double z){
        double hash = add_node(t);
        auto it = nodes.find(hash);
        if(it == nodes.end()) return;
        Node<T>& node = it->second;
        node.x = x;
        node.y = y;
        node.z = z;
    }

    /**
     * Expand the graph by adding neighboring nodes.
     */
    void expand_graph_dfs(){
        while(true){
            double id;
            {
                if (dfs_stack.empty()) {
                    return;
                }
                id = dfs_stack.top();
                dfs_stack.pop();
            }

            std::unordered_set<T*> neighbor_nodes = nodes.find(id)->second.data->get_neighbors();
            for(auto it = neighbor_nodes.begin(); it != neighbor_nodes.end(); ++it){
                double child_hash = (*it)->get_hash();
                {
                    nodes.find(id)->second.neighbors.insert(Edge(id, child_hash));
                    if(!node_exists(child_hash)){
                        add_node(*it);
                        dfs_stack.push(child_hash);
                    }
                }
            }
        }
        std::cout << "Finished expansion" << std::endl;
    }

    /**
     * Expand the graph by adding neighboring nodes.
    void expand_graph_bfs(){
        while(true){
            double id;
            {
                if (bfs_queue.empty()) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::cout << "Sleeping and awaiting more work..." << std::endl;
                    return;
                }
                id = bfs_queue.front();
                bfs_queue.pop();
            }

            std::unordered_set<T*> neighbor_nodes = nodes.find(id)->second.data->get_neighbors();
            for(auto it = neighbor_nodes.begin(); it != neighbor_nodes.end(); ++it){
                double child_hash = (*it)->get_hash();
                {
                    nodes.find(id)->second.neighbors.insert(child_hash);
                    if(!node_exists(child_hash)){
                        add_node(*it);
                        bfs_queue.push(child_hash);
                    }
                }
            }
        }
        std::cout << "Finished expansion" << std::endl;
    }
     */

    void highlight_nodes(Graph<T>& g){
        int ct = 0;
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            double node_id = it->first;
            if(g.node_exists(node_id)) {it->second.highlight = true;ct++;}
        }
        std::cout << "Nodes highlighted: " << ct << std::endl;
    }

    /**
     * Connect two nodes in the graph.
     * @param node1 The hash of the first node.
     * @param node2 The hash of the second node.
     */
    void connect_nodes(double node1, double node2) {
        // Check if both nodes exist in the graph
        if (!node_exists(node1) || !node_exists(node2)) {
            return; // One or both nodes don't exist
        }

        // Add edge between the two nodes
        nodes.find(node1)->second.neighbors.insert(Edge(node1, node2));
        nodes.find(node2)->second.neighbors.insert(Edge(node2, node1));
    }

    /**
     * Make all edges in the graph bidirectional by identifying unidirectional edges.
     */
    void make_edges_bidirectional() {
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            double node_id = it->first;
            Node<T>* node = &(it->second);
            EdgeSet neighbor_nodes = node->neighbors;
            for(Edge neighbor_edge : neighbor_nodes){
                double neighbor_id = neighbor_edge.to;
                Node<T>* neighbor = &(nodes.find(neighbor_id)->second);
                // Check if the edge is unidirectional (one node points to the other)
                neighbor->neighbors.insert(Edge(neighbor_id, node_id));
            }
        }
    }

    /**
     * Sanitize the graph for closure under edges.
     */
    void sanitize_for_closure() {
        add_edges_that_are_missing();
        delete_edges_that_point_nowhere();
    }

    /**
     * Sanitize the graph by adding edges which should be present but are not.
     */
    void add_edges_that_are_missing_for_specific_node_and_teleport_to_parent(double hash) {
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            Node<T>& node = it->second;
            std::unordered_set<double> neighbor_hashes;

            //if(node.neighbors.size() == 0){ // this if doesnt work because sometimes we only add one neighbor and prune the others for closure
                // if there are no neighbors YET IN THE LIST. We may have already found the neighbors of this node and there might just not be any. TODO add bool to track.
                std::unordered_set<T*> neighbor_nodes = node.data->get_neighbors();
                for(T* neighbor_it : neighbor_nodes){
                    neighbor_hashes.insert(neighbor_it->get_hash());
                }
            //}
            //else {
            //    neighbor_hashes = node.neighbors;
            //}

            for(double neighbor_hash : neighbor_hashes){
                Node<T>& neighbor = nodes.find(neighbor_hash)->second;
                if (node_exists(neighbor_hash)) {
                    if (node.hash == hash) {
                        connect_nodes(node.hash, neighbor_hash);
                        node.x = neighbor.x;
                        node.y = neighbor.y;
                        node.z = neighbor.z;
                    }
                    if (neighbor_hash == hash) {
                        connect_nodes(node.hash, neighbor_hash);
                        neighbor.x = node.x;
                        neighbor.y = node.y;
                        neighbor.z = node.z;
                    }
                }
            }
        }
    }

    /**
     * Sanitize the graph by adding edges which should be present but are not.
     */
    void add_edges_that_are_missing() {
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            const Node<T>& node = it->second;
            std::unordered_set<double> neighbor_hashes;

            //if(node.neighbors.size() == 0){ // this if doesnt work because sometimes we only add one neighbor and prune the others for closure
                // if there are no neighbors YET IN THE LIST. We may have already found the neighbors of this node and there might just not be any. TODO add bool to track.
                std::unordered_set<T*> neighbor_nodes = node.data->get_neighbors();
                for(T* neighbor_it : neighbor_nodes){
                    neighbor_hashes.insert(neighbor_it->get_hash());
                }
            //}
            //else {
            //    neighbor_hashes = node.neighbors;
            //}

            for(double neighbor_hash : neighbor_hashes){
                if (node_exists(neighbor_hash)) {
                    connect_nodes(node.hash, neighbor_hash);
                }
            }
        }
    }

    /**
     * Sanitize the graph by removing neighbor edges that point to nonexistent nodes.
     */
    void delete_edges_that_point_nowhere() {
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            std::list<double> to_remove;
            for (double d : it->second.neighbors) {
                if (nodes.find(d) == nodes.end()) {
                    to_remove.push_back(d);
                }
            }
            for (double d : to_remove) {
                std::cout << "Graph not sanitized!" << std::endl;
                exit(1);
                it->second.neighbors.erase(d);
            }
        }
    }

    /**
     * Get the set of solutions (nodes with is_solution() true).
     * @return A set containing the hashes of solution nodes.
     */
    std::set<double> get_solutions(){
        std::set<double> solutions;
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            double id = it->first;
            Node<T>* node = &(it->second);
            if(node->data->is_solution())
                solutions.insert(id);
        }
        return solutions;
    }

    /**
     * Mark all nodes presently in the graph as mobile.
     */
    void mobilize_all_nodes(){
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            it->second.immobile = false;
        }
    }

    /**
     * Mark all nodes presently in the graph as immobile.
     */
    void immobilize_all_nodes(){
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            it->second.immobile = true;
        }
    }

    /**
     * Remove leaf nodes from the graph.
     * @param repeat If true, repeat the removal process iteratively.
     */
    void remove_leaves(bool repeat){
        int presize = nodes.size();
        std::queue<double> leaves;
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            double id = it->first;
            Node<T>* node = &(it->second);
            if(node->neighbors.size() == 1)
                leaves.push(id);
        }
        while(!leaves.empty()){
            remove_node(leaves.front());
            leaves.pop();
        }
        if(presize != nodes.size() && repeat)
            remove_leaves(true);
    }

    /**
     * Check if a node with the given hash exists in the graph.
     * @param id The hash of the node to check.
     * @return True if the node exists, false otherwise.
     */
    bool node_exists(double id){
        return nodes.find(id) != nodes.end();
    }

    void print_hashes(){
        for(auto it = nodes.begin(); it != nodes.end(); ++it){
            it->second.data->print();
            std::cout << it->first << std::endl << std::endl << std::endl;
        }
    }

    /**
     * Remove a node from the graph.
     * @param id The hash of the node to be removed.
     */
    void remove_node(double id){
        if(!node_exists(id)) return;
        Node<T>* node = &(nodes.find(id)->second);
        EdgeSet& neighbor_nodes = node->neighbors;
        for(Edge neighbor_edge : neighbor_nodes){
            double neighbor_id = neighbor_edge.to;
            Node<T>* neighbor = &(nodes.find(neighbor_id)->second);
            neighbor->neighbors.erase(neighbor_edge);
        }
        delete node->data;
        nodes.erase(id);
    }

    /**
     * Find a neighboring node that is closer to the origin.
     * @param id The hash of the node from which to find a neighbor.
     * @return The hash of the neighbor node closer to the origin.
     */
    double approach_origin(double id){
        if(!node_exists(id)) return 0;
        Node<T>* node = &(nodes.find(id)->second);
        std::unordered_set<double> neighbor_nodes = node->neighbors;
        for(double neighbor_id : neighbor_nodes){
            Node<T>* neighbor = &(nodes.find(neighbor_id)->second);
            if(neighbor->dist == node->dist-1)
                return neighbor_id;
        }
        return 0;
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
                    Node<T>& neighbor = nodes.find(neighbor_id)->second;
                    
                    if (neighbor_id >= node->hash) continue;
                    
                    double dx = node->x - neighbor.x;
                    double dy = node->y - neighbor.y;
                    double dz = node->z - neighbor.z;
                    double dw = node->w - neighbor.w;
                    double force = attract_force;
                    double dist_sq = dx * dx + dy * dy + dz * dz + dw * dw + 1;
                    if(sqrty){
                        force *= .5*(dist_sq-1)/dist_sq;
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
