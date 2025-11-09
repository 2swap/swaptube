#pragma once
#include <fstream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <string>

#include "KlotskiBoard.h"

// Utility to print hashes safely (avoid precision loss).
// If your hashes are actually integers, switch to uint64_t and print as integers.
static inline std::string hash_to_str(double h) {
    std::ostringstream oss;
    oss << std::setprecision(17) << h;
    return oss.str();
}

// Core BFS that streams an edge list: "src_hash dst_hash\n" per line
inline void export_edgelist_from_klotski(
    const KlotskiBoard& start,
    const std::string& edge_path,
    const std::string& node_path = ""  // optional: write node list "hash\n"
){
    std::ofstream eout(edge_path);
    if(!eout) throw std::runtime_error("Failed to open edge output: " + edge_path);

    std::ofstream nout;
    const bool write_nodes = !node_path.empty();
    if(write_nodes){
        nout.open(node_path);
        if(!nout) throw std::runtime_error("Failed to open node output: " + node_path);
    }

    std::queue<KlotskiBoard> q;
    std::unordered_set<double> seen;

    q.push(start);
    seen.insert(start.get_hash());
    if(write_nodes) nout << hash_to_str(start.get_hash()) << "\n";

    // We assume KlotskiBoard::get_children() returns newly allocated GenericBoard*
    // that we must delete after use.
    while(!q.empty()){
        KlotskiBoard cur = q.front(); q.pop();
        const double cur_h = cur.get_hash();

        auto kids = cur.get_children(); // unordered_set<GenericBoard*>
        for (auto* gptr : kids){
            // Downcast to KlotskiBoard (this is safe if your implementation returns the same type)
            auto* kptr = dynamic_cast<KlotskiBoard*>(gptr);
            if(!kptr){
                // Fallback: if it's not a KlotskiBoard, still try to get hash and skip enqueue
                const double ch = gptr->get_hash();
                eout << hash_to_str(cur_h) << " " << hash_to_str(ch) << "\n";
                delete gptr;
                continue;
            }

            const double ch = kptr->get_hash();
            // Write edge
            eout << hash_to_str(cur_h) << " " << hash_to_str(ch) << "\n";

            // Enqueue unseen child
            if(seen.insert(ch).second){
                if(write_nodes) nout << hash_to_str(ch) << "\n";
                q.push(*kptr);
            }
            delete kptr;
        }
    }
}

// Optional: GraphML export (directed graph)
inline void export_graphml_from_klotski(
    const KlotskiBoard& start,
    const std::string& graphml_path
){
    std::ofstream out(graphml_path);
    if(!out) throw std::runtime_error("Failed to open GraphML output: " + graphml_path);

    out << R"(<?xml version="1.0" encoding="UTF-8"?>)"
        << "\n<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">"
        << "\n  <graph edgedefault=\"directed\">";

    std::queue<KlotskiBoard> q;
    std::unordered_set<double> seen;
    std::vector<std::pair<double,double>> edges;

    q.push(start);
    seen.insert(start.get_hash());

    // Collect nodes+edges first so we can write <node> before <edge>
    while(!q.empty()){
        KlotskiBoard cur = q.front(); q.pop();
        double ch = cur.get_hash();

        auto kids = cur.get_children();
        for (auto* gptr : kids){
            auto* kptr = dynamic_cast<KlotskiBoard*>(gptr);
            double kh = kptr ? kptr->get_hash() : gptr->get_hash();
            edges.emplace_back(ch, kh);
            if(seen.insert(kh).second && kptr){
                q.push(*kptr);
            }
            delete gptr;
        }
    }

    // Write nodes
    for (double h : seen){
        out << "\n    <node id=\"n" << hash_to_str(h) << "\"/>";
    }
    // Write edges
    for (auto& e : edges){
        out << "\n    <edge source=\"n" << hash_to_str(e.first)
            << "\" target=\"n" << hash_to_str(e.second) << "\"/>";
    }

    out << "\n  </graph>\n</graphml>\n";
}

// Optional: DOT export (Graphviz)
inline void export_dot_from_klotski(
    const KlotskiBoard& start,
    const std::string& dot_path
){
    std::ofstream out(dot_path);
    if(!out) throw std::runtime_error("Failed to open DOT output: " + dot_path);

    out << "digraph G {\n";

    std::queue<KlotskiBoard> q;
    std::unordered_set<double> seen;

    q.push(start);
    seen.insert(start.get_hash());

    while(!q.empty()){
        KlotskiBoard cur = q.front(); q.pop();
        double ch = cur.get_hash();
        auto kids = cur.get_children();
        for (auto* gptr : kids){
            auto* kptr = dynamic_cast<KlotskiBoard*>(gptr);
            double kh = kptr ? kptr->get_hash() : gptr->get_hash();
            out << "  \"n" << hash_to_str(ch) << "\" -> \"n" << hash_to_str(kh) << "\";\n";
            if(seen.insert(kh).second && kptr){
                q.push(*kptr);
            }
            delete gptr;
        }
    }

    out << "}\n";
}
