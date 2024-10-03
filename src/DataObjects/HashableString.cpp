#pragma once

#include <string>
#include <unordered_map>

class HashableString : public GenericBoard {
public:
    HashableString(const string& str) : representation(str) {}

    void print() const override {
        cout << representation << endl;
    }

    bool is_solution() override {
        return false;
    }

    double board_specific_hash() const override {
        std::hash<string> hasher;
        return static_cast<double>(hasher(representation));
    }

    double board_specific_reverse_hash() const override {
        return 0; // Not needed, return 0.
    }

    unordered_set<HashableString*> get_children() {
        // Parse the lambda expression from the representation
        shared_ptr<LambdaExpression> le = parse_lambda_from_string(representation);
        
        // Get all legal reductions of the parsed lambda expression
        unordered_set<shared_ptr<LambdaExpression>> children = le->get_all_legal_reductions();
        
        // Create a set to store HashableString pointers
        unordered_set<HashableString*> result;
        
        // For each child (reduced expression) in the set of children
        for (const auto& child : children) {
            // Create a HashableString from the child's string representation and insert it into the result set
            result.insert(new HashableString(child->get_string()));
        }
        
        return result;
    }

    unordered_set<double> get_children_hashes() {
        std::hash<string> hasher;
        // Create a set to store the hashes of the children
        unordered_set<double> child_hashes;
        
        // For each child in the set of children
        for (const auto& child : get_children()) {
            // Hash the string representation of the child and insert it into the hash set
            child_hashes.insert(hasher(child->representation));
        }
        
        return child_hashes;
    }

    string representation;
};
