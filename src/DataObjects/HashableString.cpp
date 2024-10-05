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
        shared_ptr<LambdaExpression> le = parse_lambda_from_string(representation);
        unordered_set<shared_ptr<LambdaExpression>> children = le->get_all_legal_reductions();
        unordered_set<HashableString*> result;
        for (const auto& child : children) {
            result.insert(new HashableString(child->get_string()));
        }
        return result;
    }

    unordered_set<double> get_children_hashes() {
        std::hash<string> hasher;
        unordered_set<double> child_hashes;
        shared_ptr<LambdaExpression> le = parse_lambda_from_string(representation);
        unordered_set<shared_ptr<LambdaExpression>> children = le->get_all_legal_reductions();
        for (const auto& child : children) {
            child_hashes.insert(hasher(child->get_string()));
        }
        return child_hashes;
    }

    string representation;
};
