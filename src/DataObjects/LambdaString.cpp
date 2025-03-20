#pragma once

#include <string>
#include <unordered_map>
#include "HashableString.cpp"

class LambdaString : public HashableString {
public:
    LambdaString(const string& str) : representation(str) {}

    bool is_solution() override {
        shared_ptr<LambdaExpression> le = parse_lambda_from_string(representation);
        return le->is_reducible();
    }

    unordered_set<GenericBoard*> get_children() {
        shared_ptr<LambdaExpression> le = parse_lambda_from_string(representation);
        unordered_set<shared_ptr<LambdaExpression>> children = le->get_all_legal_reductions();
        unordered_set<LambdaString*> result;
        for (const auto& child : children) {
            result.insert(new LambdaString(child->get_string()));
        }
        return result;
    }

    string representation;
};
