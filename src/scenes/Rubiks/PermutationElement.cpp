#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "PieceSet.cpp"

// Define composition operator
vector<int> operator+(const vector<int>& a, const vector<int>& b) {
    vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[b[i]];
    }
    return result;
}

// Define inversion operator
vector<int> operator~(const vector<int>& a) {
    vector<int> inverse(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        inverse[a[i]] = i;
    }
    return inverse;
}

class PermutationElement {
public:
    virtual ~PermutationElement() = default;
    virtual void print() const = 0;
    string get_name() const {
        return name;
    }
    vector<int> get_effect() const {
        return effect;
    }

    PieceSet get_modified_set() const {
        vector<int> modified_indices;
        for (size_t i = 0; i < effect.size(); ++i) {
            if (effect[i] != i) {
                modified_indices.push_back(i);
            }
        }
        return PieceSet(modified_indices);
    }

    int order() const {
        vector<int> current_effect = effect;
        int count = 1;
        while (current_effect != identity(effect.size())) {
            current_effect = current_effect + effect;
            count++;
        }
        return count;
    }

protected:
    const vector<int> effect;
    const string name;

    // Constructor initializes const member
    PermutationElement(const string& name, const vector<int>& effect)
        : name(name), effect(effect) {}

private:
    static vector<int> identity(size_t size) {
        vector<int> id(size);
        for (size_t i = 0; i < size; ++i) {
            id[i] = i;
        }
        return id;
    }
};
