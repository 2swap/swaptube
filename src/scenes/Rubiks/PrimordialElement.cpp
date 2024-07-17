#pragma once

#include "PermutationElement.cpp"
#include <unordered_set>
#include <algorithm>

class PrimordialElement : public PermutationElement {
public:
    PrimordialElement(const string& name, const vector<int>& effect) : PermutationElement(name, effect) {
        if (name.size() != 1 || !isupper(name[0])) {
            cout << "ERROR: Primordial element names must be single uppercase letters." << endl;
            exit(1);
        }
        if (!is_valid_permutation(effect)) {
            cout << "ERROR: Invalid permutation effect." << endl;
            exit(1);
        }
    }

    void print() const override {
        cout << "PrimordialElement '" << name << "' modifies " + to_string(get_modified_set().size()) + " pieces." << endl;
    }

private:
    bool is_valid_permutation(const vector<int>& effect) {
        unordered_set<int> elements;
        for (int num : effect) {
            if (num < 0 || num >= effect.size() || !elements.insert(num).second) {
                return false;
            }
        }
        return elements.size() == effect.size();
    }
};
