#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include "PieceSet.cpp"
#include "VectorOperations.cpp"

class PermutationElement {
public:
    virtual ~PermutationElement() = default;

    string get_name() const {
        return name;
    }
    vector<int> get_effect() const {
        return effect;
    }
    int get_primordial_size() const {
        return primordial_size;
    }
    int get_impact() const {
        return impact;
    }
    int get_yuckiness() const {
        return impact * primordial_size;
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

    void print() const {
        cout << setw(50) << left << name 
             << " impacts " << setw(2) << right << impact
             << " pieces and has primordial_size " << setw(5) << right << primordial_size << "." << endl;
    }

    static vector<int> identity(size_t size) {
        vector<int> id(size);
        for (size_t i = 0; i < size; ++i) {
            id[i] = i;
        }
        return id;
    }

protected:
    const int primordial_size;
    const vector<int> effect;
    const string name;
    const int impact;

    // Constructor initializes const members
    PermutationElement(const string& name, const vector<int>& effect, int primordial_size)
        : name(name), effect(effect), primordial_size(primordial_size), impact(PieceSet(effect).get_size()) {}
};
