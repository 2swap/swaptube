#pragma once

#include <vector>
#include <iostream>

class PieceSet {
public:
    PieceSet(const vector<int>& indices) : indices(indices) {}
    
    void print() const {
        cout << "PieceSet indices: ";
        for (int index : indices) {
            cout << index << " ";
        }
        cout << endl;
    }

    vector<int> get_indices() const {
        return indices;
    }

    // Helper function to check if a PieceSet contains an index
    bool contains(int value) const {
        for (int idx : indices) {
            if (idx == value) {
                return true;
            }
        }
        return false;
    }

    size_t size() const {
        return indices.size();
    }

private:
    const vector<int> indices;
};

PieceSet intersect(const PieceSet& ps1, const PieceSet& ps2) {
    vector<int> result;
    for (int idx : ps1.get_indices()) {
        if (ps2.contains(idx)) {
            result.push_back(idx);
        }
    }
    return PieceSet(result);
}
