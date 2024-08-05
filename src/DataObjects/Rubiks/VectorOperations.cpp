#pragma once

#include <vector>

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
