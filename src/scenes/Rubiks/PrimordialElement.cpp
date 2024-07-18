#pragma once

#include "PermutationElement.cpp"
#include <unordered_set>
#include <algorithm>

class PrimordialElement : public PermutationElement {
public:
    PrimordialElement(const string& name, const vector<int>& effect) : PermutationElement(name, effect, 1) {
        if (!string_is_valid_name(name)) {
            cout << "ERROR: Primordial element names must be uppercase letters and apostrophes only." << endl;
            exit(1);
        }
        if (!is_valid_permutation(effect)) {
            cout << "ERROR: Invalid permutation effect." << endl;
            exit(1);
        }
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

    static inline bool invalid_char(char c) {
        return !(c == '\'' || isupper(c));
    }

    bool string_is_valid_name(const string &str) {
        for(int i = 0; i < str.size(); i++){
            if(invalid_char(str[i]))
                return false;
        }
        return true;
    }
};
