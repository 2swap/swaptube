#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <set>

class GenericBoard {
public:
    GenericBoard(const string& s) : representation(s) {}
    virtual unordered_set<GenericBoard*> get_children() const = 0;
    unordered_set<double> get_children_hashes() const {
        unordered_set<GenericBoard*> kids = get_children();
        unordered_set<double> ret;
        for(GenericBoard* kid : kids){
            ret.insert(kid->get_hash());
            delete kid;
        }
        return ret;
    }

    double get_hash() {
        if(hash == 0)
            hash = type_specific_hash();
        return hash;
    }
    string representation;

    int get_highlight_type() {
        if(highlight_type == -1)
            highlight_type = is_solution() ? 1 : 0;
        return highlight_type;
    }

private:
    virtual bool is_solution() = 0;
    virtual double type_specific_hash() const = 0;
    double hash = 0;
    int highlight_type = -1;
};
