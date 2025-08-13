#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <set>
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Scene.cpp"
#include "../misc/json.hpp"
using json = nlohmann::json;

class GenericBoard {
public:
    GenericBoard(const string& s) : representation(s) {}
    GenericBoard() : representation("") {}
    virtual unordered_set<GenericBoard*> get_children() = 0;
    unordered_set<double> get_children_hashes() {
        unordered_set<GenericBoard*> kids = get_children();
        unordered_set<double> ret;
        for(GenericBoard* kid : kids){
            ret.insert(kid->get_hash());
            delete kid;
        }
        return ret;
    }

    void reset_hashes() { hash = 0; reverse_hash = 0; reverse_hash_2 = 0; }
    double get_hash() {
        if(hash == 0)
            hash = type_specific_hash();
        return hash;
    }
    double get_reverse_hash() {
        if(reverse_hash == 0)
            reverse_hash = type_specific_reverse_hash();
        return reverse_hash;
    }
    double get_reverse_hash_2() {
        if(reverse_hash_2 == 0)
            reverse_hash_2 = type_specific_reverse_hash_2();
        return reverse_hash_2;
    }
    string representation;

    int get_highlight_type() {
        if(highlight_type == -1)
            highlight_type = is_solution() ? 1 : 0;
        return highlight_type;
    }

    virtual json get_data() const {
        return json();
    }

    virtual shared_ptr<Scene> make_scene() const {
        return make_shared<LatexScene>(representation, 1);
    }

private:
    virtual bool is_solution() = 0;

    // Which mirror side this node is on, if applicable.
    virtual int which_side() const { return 0; }

    virtual double type_specific_hash() = 0;
    virtual double type_specific_reverse_hash() {return -3.1415;}
    virtual double type_specific_reverse_hash_2() {return -3.1415;}
    double hash = 0;
    double reverse_hash = 0;
    double reverse_hash_2 = 0;
    int highlight_type = -1;
};
