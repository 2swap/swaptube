#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <set>
#include <unordered_set>
#include <memory>
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Scene.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using std::string;

class GenericBoard {
public:
    GenericBoard(const string& s);
    GenericBoard();
    virtual ~GenericBoard() = default;
    virtual std::unordered_set<GenericBoard*> get_children() = 0;
    std::unordered_set<double> get_children_hashes();

    void reset_hashes();
    double get_hash();
    double get_reverse_hash();
    double get_reverse_hash_2();
    string representation;

    int get_highlight_type();

    virtual json get_data() const;

    virtual std::shared_ptr<Scene> make_scene() const;

    // Which mirror side this node is on, if applicable.
    virtual int which_side() const;

private:
    virtual bool is_solution() = 0;

    virtual double type_specific_hash() = 0;
    virtual double type_specific_reverse_hash();
    virtual double type_specific_reverse_hash_2();
    double hash;
    double reverse_hash;
    double reverse_hash_2;
    int highlight_type;
};
