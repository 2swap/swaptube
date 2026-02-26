#pragma once

#include <vector>
#include <string>
#include "../DataObject.h"
using namespace std;

class Disc {
public:
    int x; // 0-indexed column
    double py;
    double vy = 0.0;
    double ay = -0.05;
    int index;
};

class C4Physics : public DataObject {
public:
    bool fast_mode = false;
    vector<Disc> discs;
    string queue = "";
    int w; int h;
    int moves_yet = 0;
    int time_since_last_drop = 1000;

    C4Physics(const int width, const int height);

    void add_disc_from_queue();
    void flush_queue();
    void undo(int steps);
    void undo_once();
    void inverse_gravity_up_to_index(int index);
    void remove_all_discs_above_top();
    bool all_discs_below_top() const;
    bool all_discs_accelerating_down() const;
    void append_to_queue(string s);
    void use_up_queue();
    void check_queue();

    ~C4Physics();

    void iterate_physics();
};
