#pragma once

#include <vector>
#include "../DataObject.cpp"

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

    C4Physics(const int width, const int height) : w(width), h(height), moves_yet(0) {
        mark_updated();
    }

    void add_disc_from_queue() {
        int column = queue[0] - '1';

        Disc new_disc;
        new_disc.index = moves_yet;
        new_disc.py = h;
        new_disc.x = column;
        discs.push_back(new_disc);
        mark_updated();

        moves_yet++;

        if(fast_mode) {
            // immediately settle the disc
            new_disc.py = 0;
            for (const Disc& disc : discs) {
                if (disc.x == new_disc.x && disc.index != new_disc.index) {
                    new_disc.py = max(new_disc.py, disc.py + 1);
                }
            }
            discs.back() = new_disc;
            mark_updated();
        }
    }

    void undo(int steps) {
        // Push back "x" to the queue steps times
        for (int i = 0; i < steps; i++)
            queue.push_back('x');
    }

    void undo_once() {
        moves_yet = max(0, moves_yet - 1);
        inverse_gravity_up_to_index(moves_yet);
    }

    void inverse_gravity_up_to_index(int index) {
        auto it = discs.begin();
        while (it != discs.end()) {
            Disc& disc = *it;
            if (disc.index >= index) {
                if(fast_mode)
                    it = discs.erase(it);
                else {
                    disc.ay = (disc.py + disc.x % 3 + 3) / 100;
                    ++it;
                }
            } else {
                ++it;
            }
        }
        mark_updated();
    }

    void remove_all_discs_above_top() {
        auto it = discs.begin();
        while (it != discs.end()) {
            if (it->py > h) {
                it = discs.erase(it);
                mark_updated();
            } else {
                ++it;
            }
        }
    }

    bool all_discs_below_top() const {
        for (const Disc& disc : discs) {
            if (disc.py >= h - .5) {
                return false;
            }
        }
        return true;
    }

    void append_to_queue(string s) {
        queue += s;
    }

    void check_queue() {
        if (queue.empty() || !all_discs_below_top()) {
            return;
        }
        if(queue.front() == 'x')
            undo_once();
        else
            add_disc_from_queue();
        queue.erase(0, 1);
    }

    ~C4Physics() { }

    void iterate_physics() {
        check_queue();

        const double elasticity = 0.35;
        bool any_changed = false;
        for (Disc& disc : discs) {
            double pre_y = disc.py;
            disc.vy += disc.ay;
            disc.py += disc.vy;
            if (disc.py < 0) {
                disc.vy *= -elasticity;
                disc.py = 0;
            }
            for(Disc& other_disc : discs) {
                if (&disc != &other_disc && disc.x == other_disc.x) {
                    if (disc.py < other_disc.py + 1 && disc.py > other_disc.py) {
                        disc.py = other_disc.py + 1;
                        if(abs(other_disc.vy) < 0.1) {
                            disc.vy *= -0.3;
                        } else {
                            double u1 = disc.vy;
                            double u2 = other_disc.vy;
                            double e = elasticity;
                            double v1 = ( -e * u1 + (1 + e) * u2 ) / 2;
                            double v2 = ( -e * u2 + (1 + e) * u1 ) / 2;
                            disc.vy = v1;
                            other_disc.vy = v2;
                        }
                    }
                }
            }
            if(abs(disc.py - pre_y) > 0.0001) {
                any_changed = true;
            }
        }

        remove_all_discs_above_top();

        if(any_changed) mark_updated();
    }
};
