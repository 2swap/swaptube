#include "KlotskiBoard.h"
#include "../Scenes/KlotskiScene.h"
#include <cmath>
#include <cstdlib>

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::unordered_set;
using std::set;

bool in_bounds(int min, int val, int max) {
    return min <= val && val < max;
}

KlotskiBoard::KlotskiBoard(int w, int h, const string& rep, const bool rush)
    : GenericBoard(rep), h(h), w(w), rushhour(rush) { }

KlotskiBoard::~KlotskiBoard() { }

shared_ptr<Scene> KlotskiBoard::make_scene() const {
    return make_shared<KlotskiScene>(w, h, representation, rushhour);
}

bool KlotskiBoard::is_solution() {
    if (rushhour)
        return representation[17] == representation[16] && representation[16] != EMPTY_SPACE;
    else {
        if(h==5 && w==4)
            return representation[18]==representation[17] &&
                   representation[14]==representation[13] &&
                   representation[14]==representation[17];
        else
            return false;
    }
}

void KlotskiBoard::compute_letters() {
    if(letters.empty()) {
        for(int i = 0; i < h*w; i++) {
            letters.insert(representation[i]);
        }
        letters.erase(EMPTY_SPACE);
    }
}

double KlotskiBoard::type_specific_hash() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[y*w + x] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
}

bool KlotskiBoard::can_move_piece(char letter, int dx, int dy) {
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            if(representation[y*w + x] == letter) {
                bool inside = in_bounds(0, y+dy, h) && in_bounds(0, x+dx, w);
                if(!inside) return false;
                char target = representation[(y+dy)*w + (x+dx)];
                if(target != EMPTY_SPACE && target != letter)
                    return false;
            }
        }
    return true;
}

KlotskiBoard KlotskiBoard::move_piece(char letter, int dx, int dy) {
    string rep(h*w, EMPTY_SPACE);
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            int pos = y*w + x;
            char letter_here = representation[pos];
            if(letter_here == letter) {
                bool inside = in_bounds(0, y+dy, h) && in_bounds(0, x+dx, w);
                if(inside) {
                    int target = (y+dy)*w + (x+dx);
                    rep[target] = letter;
                }
            } else if(letter_here != EMPTY_SPACE)
                rep[pos] = letter_here;
        }

    return KlotskiBoard(w, h, rep, rushhour);
}

unordered_set<GenericBoard*> KlotskiBoard::get_children() {
    unordered_set<GenericBoard*> children;

    compute_letters();

    for (const char& letter: letters) {
        for(int dy = -1; dy <= 1; dy++)
            for(int dx = -1; dx <= 1; dx++) {
                if((dx+dy)%2 == 0) continue;
                if(rushhour && (letter - 'a' + dy)%2 == 0) continue;

                if(can_move_piece(letter, dx, dy))
                    children.insert(new KlotskiBoard(move_piece(letter, dx, dy)));
            }
    }

    return children;
}

void KlotskiBoard::get_random_move(char& rc, int& rdx, int& rdy) {
    int idx = 0;
    compute_letters();

    for(int count_dir = 0; count_dir < 2; count_dir++) {
        for (const char& letter: letters) {
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    if((dx+dy)%2 == 0) continue;
                    if(rushhour && (letter - 'a' + dy)%2 == 0) continue;

                    if(can_move_piece(letter, dx, dy)) {
                        if(count_dir == 0)
                            idx++;
                        else {
                            if(idx == 0) {
                                rc = letter;
                                rdx = dx;
                                rdy = dy;
                                return;
                            }
                            idx--;
                        }
                    }
                }
            }
        }
        idx = rand() % idx;
    }
}
