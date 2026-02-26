#include "KlotskiBoard.h"
#include "../Scenes/Math/KlotskiScene.h"
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
    : GenericBoard(rep), h(h), w(w), rushhour(rush) {
    if(rep.size() != w*h) throw runtime_error("Rushhour board: invalid string length");
}

KlotskiBoard::~KlotskiBoard() { }

shared_ptr<Scene> KlotskiBoard::make_scene() const {
    return make_shared<KlotskiScene>(KlotskiBoard(*this));
}

void KlotskiBoard::print() {
    cout << endl;
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++)
            cout << representation[y*w+x];
        cout << endl;
    }
    cout << endl;
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

double KlotskiBoard::type_specific_reverse_hash_2() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[(h-1-y)*w + x] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
}

double KlotskiBoard::type_specific_reverse_hash() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[y*w + (w-1-x)] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
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

bool KlotskiBoard::can_move_piece(const KlotskiMove& km) {
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            if(representation[y*w + x] == km.piece) {
                bool inside = in_bounds(0, y+km.dy, h) && in_bounds(0, x+km.dx, w);
                if(!inside) return false;
                char target = representation[(y+km.dy)*w + (x+km.dx)];
                if(target != EMPTY_SPACE && target != km.piece)
                    return false;
            }
        }
    return true;
}

KlotskiBoard KlotskiBoard::move_piece(const KlotskiMove& km) {
    string rep(h*w, EMPTY_SPACE);
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            int pos = y*w + x;
            char letter_here = representation[pos];
            if(letter_here == km.piece) {
                bool inside = in_bounds(0, y+km.dy, h) && in_bounds(0, x+km.dx, w);
                if(inside) {
                    int target = (y+km.dy)*w + (x+km.dx);
                    rep[target] = km.piece;
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

                KlotskiMove cmove{letter, dx, dy};
                if(can_move_piece(cmove))
                    children.insert(new KlotskiBoard(move_piece(cmove)));
            }
    }

    return children;
}

void KlotskiBoard::get_random_move(KlotskiMove& km) {
    int idx = 0;
    compute_letters();

    for(int count_dir = 0; count_dir < 2; count_dir++) {
        for (const char& letter: letters) {
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    if((dx+dy)%2 == 0) continue;
                    if(rushhour && (letter - 'a' + dy)%2 == 0) continue;

                    if(can_move_piece({letter, dx, dy})) {
                        if(count_dir == 0)
                            idx++;
                        else {
                            if(idx == 0) {
                                km = {letter, dx, dy};
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

KlotskiMove KlotskiBoard::move_required_to_reach(const KlotskiBoard& kb) {
    compute_letters();
    double kb_hash = KlotskiBoard(kb).get_hash();
    for (const char& letter: letters) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if ((dx + dy) % 2 == 0)
                    continue;
                if (rushhour && (letter - 'a' + dy) % 2 == 0)
                    continue;
                KlotskiMove cmove{letter, dx, dy};
                if (can_move_piece(cmove)) {
                    KlotskiBoard newBoard = move_piece(cmove);
                    if (newBoard.get_hash() == kb_hash)
                        return cmove;
                }
            }
        }
    }
    return KlotskiMove{EMPTY_SPACE, 0, 0};
}
