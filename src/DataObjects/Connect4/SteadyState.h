#pragma once

#include <unordered_set>
#include <list>
#include <array>
#include <algorithm>
#include <fstream>
#include <vector>
#include "Bitboard.h"
#include "C4Board.h"
#include "JsonC4Cache.h"

using namespace std;

inline bool is_miai(char c){
    return c == '@' || c == '#';
}

class SteadyState {
public:
    SteadyState(const array<string, C4_HEIGHT>& chars);
    int query_steady_state(const C4Board& board) const;
    void mutate();
    void drop(const int x, const char c);
    C4Result play_one_game(const C4Board& board) const;
    void print() const;
    void clear();
    Bitboard bitboard_yellow = 0ul;
    Bitboard bitboard_red = 0ul;
    Bitboard bitboard_miai = 0ul;
    Bitboard bitboard_claimeven = 0ul;
    Bitboard bitboard_claimodd = 0ul;
    Bitboard bitboard_plus = 0ul;
    Bitboard bitboard_equal = 0ul;
    Bitboard bitboard_minus = 0ul;
    Bitboard bitboard_urgent = 0ul;
    Bitboard bitboard_then = 0ul;
    Bitboard bitboard_if = 0ul;
    void populate_char_array(const array<string, C4_HEIGHT>& source);
    bool validate(C4Board b, bool verbose = false);
    bool validate_recursive_call(C4Board b, unordered_set<double>& wins_cache, bool verbose = false);
    char get_char(const int x, const int y) const;
    void set_char(const int x, const int y, const char c);
    void set_char_bitboard(const Bitboard point, const char c);
    char get_char_from_char_array(const int x, const int y) const;
    char get_char_from_bitboards(const int x, const int y) const;
    bool check_ss_matches_board(C4Board b);
    bool check_matches_board(const C4Board& b) const;
    bool check_no_illegal_characters() const;

    string to_string() const {
        string s(C4_WIDTH*C4_HEIGHT, ' ');
        for (int y = 0; y < C4_HEIGHT; ++y)
            for (int x = 0; x < C4_WIDTH; ++x)
                s[x+y*C4_WIDTH] = get_char(x, y);
        return s;
    }
};

SteadyState create_empty_steady_state(const C4Board& b);
SteadyState create_random_steady_state(const C4Board& b);
SteadyState make_steady_state_from_string(const string& input);
shared_ptr<SteadyState> find_cached_steady_state(C4Board b);
shared_ptr<SteadyState> modify_child_suggestion(const shared_ptr<SteadyState> parent, const C4Board& b);
shared_ptr<SteadyState> find_steady_state(const string& representation, const shared_ptr<SteadyState> suggestion, bool verbose = false, bool read_from_cache = true, int pool = 40, int generations = 50);
SteadyState read_from_file(const string& filename, bool read_reverse);
string reverse_ss(const std::string& input);
