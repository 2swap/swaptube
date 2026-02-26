#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <set>
#include <unordered_set>
#include <climits>
#include "../GenericBoard.h"
#include "Bitboard.h"
#include "../Graph.h"
class SteadyState; // forward declaration to avoid circular dependency
#include <nlohmann/json.hpp>
using json = nlohmann::json;

const int C4_HEIGHT = 6;
const int C4_WIDTH = 7;

extern shared_ptr<Graph> graph_to_check_if_points_are_in;

enum C4BranchMode {
    UNION_WEAK,
    TRIM_STEADY_STATES,
    SIMPLE_WEAK,
    RANDOM_WEAK,
    LEFTMOST_LOWEST_2,
    LEFTMOST_WEAK,
    RIGHTMOST_WEAK,
    FULL,
    MANUAL
};

enum C4Result {
    TIE,
    RED,
    YELLOW,
    INCOMPLETE
};

inline string disk_col(int i){
    if(i == 1) return "\033[31mx\033[0m";  // Red "x"
    if(i == 2) return "\033[33mo\033[0m";  // Yellow "o"
    return ".";
}

class C4Board : public GenericBoard {
public:
    C4BranchMode c4_branch_mode;
    int BOARD_WIDTH = C4_WIDTH;
    int BOARD_HEIGHT = C4_HEIGHT;
    Bitboard red_bitboard = 0ul, yellow_bitboard = 0ul;
    string game_name = "c4";
    unordered_set<double> children_hashes;
    bool children_hashes_initialized = false;

    bool has_steady_state = false;
    shared_ptr<SteadyState> steadystate;

    //C4Board(const C4Board& other);
    C4Board(const C4BranchMode mode);
    C4Board(const C4BranchMode mode, const string& rep);
    C4Board(const C4BranchMode mode, const string& rep, shared_ptr<SteadyState> ss);
    int piece_code_at(int x, int y) const;
    string reverse_representation() const;
    double reverse_hash();
    int burst() const;
    Bitboard legal_moves() const;
    int get_instant_win() const;
    vector<int> get_winning_moves() const;
    int get_blocking_move() const;
    C4Board get_mirror_board() const;
    void print();
    int which_side() const override;
    int random_legal_move() const;
    bool is_legal(int x) const;
    bool search_for_steady_states(bool verbose) const;
    int search_nply(const int depth, int& num_ordered_unfound, bool verbose, vector<int> ordering_in, vector<int>& ordering_out) const;
    int search_nply_id(const int depth, const vector<int>& order_in, vector<int>& order_out) const;
    C4Result who_won() const;
    bool is_reds_turn() const;
    bool is_solution() override;
    double type_specific_hash() override;
    double type_specific_reverse_hash() override;
    void fill_board_from_string(const string& rep);
    C4Board* remove_piece();
    void play_piece(int piece);
    C4Board child(int piece) const;
    C4Result who_is_winning(int& work, bool verbose = false);
    int get_human_winning_fhourstones();
    int get_best_winning_fhourstones() const;
    void add_best_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const;
    void add_all_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const;
    void add_random_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const;
    void add_rightmost_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const;
    void add_leftmost_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const;
    void add_all_legal_children(unordered_set<GenericBoard*>& neighbors) const;
    void add_all_good_children(unordered_set<GenericBoard*>& neighbors) const;
    void add_only_child_steady_state(unordered_set<GenericBoard*>& neighbors) const;
    void insert_sorted_children_by_min_hash(vector<GenericBoard*>& children, unordered_set<GenericBoard*>& neighbors) const;
    unordered_set<GenericBoard*> get_children();
    unordered_set<double> get_children_hashes();
    json get_data() const override;
    Bitboard winning_discs() const;
};
