#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <set>
#include <unordered_set>
#include <climits>
#include "../GenericBoard.cpp"
#include "Bitboard.cpp"
#include "SteadyState.h"
#include "../../misc/json.hpp"
using json = nlohmann::json;

enum C4BranchMode {
    UNION_WEAK,
    TRIM_STEADY_STATES,
    SIMPLE_WEAK,
    FULL,
    MANUAL
};
C4BranchMode c4_branch_mode = TRIM_STEADY_STATES;

string disk_col(int i){
    if(i == 1) return "\033[31mx\033[0m";  // Red "x"
    if(i == 2) return "\033[33mo\033[0m";  // Yellow "o"
    return ".";
}

class C4Board : public GenericBoard {
public:
    int BOARD_WIDTH = C4_WIDTH;
    int BOARD_HEIGHT = C4_HEIGHT;
    Bitboard red_bitboard = 0ul, yellow_bitboard = 0ul;
    string game_name = "c4";
    unordered_set<double> children_hashes;
    bool children_hashes_initialized = false;

    bool has_steady_state = false;
    shared_ptr<SteadyState> steadystate;

    //C4Board(const C4Board& other);
    C4Board(const string& rep);
    C4Board(const string& rep, shared_ptr<SteadyState> ss);
    C4Board();
    int piece_code_at(int x, int y) const;
    string reverse_representation() const;
    double reverse_hash();
    int burst() const;
    Bitboard legal_moves() const;
    int get_instant_win() const;
    vector<int> get_winning_moves() const;
    int get_blocking_move() const;
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
    void add_all_legal_children(unordered_set<GenericBoard*>& neighbors) const;
    void add_all_good_children(unordered_set<GenericBoard*>& neighbors) const;
    void add_only_child_steady_state(unordered_set<GenericBoard*>& neighbors) const;
    void insert_sorted_children_by_min_hash(vector<GenericBoard*>& children, unordered_set<GenericBoard*>& neighbors) const;
    unordered_set<GenericBoard*> get_children();
    unordered_set<double> get_children_hashes();
    json get_data() const override;
};

void fhourstones_tests(){
    cout << "fhourstones_tests" << endl;
    C4Board b("4444445623333356555216622");
    int work = -1;
    C4Result winner = b.who_is_winning(work);
    assert(work == 8);
    assert(winner == RED);
    b.play_piece(5);
    assert(b.get_best_winning_fhourstones() == 2);
    cout << "fhourstones_tests passed" << endl;
}

void construction_tests(){
    cout << "construction_tests" << endl;
    C4Board a("71");
    a.print();
    assert(a.is_reds_turn());
    cout << a.red_bitboard << endl;
    cout << a.yellow_bitboard << endl;
    assert(a.red_bitboard == 70368744177664UL);
    assert(a.yellow_bitboard == 1099511627776UL);

    C4Board b("4444445623333356555216622");
    assert(!b.is_reds_turn());
    b.print();
    int expected[C4_HEIGHT][C4_WIDTH] = {
        {0, 0, 0, 2, 0, 0, 0},
        {0, 0, 2, 1, 1, 0, 0},
        {0, 1, 1, 2, 2, 1, 0},
        {0, 2, 2, 1, 1, 2, 0},
        {0, 2, 1, 2, 1, 2, 0},
        {1, 1, 2, 1, 1, 2, 0}
    };
    for(int y = 0; y < C4_HEIGHT; y++)
        for(int x = 0; x < C4_WIDTH; x++){
            assert(((b.red_bitboard >> (y*(1+C4_WIDTH)+x)) & 1UL) == (expected[y][x] == 1));
            assert(((b.yellow_bitboard >> (y*(1+C4_WIDTH)+x)) & 1UL) == (expected[y][x] == 2));
        }
    assert(b.is_legal(1));
    assert(b.is_legal(2));
    assert(b.is_legal(3));
    assert(!b.is_legal(4));
    assert(b.is_legal(5));
    assert(b.is_legal(6));
    assert(b.is_legal(7));
}

void winner_tests(){
    cout << "winner_tests" << endl;
    list<pair<string, C4Result>> pairs;
    //verticals
    pairs.emplace_back("4141414", RED);
    pairs.emplace_back("1212121", RED);
    pairs.emplace_back("7171717", RED);
    pairs.emplace_back("777171717", RED);
    pairs.emplace_back("34141414", YELLOW);
    pairs.emplace_back("31212121", YELLOW);
    pairs.emplace_back("37171717", YELLOW);
    //horizontals
    pairs.emplace_back("44444156666222262114155112767377373337355", RED);
    pairs.emplace_back("23534224552467456663", YELLOW);
    //diagonals
    pairs.emplace_back("12344324324", RED);
    pairs.emplace_back("76544564564", RED);
    pairs.emplace_back("126552422441467575776336", YELLOW);
    //ties
    pairs.emplace_back("444444562333365666321755523756177122711172", TIE);
    pairs.emplace_back("343344444217622613116332266627117555775755", TIE);

    int n = 0;
    for (const auto& pair : pairs) {
        const string& rep = pair.first;
        C4Result winner = pair.second;

        C4Board b("");
        for (int i = 0; i < rep.size(); i++) {
            b.play_piece(rep[i] - '0'); // Convert char to int
            C4Result observed_winner = b.who_won();
            cout << b.red_bitboard << ":" << observed_winner << endl;
            assert(observed_winner == (i == rep.size()-1 ? winner : INCOMPLETE));
        }
        n++;
        cout << "Passed c4 unit test " << n << "!" << endl;
    }
}

void c4_unit_tests() {
    construction_tests();
    winner_tests();
    fhourstones_tests();
}
