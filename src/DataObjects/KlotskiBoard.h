#pragma once

#include <vector>
#include <set>
#include <unordered_set>
#include <string>
#include <memory>

#include "GenericBoard.cpp"

class KlotskiScene;  // Forward declaration

#define EMPTY_SPACE '.'

struct KlotskiMove {
    char piece = '.';
    int dx = 0;
    int dy = 0;
};

bool in_bounds(int min, int val, int max);

class KlotskiBoard : public GenericBoard {
public:
    int h, w;
    unordered_set<char> letters;
    string blurb;
    bool rushhour = true;
    string game_name = "klotski";

    KlotskiBoard(int w, int h, const string& rep, const bool rush);
    ~KlotskiBoard();

    shared_ptr<Scene> make_scene() const override;
    bool is_solution() override;
    double type_specific_hash() override;
    double type_specific_reverse_hash();

    void print();
    void compute_letters();
    bool can_move_piece(const KlotskiMove& km);
    KlotskiBoard move_piece(const KlotskiMove& km);
    unordered_set<GenericBoard*> get_children();
    void get_random_move(KlotskiMove& km);
    KlotskiMove move_required_to_reach(const KlotskiBoard& kb);
};

