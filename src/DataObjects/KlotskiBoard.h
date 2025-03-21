#pragma once

#include <vector>
#include <set>
#include <unordered_set>
#include <string>
#include <memory>

#include "GenericBoard.h"

class KlotskiScene;  // Forward declaration

#define EMPTY_SPACE '.'

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

    void compute_letters();
    bool can_move_piece(char letter, int dx, int dy);
    KlotskiBoard move_piece(char letter, int dx, int dy);
    unordered_set<GenericBoard*> get_children();
    void get_random_move(char& rc, int& rdx, int& rdy);
};

