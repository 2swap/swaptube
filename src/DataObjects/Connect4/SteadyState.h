#pragma once

class C4Board;

#include <unordered_set>
#include <list>
#include <array>

const int C4_HEIGHT = 6;
const int C4_WIDTH = 7;

enum C4Result {
    TIE,
    RED,
    YELLOW,
    INCOMPLETE
};

vector<char> miai = {'@', '#'};
vector<char> priority_list = {'+', '=', '-'};
vector<char> claims = {' ', '|'};
vector<char> disks = {'1', '2'};

bool is_miai(char c){
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
    void populate_char_array(const array<string, C4_HEIGHT>& source);
    bool validate_steady_state(const C4Board& b, int& branches_searched);
    char get_char(const int x, const int y) const;
    void set_char(const int x, const int y, const char c);
    char get_char_from_char_array(const int x, const int y) const;
    char get_char_from_bitboards(const int x, const int y) const;


    void write_to_file(const string& filename) const {
        ofstream file(filename);
        if (file.is_open()) {
            for (int y = 0; y < C4_HEIGHT; ++y) {
                for (int x = 0; x < C4_WIDTH; ++x) {
                    file << get_char(x, y);
                }
                file << endl;
            }
        }
    }

};

SteadyState read_from_file(const string& filename) {
    array<string, C4_HEIGHT> chars;

    ifstream file(filename);
    if (file.is_open()) {
        for (int y = 0; y < C4_HEIGHT; ++y) {
            string line;
            if (getline(file, line)) { // Read the entire line as a string
                // Check if the line length matches the expected width
                if (line.length() == static_cast<size_t>(C4_WIDTH)) {
                    chars[y] = line;
                } else failout("Invalid line length in the file.");
            } else failout("STEADYSTATE CACHE READ ERROR");
        }
    } else failout("Failed to read SteadyState file.");
    return SteadyState(chars);
}
