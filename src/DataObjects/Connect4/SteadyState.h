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
    SteadyState();
    SteadyState(const array<string, C4_HEIGHT>& chars);
    int query_steady_state(const C4Board board) const;
    void mutate();
    void drop(int x, char c);
    C4Result play_one_game(const string& boardString) const;
    void print() const;
    int bitboard_yellow = 0;
    int bitboard_red = 0;
    int bitboard_miai = 0;
    int bitboard_claimeven = 0;
    int bitboard_claimodd = 0;
    int bitboard_plus = 0;
    int bitboard_equal = 0;
    int bitboard_minus = 0;
    char steadystate[C4_HEIGHT][C4_WIDTH];
    bool validate_steady_state(const C4Board& b, int& branches_searched);
    char get_char(int x, int y) const;
    void set_char(int x, int y, char c);


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

    void read_from_file(const string& filename) {
        print();
        ifstream file(filename);
        if (file.is_open()) {
            for (int y = 0; y < C4_HEIGHT; ++y) {
                string line;
                if (getline(file, line)) { // Read the entire line as a string
                    // Check if the line length matches the expected width
                    if (line.length() == static_cast<size_t>(C4_WIDTH)) {
                        for (int x = 0; x < C4_WIDTH; ++x) {
                            set_char(x, y, line[x]); // Assign characters to the array
                        }
                    } else {
                        cout << "Invalid line length in the file." << endl;
                        exit(1);
                    }
                } else {
                    cout << "STEADYSTATE CACHE READ ERROR" << endl;
                    exit(1);
                }
            }
        }
    }
};
