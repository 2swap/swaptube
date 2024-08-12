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
    C4Result play_one_game(const string& boardString, string& defeat, const string& prior_defeat, bool verbose) const;
    void print() const;
    char steadystate[C4_HEIGHT][C4_WIDTH];


    void write_to_file(const string& filename) const {
        ofstream file(filename);
        if (file.is_open()) {
            for (int row = 0; row < C4_HEIGHT; ++row) {
                for (int col = 0; col < C4_WIDTH; ++col) {
                    file << steadystate[row][col];
                }
                file << endl;
            }
        }
    }

    void read_from_file(const string& filename) {
        print();
        ifstream file(filename);
        if (file.is_open()) {
            for (int row = 0; row < C4_HEIGHT; ++row) {
                string line;
                if (getline(file, line)) { // Read the entire line as a string
                    // Check if the line length matches the expected width
                    if (line.length() == static_cast<size_t>(C4_WIDTH)) {
                        for (int col = 0; col < C4_WIDTH; ++col) {
                            char c = line[col];
                            steadystate[row][col] = c; // Assign characters to the array
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
