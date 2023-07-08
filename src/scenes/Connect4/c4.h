#pragma once
#include <iostream>
#include <string>
#include <cmath>

#define WIDTH 7
#define HEIGHT 6

using namespace std;

class Board{
public:
    Board(string rep) : representation(rep) {
        clear();
        play(rep);
    };
    Board(string rep, string a, string h) : representation(rep) {
        clear();
        play(rep);
        if(a.size() == WIDTH*HEIGHT)annotations = a;
        if(h.size() == WIDTH*HEIGHT)highlight = h;
    };
    int col=1; // whose move is it
    int grid[HEIGHT][WIDTH];
    bool blink[HEIGHT][WIDTH];
    string annotations = "                                          ";
    string highlight = "                                          ";
    string representation;
    void clear(){
        for(int x = 0; x < WIDTH; x++)
            for(int y = 0; y < HEIGHT; y++){
                grid[y][x] = 0;
                blink[y][x] = false;
            }
    }
    char get_annotation(int x, int y){
        return annotations[x+(HEIGHT-1-y)*WIDTH];
    }
    char get_highlight(int x, int y){
        return highlight[x+(HEIGHT-1-y)*WIDTH];
    }

private:
    void play(string s){
        for(char c : s){
            int x = c-'1';
            for(int y = 0; y < HEIGHT; y++){
                if(grid[y][x] == 0){
                    grid[y][x] = col;
                    break;
                }
            }
            col=col==1?2:1;
        }
        detect_win();
    }
    void detect_win(){
        for(int x = 0; x < WIDTH; x++)
            for(int y = 0; y < HEIGHT; y++){
                if(grid[y][x] == 0) continue;
                if(x<4 && y<3)detect_win_directional(x, y, 1, 1);
                if(x<4)detect_win_directional(x, y, 1, 0);
                if(y<3)detect_win_directional(x, y, 0, 1);
                if(x>=3 && y<3)detect_win_directional(x, y, -1, 1);
            }
    }
    void detect_win_directional(int x, int y, int dx, int dy){
        int col_here = grid[y][x];
        int count = 0;
        for(int c = 0; c < 4; c++)
            if(grid[y+dy*c][x+dx*c] != col_here) break;
            else count++;
        if(count >= 4){
            for(int c = 0; c < 4; c++)
                if(grid[y+dy*c][x+dx*c] != col_here) break;
                else blink[y+dy*c][x+dx*c] = true;
        }
    }
};

string shared(const string& str1, const string& str2) {
    string result;

    if (str1.length() != str2.length()) {
        return result; // Return an empty string if the lengths are different
    }

    // Iterate over each character in the strings
    for (size_t i = 0; i < str1.length(); ++i) {
        // Check if the characters are the same
        if (str1[i] == str2[i]) {
            result += str1[i]; // Add the character to the result string
        } else {
            result += ' '; // Add a space if the characters are different
        }
    }

    return result;
}

string replerp(const string& b1, const string& b2, double w) {
    if (b1.find(b2) == 0 || b2.find(b1) == 0) {
        // One string begins with the other
        int range = b2.size() - b1.size();
        return (b1.size() > b2.size() ? b1 : b2).substr(0, round(b1.size() + w * range));
    } else {
        // Neither string begins with the other
        int common_prefix_len = 0;
        while (b1[common_prefix_len] == b2[common_prefix_len]) {
            common_prefix_len++;
        }

        string common_prefix = b1.substr(0, common_prefix_len);
        if(w<0.5) return replerp(b1, common_prefix, w*2);
        else return replerp(common_prefix, b2, (w-.5)*2);
    }
}

Board c4lerp(Board b1, Board b2, double w){
    string representation = replerp(b1.representation, b2.representation, smoother2(w));
    string annotations = shared(b1.annotations, b2.annotations);
    Board transition(representation, annotations);
    return transition;
}

// Unit test for replerp function
void replerp_ut() {
    std::string b1 = "12345";
    std::string b2 = "1";
    bool pass = true;
    pass &= replerp(b1, b2, 0.) == "12345";
    pass &= replerp(b1, b2, 0.25) == "1234";
    pass &= replerp(b1, b2, 0.5) == "123";
    pass &= replerp(b1, b2, 0.75) == "12";
    pass &= replerp(b1, b2, 1.) == "1";

    pass &= replerp(b2, b1, 0.) == "1";
    pass &= replerp(b2, b1, 0.25) == "12";
    pass &= replerp(b2, b1, 0.5) == "123";
    pass &= replerp(b2, b1, 0.75) == "1234";
    pass &= replerp(b2, b1, 1.) == "12345";
    if (pass) {
        std::cout << "replerp_ut - Case 1: Passed." << std::endl;
    } else {
        std::cout << "replerp_ut - Case 1: Failed." << std::endl;
        exit(1);
    }

    std::string b3 = "abc";
    std::string b4 = "de";
    pass = true;
    pass &= replerp(b3, b4, 0.) == "abc";
    pass &= replerp(b3, b4, 0.2) == "ab";
    pass &= replerp(b3, b4, 0.4) == "a";
    pass &= replerp(b3, b4, 0.6) == "";
    pass &= replerp(b3, b4, 0.8) == "d";
    pass &= replerp(b3, b4, 1.) == "de";

    pass &= replerp(b4, b3, 0.) == "de";
    pass &= replerp(b4, b3, 0.2) == "d";
    pass &= replerp(b4, b3, 0.4) == "";
    pass &= replerp(b4, b3, 0.6) == "a";
    pass &= replerp(b4, b3, 0.8) == "ab";
    pass &= replerp(b4, b3, 1.) == "abc";
    if (pass) {
        std::cout << "replerp_ut - Case 2: Passed." << std::endl;
    } else {
        std::cout << "replerp_ut - Case 2: Failed." << std::endl;
        exit(1);
    }
}

void run_c4_unit_tests(){
    replerp_ut();
}