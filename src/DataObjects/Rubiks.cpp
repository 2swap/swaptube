#include "Rubiks.h"
#include <unordered_map>
#include "../Host_Device_Shared/vec.h"



// For now, a face is a matrix, and a cube is a vector of 6 faces, maybe in the future, I'll rewrite with either a coordinate
//system for each piece, or even something like permutations, this could be usefull for some of the graph views we planned


// a face is a square matrice, size is "size", with the same char everywhere, each char corresponds to a color.


Cut::Cut(const vec3& axis, float dist) : axis(axis), dist(dist) {};
Cut::Cut() : axis(vec3(0, 0, 0)), dist(0) {}



void Rubiks::tick(const StateReturn& state) {}





// Here I had to define each face rotation individually, 
// this is due to the fact that matrices defining faces have different orientations


// initially, depth would only turn the said depth, but 2swap asked that depth also turns the layers in between, so now depth is the maximum depth to turn, and all layers in between are turned as well.
void Rubiks::rotateU(int depth){
    int n = pattern.pattern[0].size();
    pattern.rotateFaceClockwise(0);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[2][d];
        pattern.pattern[2][d] = pattern.pattern[5][d];
        pattern.pattern[3][d] = pattern.pattern[4][d];
        pattern.pattern[5][d] = pattern.pattern[3][d];
        pattern.pattern[4][d] = temp;
    }
}


void Rubiks::rotateD(int depth) {
    int n = pattern.pattern[1].size();
    pattern.rotateFaceClockwise(1);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[2][n - 1 - d];
        pattern.pattern[2][n - 1 - d] = pattern.pattern[4][n - 1 - d];
        pattern.pattern[4][n - 1 - d] = pattern.pattern[3][n - 1 - d];
        pattern.pattern[3][n - 1 - d] = pattern.pattern[5][n - 1 - d];
        pattern.pattern[5][n - 1 - d] = temp;
    }
}

void Rubiks::rotateF(int depth) {
    int n = pattern.pattern[2].size();
    pattern.rotateFaceClockwise(2);

    pattern.transposeFace(4);
    pattern.transposeFace(5);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[0][n - 1 - d];
        std::reverse(pattern.pattern[4][n - 1 - d].begin(), pattern.pattern[4][n - 1 - d].end());
        std::reverse(pattern.pattern[5][d].begin(), pattern.pattern[5][d].end());
        
        pattern.pattern[0][n - 1 - d] = pattern.pattern[4][n - 1 - d];
        pattern.pattern[4][n - 1 - d] = pattern.pattern[1][d];
        pattern.pattern[1][d] = pattern.pattern[5][d];
        pattern.pattern[5][d] = temp;
        
        std::reverse(pattern.pattern[4][n - 1 - d].begin(), pattern.pattern[4][n - 1 - d].end());
        std::reverse(pattern.pattern[5][d].begin(), pattern.pattern[5][d].end());
    }

    pattern.transposeFace(4);
    pattern.transposeFace(5);
}

void Rubiks::rotateB(int depth) {
    int n = pattern.pattern[3].size();
    pattern.rotateFaceClockwise(3);

    pattern.transposeFace(4);
    pattern.transposeFace(5);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[0][d];
        std::reverse(pattern.pattern[4][d].begin(), pattern.pattern[4][d].end());
        std::reverse(pattern.pattern[5][n - 1 - d].begin(), pattern.pattern[5][n - 1 - d].end());
        
        pattern.pattern[0][d] = pattern.pattern[5][n - 1 - d];
        pattern.pattern[5][n - 1 - d] = pattern.pattern[1][n - 1 - d];
        pattern.pattern[1][n - 1 - d] = pattern.pattern[4][d];
        pattern.pattern[4][d] = temp;

        std::reverse(pattern.pattern[4][d].begin(), pattern.pattern[4][d].end());
        std::reverse(pattern.pattern[5][n - 1 - d].begin(), pattern.pattern[5][n - 1 - d].end());
    }

    pattern.transposeFace(4);
    pattern.transposeFace(5);
}

void Rubiks::rotateR(int depth) {
    int n = pattern.pattern[5].size();
    pattern.rotateFaceClockwise(5);

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(1);
    pattern.transposeFace(3);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[0][n - 1 - d];
        std::reverse(pattern.pattern[3][d].begin(), pattern.pattern[3][d].end());
        std::reverse(temp.begin(), temp.end());
        
        pattern.pattern[0][n - 1 - d] = pattern.pattern[2][n - 1 - d];
        pattern.pattern[2][n - 1 - d] = pattern.pattern[1][n - 1 - d];
        pattern.pattern[1][n - 1 - d] = pattern.pattern[3][d];
        pattern.pattern[3][d] = temp;

        std::reverse(pattern.pattern[3][d].begin(), pattern.pattern[3][d].end());
    }

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(1);
    pattern.transposeFace(3);
}

void Rubiks::rotateL(int depth) {
    int n = pattern.pattern[4].size();
    pattern.rotateFaceClockwise(4);

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(1);
    pattern.transposeFace(3);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        std::vector<char> temp = pattern.pattern[0][d];
        std::reverse(temp.begin(), temp.end());
        std::reverse(pattern.pattern[3][n - 1 - d].begin(), pattern.pattern[3][n - 1 - d].end());
        
        pattern.pattern[0][d] = pattern.pattern[3][n - 1 - d];
        pattern.pattern[3][n - 1 - d] = pattern.pattern[1][d];
        pattern.pattern[1][d] = pattern.pattern[2][d];
        pattern.pattern[2][d] = temp;

        std::reverse(pattern.pattern[3][n - 1 - d].begin(), pattern.pattern[3][n - 1 - d].end());
    }

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(1);
    pattern.transposeFace(3);
}


Move Rubiks::parseMove(const std::string& token){
    Move m;

    size_t pos = 0;

    // depth of the move
    int layer = 0;

    while (pos < token.size() && std::isdigit(token[pos]))
    {
        layer = layer * 10 + (token[pos] - '0'); // dunno why I substracted 0 here
        pos++;
    }

    if (layer == 0) {
        m.depth = 0;
    } else {
        m.depth = layer - 1;
    }

    // face to apply the move
    m.face = token[pos++];

    // suffixe (number of times we do the move)
    m.turns = 1;
    if (pos < token.size())
    {
        if (token[pos] == '\'')
            m.turns = 3;
        else if (token[pos] == '2')
            m.turns = 2;
    }

    return m;
}

// Write alg with the form : dFn, where d is the depth+1, F is the face, and n is either ' or 2, d and n are optionnal.
// This does not know moves like M, S, E, r, f etc..., support for those might be added in the future.
void Rubiks::exec(const std::string& alg){ 
    std::stringstream ss(alg);
    std::string token;

    
    while (ss >> token) { // separate alg into individual moves (HTM)
        Move m = parseMove(token);

        for (int i = 0; i < m.turns; ++i) {
            switch (m.face) {
                case 'U': rotateU(m.depth); break;
                case 'D': rotateD(m.depth); break;
                case 'F': rotateF(m.depth); break;
                case 'B': rotateB(m.depth); break;
                case 'R': rotateR(m.depth); break;
                case 'L': rotateL(m.depth); break;
                default:
                    std::cerr << "Unknown face : " << m.face << std::endl;
                    break;
            }
        }
    }
}

int test_rubiks(){
    // Rubiks cube(4);
    
    // some testing for rotateclockwise
    // cube.B.stickers[0][0] = '1';
    // cube.B.stickers[0][1] = '2';
    // cube.B.stickers[0][2] = '3';
    // cube.B.stickers[1][0] = '4';
    // cube.B.stickers[1][1] = '5';
    // cube.B.stickers[1][2] = '6';
    // cube.B.stickers[2][0] = '7';
    // cube.B.stickers[2][1] = '8';
    // cube.B.stickers[2][2] = '9';
    

    // std::cout << "Before alg:\n";
    // cube.print();

    // // exec a Y perm
    // cube.exec("F R U' R' U' R U R' F' R U R' U' R' F R F'"); // yay Y-perm
    
    // // exec a J perm
    // // cube.exec(" R U R' F' R U R' U' R' F R2 U' R' U'"); // yay J-perm

    // // cube.exec(" R U R' U' R' F R2 U' R' U' R U R' F'"); // yay T-perm

    // std::cout << "\n After alg:\n";
    // cube.print();
    return 0;
}




