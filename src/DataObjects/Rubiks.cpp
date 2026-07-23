#include "Rubiks.h"
#include <cmath>
#include <unordered_map>
#include "../Host_Device_Shared/vec.h"



// For now, a face is a matrix, and a cube is a vector of 6 faces, maybe in the future, I'll rewrite with either a coordinate
//system for each piece, or even something like permutations, this could be usefull for some of the graph views we planned


// a face is a square matrice, size is "size", with the same char everywhere, each char corresponds to a color.


Cut::Cut(const vec3& axis, float dist) : axis(axis), dist(dist) {};
Cut::Cut() : axis(vec3(0, 0, 0)), dist(0) {}



void Rubiks::tick(const StateReturn& state) {}


double Rubiks::get_hash() {
    double hash = 0.0;
    double c = 0.0;
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            for (int j = 0; j < MAX_CUBE_SIZE; ++j) {
                c+= 1.14159*j+25*i-27*f;
                switch (pattern.pattern[f][i][j]) {
                    case 'W': {
                        hash += extended_mod(135.41*c, 13*i-142*j+2*f+123.12);
                    };
                    case 'O': {
                        hash += extended_mod(147.425*c, 474*f+2*j-8*i-5.15);
                    };
                    case 'G': {
                        hash += extended_mod(20.1331*c, 501*i-25*f-4*j+12.12);
                    };
                    case 'R': {
                        hash += extended_mod(0.14157*c, 3*j+7*i-2*f-48.56);
                    };
                    case 'B': {
                        hash += extended_mod(3.141*c, 45*f-78*j+36*i+7.00125);
                    };
                    case 'Y': {
                        hash += extended_mod(70.31*c, 3*i+2*j-f+45.36);
                    };
                }
            }
        }
    }
    return hash;
}


// Here I had to define each face rotation individually, 
// this is due to the fact that matrices defining faces have different orientations


// initially, depth would only turn the said depth, but 2swap asked that depth also turns the layers in between, so now depth is the maximum depth to turn, and all layers in between are turned as well.
void Rubiks::rotateU(int depth){
    pattern.rotateFaceClockwise(0);

    for (int d = 0; d <= depth && d < MAX_CUBE_SIZE - 1; ++d) {
        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[2][d][i];
        }
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[2][d][i] = pattern.pattern[3][d][i];
            pattern.pattern[3][d][i] = pattern.pattern[4][d][i];
            pattern.pattern[4][d][i] = pattern.pattern[1][d][i];
            pattern.pattern[1][d][i] = temp[i];
        }

    }
    //print();
}



void Rubiks::rotateD(int depth) {
    pattern.rotateFaceClockwise(5);

    for (int d = 0; d <= depth && d < MAX_CUBE_SIZE - 1; ++d) {
        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[2][MAX_CUBE_SIZE - 1 - d][i];
        }
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[2][MAX_CUBE_SIZE - 1 - d][i] = pattern.pattern[1][MAX_CUBE_SIZE - 1 - d][i];
            pattern.pattern[1][MAX_CUBE_SIZE - 1 - d][i] = pattern.pattern[4][MAX_CUBE_SIZE - 1 - d][i];
            pattern.pattern[4][MAX_CUBE_SIZE - 1 - d][i] = pattern.pattern[3][MAX_CUBE_SIZE - 1 - d][i];
            pattern.pattern[3][MAX_CUBE_SIZE - 1 - d][i] = temp[i];
        }
    }

    //print();
}


void Rubiks::rotateF(int depth) {
    int n = MAX_CUBE_SIZE;
    pattern.rotateFaceClockwise(2);

    pattern.transposeFace(1);
    pattern.transposeFace(3);

    for (int d = 0; d <= depth && d < n - 1; ++d) {

        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[0][n - 1 - d][i];
        }
        std::reverse(pattern.pattern[1][n - 1 - d], pattern.pattern[1][n - 1 - d+1]);
        std::reverse(pattern.pattern[3][d], pattern.pattern[3][d+1]);
        
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[0][n - 1 - d][i] = pattern.pattern[1][n - 1 - d][i];
            pattern.pattern[1][n - 1 - d][i] = pattern.pattern[5][d][i];
            pattern.pattern[5][d][i] = pattern.pattern[3][d][i];
            pattern.pattern[3][d][i] = temp[i];
        }
        
        // std::reverse(pattern.pattern[4][n - 1 - d], pattern.pattern[4][n - 1 - d+1]);
        // std::reverse(pattern.pattern[5][d], pattern.pattern[5][d+1]);
    }

    pattern.transposeFace(1);
    pattern.transposeFace(3);

    //print();
}

void Rubiks::rotateB(int depth) {
    int n = MAX_CUBE_SIZE;
    pattern.rotateFaceClockwise(4);

    pattern.transposeFace(1);
    pattern.transposeFace(3);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[0][d][i];
        }
        std::reverse(pattern.pattern[5][n - 1 - d], pattern.pattern[5][n - 1 - d+1]);
        
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[0][d][i] = pattern.pattern[3][n - 1 - d][i];
            pattern.pattern[3][n - 1 - d][i] = pattern.pattern[5][n - 1 - d][i];
            pattern.pattern[5][n - 1 - d][i] = pattern.pattern[1][d][i];
            pattern.pattern[1][d][i] = temp[i];
        }

        std::reverse(pattern.pattern[1][d], pattern.pattern[1][d+1]);
    }

    pattern.transposeFace(1);
    pattern.transposeFace(3);

    //print();
}

void Rubiks::rotateR(int depth) {
    int n = MAX_CUBE_SIZE;
    pattern.rotateFaceClockwise(3);

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(4);
    pattern.transposeFace(5);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[0][n - 1 - d][i];
        }
        std::reverse(pattern.pattern[4][d], pattern.pattern[4][d+1]);
        std::reverse(temp, &(temp[MAX_CUBE_SIZE])); 
        
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[0][n - 1 - d][i] = pattern.pattern[2][n - 1 - d][i];
            pattern.pattern[2][n - 1 - d][i] = pattern.pattern[5][n - 1 - d][i];
            pattern.pattern[5][n - 1 - d][i] = pattern.pattern[4][d][i];
            pattern.pattern[4][d][i] = temp[i];
        }
    }

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(4);
    pattern.transposeFace(5);

    //print();
}

void Rubiks::rotateL(int depth) {
    int n = MAX_CUBE_SIZE;
    pattern.rotateFaceClockwise(1);

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(4);
    pattern.transposeFace(5);

    for (int d = 0; d <= depth && d < n - 1; ++d) {
        char temp[MAX_CUBE_SIZE];
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            temp[i] = pattern.pattern[0][d][i];
        }
        std::reverse(pattern.pattern[4][n - 1 - d], pattern.pattern[4][n - 1 - d+1]);
        std::reverse(pattern.pattern[5][d], pattern.pattern[5][d+1]);

        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            pattern.pattern[0][d][i] = pattern.pattern[4][n - 1 - d][i];
            pattern.pattern[4][n - 1 - d][i] = pattern.pattern[5][d][i];
            pattern.pattern[5][d][i] = pattern.pattern[2][d][i];
            pattern.pattern[2][d][i] = temp[i];
        }

    }

    pattern.transposeFace(0);
    pattern.transposeFace(2);
    pattern.transposeFace(4);
    pattern.transposeFace(5);

    //print();
}

void Rubiks::print() {
    for (int f = 0; f < 6; ++f) {
        std::cout << "Face " << f << ":\n";
        for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
            for (int j = 0; j < MAX_CUBE_SIZE; ++j) {
                std::cout << pattern.pattern[f][i][j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
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




