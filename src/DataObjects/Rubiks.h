#pragma once
#include <vector> // for vectors and matrices
#include <iostream> // to print the cubes in the terminal
#include <algorithm> // usefull things like reverse for a vector
#include <sstream> // used in alg tokenisation
#include "../Host_Device_Shared/vec.h"
#include <unordered_map>
#include "DataObject.h"

const int MAX_CUBE_SIZE = 11; // maximum size of the cube, this is used to define the size of the pattern array in CubeStickerPattern
int test_rubiks();

enum FaceName { 
    IDX_U = 0, 
    IDX_L = 1, 
    IDX_F = 2, 
    IDX_R = 3, 
    IDX_B = 4, 
    IDX_D = 5 
};

struct CubeStickerPattern{
    char pattern[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE]; // 6 faces, each face can be up to 10x10 stickers
    CubeStickerPattern();

    CubeStickerPattern(int size) {
        char colors[6] = {'W', 'O', 'G', 'R', 'B', 'Y'};
        
        for (int f = 0; f < 6; ++f) {
            for (int i = 0; i < MAX_CUBE_SIZE; ++i) {
                for (int j = 0; j < MAX_CUBE_SIZE; ++j) {
                    pattern[f][i][j] = colors[f]; // testing
                }
            }
        }
    }

    void transposeFace(int faceIdx) {
        for (int i = 0; i < MAX_CUBE_SIZE; i++) {
            for (int j = i + 1; j < MAX_CUBE_SIZE; j++) {
                std::swap(pattern[faceIdx][i][j], pattern[faceIdx][j][i]);
            }
        }
    }

    void rotateFaceClockwise(int faceIdx) {
        transposeFace(faceIdx);
        for (int i = 0; i < MAX_CUBE_SIZE; i++) {
            std::reverse(pattern[faceIdx][i], pattern[faceIdx][i+1]);
        }
    }
};

struct Cut{
    vec3 axis;
    float dist;
    Cut(const vec3& axis, float dist);
    Cut();
};




struct Move {
            char face;
            int depth;
            int turns;
        };



class Rubiks : public DataObject {
    public:
        std::unordered_map<char, std::vector<Cut>> cut_map;

        CubeStickerPattern pattern;

        void tick(const StateReturn& state);

        Rubiks(int size) : pattern(size) {
            for (int i = 1; i < size; ++i) {
                float distance = -1.0f + (2.0f * static_cast<float>(size - i)) / static_cast<float>(size);
                
                cut_map['U'].push_back(Cut(vec3(0,  1,  0), distance));
                cut_map['D'].push_back(Cut(vec3(0, -1,  0), distance));
                cut_map['F'].push_back(Cut(vec3(0,  0, -1), distance));
                cut_map['B'].push_back(Cut(vec3(0,  0,  1), distance));
                cut_map['L'].push_back(Cut(vec3(-1, 0,  0), distance));
                cut_map['R'].push_back(Cut(vec3(1,  0,  0), distance));
            }
        }

        // Here I had to define each face rotation individually, 
        // this is due to the fact that matrices defining faces have different orientations
        void rotateU(int depth);

        void rotateD(int depth);

        void rotateF(int depth);

        void rotateB(int depth);

        void rotateR(int depth);

        void rotateL(int depth);

        Move parseMove(const std::string& token);

        void print();

        double get_hash();
        
        // Write alg with the form : dFn, where d is the depth+1, F is the face, and n is either ' or 2, d and n are optionnal.
        // This does not know moves like M, S, E, r, f etc..., support for those might be added in the future.
        void exec(const std::string& alg);

};
