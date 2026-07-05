#pragma once
#include <vector> // for vectors and matrices
#include <iostream> // to print the cubes in the terminal
#include <algorithm> // usefull things like reverse for a vector
#include <sstream> // used in alg tokenisation
#include "../Host_Device_Shared/vec.h"
#include <unordered_map>
#include "DataObject.h"


int test_rubiks();

enum FaceName { 
    IDX_U = 0, 
    IDX_D = 1, 
    IDX_F = 2, 
    IDX_B = 3, 
    IDX_L = 4, 
    IDX_R = 5 
};

struct CubeStickerPattern{
    std::vector<std::vector<std::vector<char>>> pattern;
    CubeStickerPattern();

    CubeStickerPattern(int size) {
        char colors[6] = {'W', 'Y', 'G', 'B', 'O', 'R'};
        
        pattern.reserve(6);
        for (int i = 0; i < 6; ++i) {
            pattern.push_back(std::vector<std::vector<char>>(size, std::vector<char>(size, colors[i])));
        }
    }

    void transposeFace(int faceIdx) {
        int n = pattern[faceIdx].size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                std::swap(pattern[faceIdx][i][j], pattern[faceIdx][j][i]);
            }
        }
    }

    void rotateFaceClockwise(int faceIdx) {
        transposeFace(faceIdx);
        int n = pattern[faceIdx].size();
        for (int i = 0; i < n; i++) {
            std::reverse(pattern[faceIdx][i].begin(), pattern[faceIdx][i].end());
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
        
        // Write alg with the form : dFn, where d is the depth+1, F is the face, and n is either ' or 2, d and n are optionnal.
        // This does not know moves like M, S, E, r, f etc..., support for those might be added in the future.
        void exec(const std::string& alg);

};
