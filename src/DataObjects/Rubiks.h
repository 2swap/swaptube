#pragma once
#include <vector> // for vectors and matrices
#include <iostream> // to print the cubes in the terminal
#include <algorithm> // usefull things like reverse for a vector
#include <sstream> // used in alg tokenisation

int test_rubiks();

struct Move {
            char face;
            int depth;
            int turns;
        };

class Face{
    public:
        std::vector<std::vector<char>> stickers;

        Face(int size, char color){
            stickers = std::vector<std::vector<char>>(
                size,
                std::vector<char>(size, color)
            );
        }

        void print(){
            for (auto& row : stickers)
            {
                for (char c : row)
                    std::cout << c << ' ';
                std::cout << '\n';
            }
        }

        void transpose(){
            for (int i = 0; i < stickers.size(); i++)
            {
                for (int j = i + 1; j < stickers.size(); j++)
               {
                    std::swap(stickers[i][j], stickers[j][i]);
                }
            }
        }

        void rotateClockwise(){ // to achieve this, we transpose and then reverse each row
            transpose();

            int n = stickers.size();
            for (int i = 0; i < n; i++)
            {
                std::reverse(stickers[i].begin(), stickers[i].end());
            }
        }
};

class Rubiks{
    public:
        Face U, D, F, B, L, R;

        Rubiks(int size);

        void print();

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
