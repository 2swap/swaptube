
#include <vector>
#include <iostream>
#include <algorithm>


// Data : cube state, moves to do
// State : zoom, camera position, size of the cube, shape of the cube (encoded with a number idk)


// For now, a face is a matrix, and a cube is a vector of 6 faces, maybe in the future, I'll rewrite with either a coordinate
//system for each piece, or even something like permutations, idk


// creates a square matrice, size is "size", with the same char everywhere, I'll use chars to know what color is a sticker.
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

        void rotateClockwise(){
            int n = stickers.size();

            // 1. transpose
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
               {
                    std::swap(stickers[i][j], stickers[j][i]);
                }
            }

            // 2. reverse each row
            for (int i = 0; i < n; i++)
            {
                std::reverse(stickers[i].begin(), stickers[i].end());
            }
        }
};

class Cube{
    public:
        Face U, D, F, B, L, R;

        Cube(int size)
            : U(size, 'W'),
            D(size, 'Y'),
            F(size, 'G'),
            B(size, 'B'),
            L(size, 'O'),
            R(size, 'R'){
        }

        void print(){
            int n = U.stickers.size();

            // Up face
            for (int i = 0; i < n; i++){
                std::cout << std::string(n * 2 + 2, ' ');

                for (char c : U.stickers[i])
                    std::cout << c << ' ';

                std::cout << '\n';
            }

            std::cout << '\n';

            // L F R B
            for (int i = 0; i < n; i++){
                for (char c : L.stickers[i])
                    std::cout << c << ' ';

                std::cout << "  ";

                for (char c : F.stickers[i])
                    std::cout << c << ' ';

                std::cout << "  ";

                for (char c : R.stickers[i])
                    std::cout << c << ' ';

                std::cout << "  ";

                for (char c : B.stickers[i])
                   std::cout << c << ' ';

                std::cout << '\n';
            }

            std::cout << '\n';

            // D
            for (int i = 0; i < n; i++){
                std::cout << std::string(n * 2 + 2, ' ');

                for (char c : D.stickers[i])
                    std::cout << c << ' ';

                std::cout << '\n';
            }
        }

        void rotateU(int depth){
            if (depth == 0){
                U.rotateClockwise();
            }
            if (depth < U.stickers.size() - 1){ // this does the move on the adjacent faces
                std::vector<char> temp = F.stickers[depth];
                F.stickers[depth] = R.stickers[depth];
                R.stickers[depth] = B.stickers[depth];
                B.stickers[depth] = L.stickers[depth];
                L.stickers[depth] = temp;
            };
        }

        void rotateD(int depth){
            int n = D.stickers.size();
            if (depth == 0){
                D.rotateClockwise();
            }
            if (depth < D.stickers.size() - 1){
                std::vector<char> temp = F.stickers[n - 1 - depth];
                F.stickers[n - 1 - depth] = L.stickers[n - 1 - depth];
                L.stickers[n - 1 - depth] = B.stickers[n - 1 - depth];
                B.stickers[n - 1 - depth] = R.stickers[n - 1 - depth];
                R.stickers[n - 1 - depth] = temp;
            };
        }

        void rotateF(int depth){
            int n = F.stickers.size();
            if (depth == 0){
                F.rotateClockwise();
            }
            if (depth < F.stickers.size() - 1){// -1 because just do B instead of F lol

                for (int i = 0; i < n; i++){ // transpose L
                    for (int j = i + 1; j < n; j++){
                        std::swap(L.stickers[i][j], L.stickers[j][i]);
                    }
                }
                // TODO make a transpose function for god's sake

                for (int i = 0; i < n; i++){ // transpose R
                    for (int j = i + 1; j < n; j++){
                        std::swap(R.stickers[i][j], R.stickers[j][i]);
                    }
                }

                std::vector<char> temp = U.stickers[n - 1 - depth];
                U.stickers[n - 1 - depth] = L.stickers[n - 1 - depth];
                L.stickers[n - 1 - depth] = D.stickers[depth];
                D.stickers[depth] = R.stickers[depth];
                R.stickers[depth] = temp;

                for (int i = 0; i < n; i++){ // transpose L
                    for (int j = i + 1; j < n; j++){
                        std::swap(L.stickers[i][j], L.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose R
                    for (int j = i + 1; j < n; j++){
                        std::swap(R.stickers[i][j], R.stickers[j][i]);
                    }
                }
                
            };
        }

        void rotateB(int depth){
            int n = B.stickers.size();
            if (depth == 0){
                B.rotateClockwise();
            }
            if (depth < B.stickers.size() - 1){// -1 because just do B instead of F lol

                for (int i = 0; i < n; i++){ // transpose L
                    for (int j = i + 1; j < n; j++){
                        std::swap(L.stickers[i][j], L.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose R
                    for (int j = i + 1; j < n; j++){
                        std::swap(R.stickers[i][j], R.stickers[j][i]);
                    }
                }

                std::vector<char> temp = U.stickers[depth];
                U.stickers[depth] = R.stickers[n - 1 - depth];
                R.stickers[n - 1 - depth] = D.stickers[n - 1 - depth];
                D.stickers[n - 1 - depth] = L.stickers[depth];
                L.stickers[depth] = temp;

                for (int i = 0; i < n; i++){ // transpose L
                    for (int j = i + 1; j < n; j++){
                        std::swap(L.stickers[i][j], L.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose R
                    for (int j = i + 1; j < n; j++){
                        std::swap(R.stickers[i][j], R.stickers[j][i]);
                    }
                }
                
            };
        }

        void rotateR(int depth){
            int n = R.stickers.size();
            if (depth == 0){
                R.rotateClockwise();
            }
            if (depth < R.stickers.size() - 1){// -1 because just do B instead of F lol

                for (int i = 0; i < n; i++){ // transpose U
                    for (int j = i + 1; j < n; j++){
                        std::swap(U.stickers[i][j], U.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose F
                    for (int j = i + 1; j < n; j++){
                        std::swap(F.stickers[i][j], F.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose D
                    for (int j = i + 1; j < n; j++){
                        std::swap(D.stickers[i][j], D.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose B
                    for (int j = i + 1; j < n; j++){
                        std::swap(B.stickers[i][j], B.stickers[j][i]);
                    }
                }

                std::vector<char> temp = U.stickers[n - 1 - depth];
                U.stickers[n - 1 - depth] = F.stickers[n - 1 - depth];
                F.stickers[n - 1 - depth] = D.stickers[n - 1 - depth];
                D.stickers[n - 1 - depth] = B.stickers[depth];
                B.stickers[depth] = temp;

                for (int i = 0; i < n; i++){ // transpose U
                    for (int j = i + 1; j < n; j++){
                        std::swap(U.stickers[i][j], U.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose F
                    for (int j = i + 1; j < n; j++){
                        std::swap(F.stickers[i][j], F.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose D
                    for (int j = i + 1; j < n; j++){
                        std::swap(D.stickers[i][j], D.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose B
                    for (int j = i + 1; j < n; j++){
                        std::swap(B.stickers[i][j], B.stickers[j][i]);
                    }
                }
                
            };
        }

        void rotateL(int depth){
            int n = L.stickers.size();
            if (depth == 0){
                L.rotateClockwise();
            }
            if (depth < L.stickers.size() - 1){// -1 because just do B instead of F lol

                for (int i = 0; i < n; i++){ // transpose U
                    for (int j = i + 1; j < n; j++){
                        std::swap(U.stickers[i][j], U.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose F
                    for (int j = i + 1; j < n; j++){
                        std::swap(F.stickers[i][j], F.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose D
                    for (int j = i + 1; j < n; j++){
                        std::swap(D.stickers[i][j], D.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose B
                    for (int j = i + 1; j < n; j++){
                        std::swap(B.stickers[i][j], B.stickers[j][i]);
                    }
                }

                std::vector<char> temp = U.stickers[depth];
                U.stickers[depth] = B.stickers[n - 1 - depth];
                B.stickers[n - 1 - depth] = D.stickers[depth];
                D.stickers[depth] = F.stickers[depth];
                F.stickers[depth] = temp;

                for (int i = 0; i < n; i++){ // transpose U
                    for (int j = i + 1; j < n; j++){
                        std::swap(U.stickers[i][j], U.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose F
                    for (int j = i + 1; j < n; j++){
                        std::swap(F.stickers[i][j], F.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose D
                    for (int j = i + 1; j < n; j++){
                        std::swap(D.stickers[i][j], D.stickers[j][i]);
                    }
                }

                for (int i = 0; i < n; i++){ // transpose B
                    for (int j = i + 1; j < n; j++){
                        std::swap(B.stickers[i][j], B.stickers[j][i]);
                    }
                }
                
            };
        }

};

void rubiks_test(){
    Cube cube(3);
    
    // change up face stickers
    // cube.B.stickers[0][0] = '1';
    // cube.B.stickers[0][1] = '2';
    // cube.B.stickers[0][2] = '3';
    // cube.B.stickers[1][0] = '4';
    // cube.B.stickers[1][1] = '5';
    // cube.B.stickers[1][2] = '6';
    // cube.B.stickers[2][0] = '7';
    // cube.B.stickers[2][1] = '8';
    // cube.B.stickers[2][2] = '9';
    

    std::cout << "Before rotation:\n";
    cube.print();

    cube.rotateR(0);
    cube.rotateU(0);
    cube.rotateR(0);
    cube.rotateL(0);

    std::cout << "\n After rotation:\n";
    cube.print();
}

// TODO y'a des lignes à inverser par exemple quand on fait L, colonne du fond remplace à l'envers celle du dessus


