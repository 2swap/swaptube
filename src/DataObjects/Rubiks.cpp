
#include <vector> // for vectors and matrices
#include <iostream> // to print the cubes in the terminal
#include <algorithm> // usefull things like reverse for a vector
#include <sstream> // used in alg tokenisation


// Data : cube state, moves to do
// State : zoom, camera position, size of the cube, shape of the cube (encoded with a number idk)


// For now, a face is a matrix, and a cube is a vector of 6 faces, maybe in the future, I'll rewrite with either a coordinate
//system for each piece, or even something like permutations, this could be usefull for some of the graph views we planned


// a face is a square matrice, size is "size", with the same char everywhere, each char corresponds to a color.
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

        void print(){ // this could probably be better written using the print function defined in Face class
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

        // Here I had to define each face rotation individually, 
        // this is due to the fact that matrices defining faces have different orientations
        void rotateU(int depth){
            if (depth == 0){
                U.rotateClockwise();
            }
            if (depth < U.stickers.size() - 1){ // this does the move on the adjacent faces, also, -1 because just do D instead
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
            if (depth < F.stickers.size() - 1){

                L.transpose();
                R.transpose();

                std::vector<char> temp = U.stickers[n - 1 - depth];
                std::reverse(L.stickers[n - 1 - depth].begin(), L.stickers[n - 1 - depth].end());
                std::reverse(R.stickers[depth].begin(), R.stickers[depth].end());
                U.stickers[n - 1 - depth] = L.stickers[n - 1 - depth];
                L.stickers[n - 1 - depth] = D.stickers[depth];
                D.stickers[depth] = R.stickers[depth];
                R.stickers[depth] = temp;

                L.transpose();
                R.transpose();
                
            };
        }

        void rotateB(int depth){
            int n = B.stickers.size();
            if (depth == 0){
                B.rotateClockwise();
            }
            if (depth < B.stickers.size() - 1){

                L.transpose();
                R.transpose();

                std::vector<char> temp = U.stickers[depth];
                std::reverse(L.stickers[depth].begin(), L.stickers[depth].end());
                std::reverse(R.stickers[n - 1 - depth].begin(), R.stickers[n - 1 - depth].end());
                U.stickers[depth] = R.stickers[n - 1 - depth];
                R.stickers[n - 1 - depth] = D.stickers[n - 1 - depth];
                D.stickers[n - 1 - depth] = L.stickers[depth];
                L.stickers[depth] = temp;

                L.transpose();
                R.transpose();
                
            };
        }

        void rotateR(int depth){
            int n = R.stickers.size();
            if (depth == 0){
                R.rotateClockwise();
            }
            if (depth < R.stickers.size() - 1){

                U.transpose();
                F.transpose();
                D.transpose();
                B.transpose();

                std::vector<char> temp = U.stickers[n - 1 - depth];
                std::reverse(B.stickers[depth].begin(), B.stickers[depth].end());
                std::reverse(temp.begin(), temp.end());
                U.stickers[n - 1 - depth] = F.stickers[n - 1 - depth];
                F.stickers[n - 1 - depth] = D.stickers[n - 1 - depth];
                D.stickers[n - 1 - depth] = B.stickers[depth];
                B.stickers[depth] = temp;

                U.transpose();
                F.transpose();
                D.transpose();
                B.transpose();
                
            };
        }

        void rotateL(int depth){
            int n = L.stickers.size();
            if (depth == 0){
                L.rotateClockwise();
            }
            if (depth < L.stickers.size() - 1){

                U.transpose();
                F.transpose();
                D.transpose();
                B.transpose();

                std::vector<char> temp = U.stickers[depth];
                std::reverse(temp.begin(), temp.end());
                std::reverse(B.stickers[n - 1 - depth].begin(), B.stickers[n - 1 - depth].end());
                U.stickers[depth] = B.stickers[n - 1 - depth];
                B.stickers[n - 1 - depth] = D.stickers[depth];
                D.stickers[depth] = F.stickers[depth];
                F.stickers[depth] = temp;

                U.transpose();
                F.transpose();
                D.transpose();
                B.transpose();
                
            };
        }

        struct Move {
            char face;
            int depth;
            int turns;
        };

        Move parseMove(const std::string& token){
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
        void exec(const std::string& alg){ 
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

};

// DO NOT uncomment this and push, it will break things !
// int main(){
//     Cube cube(4);
    
//     // some testing for rotateclockwise
//     // cube.B.stickers[0][0] = '1';
//     // cube.B.stickers[0][1] = '2';
//     // cube.B.stickers[0][2] = '3';
//     // cube.B.stickers[1][0] = '4';
//     // cube.B.stickers[1][1] = '5';
//     // cube.B.stickers[1][2] = '6';
//     // cube.B.stickers[2][0] = '7';
//     // cube.B.stickers[2][1] = '8';
//     // cube.B.stickers[2][2] = '9';
    

//     std::cout << "Before alg:\n";
//     cube.print();

//     cube.exec(" R U R' U' R' F R2 U' R' U' R U R' F'"); // yay T-perm

//     std::cout << "\n After alg:\n";
//     cube.print();
//     return 0;
// }




