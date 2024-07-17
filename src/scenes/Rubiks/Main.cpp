using namespace std;

#include "PuzzleBuilder.cpp"
#include "PuzzleSolver.cpp"

int main() {
    // Create the S4 puzzle using the PuzzleBuilder
    Puzzle p = create_3x3x3();
    p.scramble();

    // Print the initial state of the puzzle
    cout << "Scrambled Puzzle:" << endl;
    p.print();

    // Construct the solver for the puzzle
    PuzzleSolver solver(p);
    solver.solve_naive_greedy();

    return 0;
}
