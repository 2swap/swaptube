using namespace std;
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <typeinfo>
#include <future>
#include <string>
#include <thread>
#include <chrono>
#include "PuzzleBuilder.cpp"
#include "PuzzleSolver.cpp"

struct Statistics {
    double mean_time_taken;
    double mean_element_impact;
    double mean_primordial_size;
    double mean_num_elements;
    double success_ratio;
};

Puzzle create_3x3x3();
Puzzle create_2x2x2();
Puzzle create_swap_and_cycle_puzzle();
Puzzle create_buffered_symmetric_group_puzzle();
Puzzle create_s4_puzzle();
Puzzle create_rectangle_slider();
Puzzle create_cheese_puzzle();
Puzzle create_nonsense_puzzle_five_moves();
Puzzle create_nonsense_puzzle_two_moves();

using PuzzleCreator = Puzzle(*)();

vector<PuzzleCreator> puzzle_creators = {
    create_3x3x3,
    create_2x2x2,
    create_swap_and_cycle_puzzle,
    create_buffered_symmetric_group_puzzle,
    create_cheese_puzzle,
    create_rectangle_slider,
    create_nonsense_puzzle_two_moves,
    create_nonsense_puzzle_five_moves,
    create_s4_puzzle
};

enum class SolveMethod {
    GREEDY,
    GREEDY_TRIPLES,
    DELIBERATELY
};

vector<SolveMethod> solve_methods = {
    SolveMethod::GREEDY,
    SolveMethod::GREEDY_TRIPLES,
    SolveMethod::DELIBERATELY
};

std::string get_puzzle_name(PuzzleCreator creator) {
    if (creator == create_3x3x3) return "3x3x3";
    if (creator == create_2x2x2) return "2x2x2";
    if (creator == create_swap_and_cycle_puzzle) return "SwapAndCycle";
    if (creator == create_buffered_symmetric_group_puzzle) return "BufferedSymmetricGroup";
    if (creator == create_s4_puzzle) return "S4";
    if (creator == create_rectangle_slider) return "RectangleSlider";
    if (creator == create_cheese_puzzle) return "Cheese";
    if (creator == create_nonsense_puzzle_five_moves) return "NonsenseFiveMoves";
    if (creator == create_nonsense_puzzle_two_moves) return "NonsenseTwoMoves";
    return "Unknown";
}

std::string get_method_name(SolveMethod method) {
    switch (method) {
        case SolveMethod::GREEDY: return "Greedy";
        case SolveMethod::GREEDY_TRIPLES: return "GreedyTriples";
        case SolveMethod::DELIBERATELY: return "Deliberately";
        default: return "Unknown";
    }
}

Statistics run_experiment(PuzzleCreator create_puzzle, SolveMethod method) {
    int num_attempts = 100;
    int success_count = 0;
    double total_time = 0;
    double total_element_impact = 0;
    double total_primordial_size = 0;
    double total_num_elements = 0;

    for (int i = 0; i < num_attempts; ++i) {
        Puzzle p = create_puzzle();
        p.scramble();

        PuzzleSolver solver(p);
        auto start_time = chrono::high_resolution_clock::now();

        cout << "Puzzle: " << get_puzzle_name(create_puzzle) << ", Method: " << get_method_name(method) << ", Attempt: " << i+1 << endl;

        bool success = false;
        if (method == SolveMethod::GREEDY) {
            success = solver.solve_greedy();
        } else if (method == SolveMethod::GREEDY_TRIPLES) {
            success = solver.solve_greedy_triples();
        } else {
            success = solver.solve_deliberately();
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;

        if (success) {
            ++success_count;
            total_time += elapsed.count();
            total_element_impact += solver.get_average_element_impact();
            total_primordial_size += solver.get_average_primordial_size();
            total_num_elements += solver.get_number_of_elements();
        }
    }

    Statistics stats;
    if (success_count > 0) {
        stats.mean_time_taken = total_time / success_count;
        stats.mean_element_impact = total_element_impact / success_count;
        stats.mean_primordial_size = total_primordial_size / success_count;
        stats.mean_num_elements = total_num_elements / success_count;
    } else {
        stats.mean_time_taken = 0;
        stats.mean_element_impact = 0;
        stats.mean_primordial_size = 0;
        stats.mean_num_elements = 0;
    }
    stats.success_ratio = static_cast<double>(success_count) / num_attempts;

    return stats;
}

int main() {
    pieceset_unit_tests();
    std::ofstream output_file("experiment_results.dat");

    output_file << "# Puzzle Method Mean_Time_Taken Mean_Element_Impact Mean_Primordial_Size Mean_Num_Elements Success_Ratio\n";

    for (auto& create_puzzle : puzzle_creators) {
        for (auto& method : solve_methods) {
            Statistics stats = run_experiment(create_puzzle, method);
            output_file << get_puzzle_name(create_puzzle) << " "
                        << get_method_name(method) << " "
                        << stats.mean_time_taken << " "
                        << stats.mean_element_impact << " "
                        << stats.mean_primordial_size << " "
                        << stats.mean_num_elements << " "
                        << stats.success_ratio << "\n";
        }
    }

    output_file.close();
    return 0;
}
