#pragma once

#include "Puzzle.cpp"
#include "PermutationElement.cpp"

class PuzzleSolver {
public:
    PuzzleSolver(const Puzzle& puzzle) : puzzle(puzzle) {
        // Initialize permutation elements based on the puzzle's primordial elements
        permutation_elements = puzzle.get_primordial_elements();
    }

    void print() const {
        cout << "PuzzleSolver for Puzzle:" << endl;
        puzzle.print();
        
        cout << "Permutation Elements:" << endl;
        for (const auto& elem : permutation_elements) {
            elem->print();
        }
    }

    void solve_naive_greedy(){
        cout << "Phase 1: adding cycles" << endl;
        auto permutation_elements_copy = permutation_elements;
        for(auto p : permutation_elements_copy) {
            for(int i = 2; i < p->order(); i++){
                GeneratedElement gen(p, i);
                gen.print();
                shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                permutation_elements.push_back(elem);
            }
        }

        cout << "Phase 2: solving" << endl;
        while(!puzzle.is_solved()){
            double solvedness = puzzle.solvedness();
            //cout << solvedness << endl;

            //known elements
            for(auto p : permutation_elements) {
                if (test_accept_if_improvement(p, solvedness)){
                    goto breakout;
                }
            }

            //multiples
            for(int i = 1; i <= 6; i++){
                for(auto p : permutation_elements) {
                    GeneratedElement gen(p, i);
                    shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                    if (test_accept_if_improvement(elem, solvedness)){
                        permutation_elements.push_back(elem);
                        new_element_name_suggestion++;
                        goto breakout;
                    }
                }
            }

            //combos
            for(auto p : permutation_elements) {
                for(auto q : permutation_elements) {
                    for(GenerationMethod g : {GenerationMethod::COMPOSITION, GenerationMethod::CONJUGATION, GenerationMethod::COMMUTATION}){
                        GeneratedElement gen(p, q, g);
                        shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                        if (test_accept_if_improvement(elem, solvedness)){
                            permutation_elements.push_back(elem);
                            new_element_name_suggestion++;
                            goto breakout;
                        }
                    }
                }
            }

            //triples
            for(auto p : permutation_elements) {
                for(auto q : permutation_elements) {
                    for(auto r : permutation_elements) {
                        for(GenerationMethod g1 : {GenerationMethod::COMPOSITION, GenerationMethod::CONJUGATION, GenerationMethod::COMMUTATION}){
                            for(GenerationMethod g2 : {GenerationMethod::COMPOSITION, GenerationMethod::CONJUGATION, GenerationMethod::COMMUTATION}){
                                GeneratedElement gen1(p, q, g1);
                                shared_ptr<PermutationElement> elem1 = make_shared<GeneratedElement>(gen1);
                                GeneratedElement gen2(elem1, r, g2);
                                shared_ptr<PermutationElement> elem2 = make_shared<GeneratedElement>(gen2);
                                if (test_accept_if_improvement(elem2, solvedness)){
                                    permutation_elements.push_back(elem1);
                                    permutation_elements.push_back(elem2);
                                    new_element_name_suggestion+=2;
                                    goto breakout;
                                }
                            }
                        }
                    }
                }
            }
            puzzle.scramble();
            cout << "FAILED TO SOLVE" << endl;
            breakout:;
        }
        puzzle.print();
    }

    bool test_accept_if_improvement(const shared_ptr<PermutationElement>& elem, double solvedness){
        Puzzle test = puzzle;
        test.apply(*elem);
        if(test.solvedness() > solvedness){
            // accept this action
            puzzle.apply(*elem);
            elem->print();
            return true;
        }
        return false;
    }

private:
    int new_element_name_suggestion = 0;
    string solution = "";
    Puzzle puzzle;
    vector<shared_ptr<PermutationElement>> permutation_elements;

    string int_to_name(int x){
        return "" + string{static_cast<char>('a'+new_element_name_suggestion/26)}
                  + string{static_cast<char>('a'+new_element_name_suggestion%26)};
    }
};



