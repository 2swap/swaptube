#pragma once

#include "Puzzle.cpp"
#include "PermutationElement.cpp"
#include <set>
#include <cassert>
#include <random>

struct ComparePrimordialSize {
    bool operator()(const shared_ptr<PermutationElement>& lhs, const shared_ptr<PermutationElement>& rhs) const {
        return lhs->get_primordial_size() < rhs->get_primordial_size();
    }
};

template<typename T>
vector<T> shuffle(const multiset<T, ComparePrimordialSize>& multiset_elements) {
    vector<T> elements(multiset_elements.begin(), multiset_elements.end());
    random_device rd;
    mt19937 g(rd());
    shuffle(elements.begin(), elements.end(), g);
    return elements;
}

template<typename T, std::size_t N>
array<T, N> shuffle(const array<T, N>& array_elements) {
    array<T, N> elements = array_elements;
    random_device rd;
    mt19937 g(rd());
    shuffle(elements.begin(), elements.end(), g);
    return elements;
}

class PuzzleSolver {
public:
    PuzzleSolver(const Puzzle& puzzle) : puzzle(puzzle) {
        // Initialize permutation elements based on the puzzle's primordial elements
        for(shared_ptr<PermutationElement> elem : puzzle.get_primordial_elements()){
            permutation_elements.insert(elem);
        }
    }

    void print() const {
        cout << "PuzzleSolver for Puzzle:" << endl;
        puzzle.print();
        
        cout << "Permutation Elements:" << endl;
        for (const auto& elem : permutation_elements) {
            elem->print();
        }
    }

    bool solve_greedy(){
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
                        remember_permutation_if_nonduplicate(elem);
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
                            remember_permutation_if_nonduplicate(elem);
                            goto breakout;
                        }
                    }
                }
            }
            return false;
            breakout:;
        }
        puzzle.print();
        return true;
    }

    bool solve_greedy_triples(){
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
                        remember_permutation_if_nonduplicate(elem);
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
                            remember_permutation_if_nonduplicate(elem);
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
                                shared_ptr<PermutationElement> elem1 = make_shared<GeneratedElement>(p, q, g1);
                                shared_ptr<PermutationElement> elem2 = make_shared<GeneratedElement>(elem1, r, g2);
                                if (test_accept_if_improvement(elem2, solvedness)){
                                    remember_permutation_if_nonduplicate(elem2);
                                    goto breakout;
                                }
                            }
                        }
                    }
                }
            }
            return false;
            breakout:;
        }
        puzzle.print();
        return true;
    }

    bool solve_deliberately(){
        random_device dev;
        mt19937 rng(dev());
        uniform_int_distribution<mt19937::result_type> dist(0, 1);
        auto permutation_elements_copy = permutation_elements;

        //cout << "Phase 1: Adding all primordial cycles" << endl;
        for(auto p : permutation_elements_copy) {
            for(int i = 2; i < p->order(); i++){
                GeneratedElement gen(p, i);
                shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                remember_permutation_if_nonduplicate(elem);
            }
        }

        //cout << "Phase 2: Adding all primordial combos" << endl;
        for(auto p : permutation_elements_copy) {
            for(auto q : permutation_elements_copy) {
                for(GenerationMethod g : {GenerationMethod::COMPOSITION, GenerationMethod::CONJUGATION, GenerationMethod::COMMUTATION}){
                    GeneratedElement gen(p, q, g);
                    shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                    remember_permutation_if_nonduplicate(elem);
                }
            }
        }

        //cout << "Phase 3: Solving" << endl;
        unsigned long int known_solved_bitstrip = 0ul;
        const PieceSet full_pieceset = get_full_pieceset(puzzle.size());
        int flails = 0;
        while(!puzzle.is_solved()){
            PieceSet known_solved(known_solved_bitstrip);
            PieceSet next_location_to_solve(known_solved.get_bit_representation() + 1ul);
            assert(next_location_to_solve.get_size() == 1);
            int index = __builtin_ctz(next_location_to_solve.get_bit_representation()); // index of the one-bit.
            vector<int> state = puzzle.get_state();
            PieceSet location_of_displaced_piece = image(next_location_to_solve, state);
            assert(location_of_displaced_piece.get_size() == 1);

            if(is_equal(next_location_to_solve, location_of_displaced_piece)){
                //cout << "No-op" << endl;
                known_solved_bitstrip |= next_location_to_solve.get_bit_representation();
                goto breakout;
            }

            // Attempt commutation
            for(auto x : permutation_elements) {
                if(!is_equal(image(location_of_displaced_piece, x->get_effect()), next_location_to_solve)) continue; // doing x should put the piece in its home
                PieceSet x_set(x->get_effect());
                for(auto y : permutation_elements) {
                    if(does_intersect(image(next_location_to_solve, y->get_effect()), x_set)) continue; // y should take the piece's home out of the x-set
                    PieceSet y_set(y->get_effect());
                    if(does_intersect(location_of_displaced_piece, y_set)) continue; // the displaced piece should not be in y from the start
                    PieceSet buffer = get_intersection(x_set,y_set);
                    // we now know enough to know that this commutator will solve the piece.
                    // check that it wont break anything else
                    PieceSet x_set_to_buffer = preimage(buffer, x->get_effect());
                    PieceSet y_set_to_buffer = preimage(buffer, y->get_effect());
                    if(does_intersect(known_solved, buffer)) continue;
                    if(does_intersect(known_solved, x_set_to_buffer)) continue;
                    if(does_intersect(known_solved, y_set_to_buffer)) continue;

                    {
                        GeneratedElement gen(x, y, GenerationMethod::COMMUTATION);
                        shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                        puzzle.apply(*elem);
                        //elem->print();
                        remember_permutation_if_nonduplicate(elem);
                        known_solved_bitstrip |= next_location_to_solve.get_bit_representation();
                        goto breakout;
                    }
                }
            }

            // Attempt conjugation
            for(auto x : permutation_elements) {
                PieceSet x_set(x->get_effect());
                for(auto y : permutation_elements) {
                    PieceSet y_set(y->get_effect());
                    if(does_intersect(image(known_solved, x->get_effect()), y_set)) continue; // We should not plan a conjugation which will mess up something solved
                    GeneratedElement gen(x, y, GenerationMethod::CONJUGATION);
                    if(!is_equal(image(location_of_displaced_piece, gen.get_effect()), next_location_to_solve)) continue; // Our conjugate should solve our piece.

                    {
                        shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                        puzzle.apply(*elem);
                        //elem->print();
                        remember_permutation_if_nonduplicate(elem);
                        known_solved_bitstrip |= next_location_to_solve.get_bit_representation();
                        goto breakout;
                    }
                }
            }

            {
                if(flails++ > 100) return false;
                // Try anything that won't fuck it up more
                known_solved_bitstrip >>= known_solved.get_size()/2;
                PieceSet known_solved_new(known_solved_bitstrip);
                for(auto x : shuffle(permutation_elements)) {
                    for(auto y : shuffle(permutation_elements)) {
                        for(GenerationMethod g : shuffle(array<GenerationMethod, 3>{GenerationMethod::COMPOSITION, GenerationMethod::CONJUGATION, GenerationMethod::COMMUTATION})){
                            GeneratedElement gen(x, y, g);
                            if(does_intersect(known_solved_new, gen.get_effect())) continue; // We should not plan a move which will mess up something solved
                            if(gen.get_impact() == 0) continue; // no-ops are useless

                            shared_ptr<PermutationElement> elem = make_shared<GeneratedElement>(gen);
                            puzzle.apply(*elem);
                            //elem->print();
                            remember_permutation_if_nonduplicate(elem);
                            goto breakout;
                        }
                    }
                }
            }
            return false;
            breakout:;
        }
        puzzle.print();
        return true;
    }

    double get_average_element_impact(){
        double total = 0;
        for(auto p : permutation_elements) {
            total += p->get_impact();
        }
        return total/permutation_elements.size();
    }

    double get_number_of_elements(){
        return permutation_elements.size();
    }

    double get_average_primordial_size(){
        double total = 0;
        for(auto p : permutation_elements) {
            total += p->get_primordial_size();
        }
        return total/permutation_elements.size();
    }

private:
    Puzzle puzzle;
    multiset<shared_ptr<PermutationElement>, ComparePrimordialSize> permutation_elements;

    double get_average_yuckiness(){
        double total_yuckiness = 0;
        for(auto p : permutation_elements) {
            total_yuckiness += p->get_yuckiness();
        }
        return total_yuckiness/permutation_elements.size();
    }

    void remember_permutation_if_nonduplicate(const shared_ptr<PermutationElement>& elem) {
        if(is_duplicate(elem)) return;
        permutation_elements.insert(elem); // If no identical effect is found and it's not the identity, add the new element
        //elem->print();
    }

    bool is_duplicate(const shared_ptr<PermutationElement>& elem){
        // Check if the element is the identity permutation
        if (elem->get_effect() == PermutationElement::identity(elem->get_effect().size())) {
            return true; // If the element is the identity, do not add it
        }

        // Check if there is an existing permutation with the same effect
        for (const auto& existing_elem : permutation_elements) {
            if (existing_elem->get_effect() == elem->get_effect()) {
                return true; // If an identical effect is found, do not add the new element
            }
        }
        return false;
    }

    bool test_accept_if_improvement(const shared_ptr<PermutationElement>& elem, double solvedness){
        Puzzle test = puzzle;
        test.apply(*elem);
        if(test.solvedness() > solvedness){
            // accept this action
            puzzle.apply(*elem);
            //elem->print();
            return true;
        }
        return false;
    }
};
