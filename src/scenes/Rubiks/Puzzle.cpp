#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <unordered_set>
#include "PermutationElement.cpp"
#include "PrimordialElement.cpp"
#include "GeneratedElement.cpp"

/*
 * A manipulable instantiation of a permutation group, with a tracked state
 */
class Puzzle {
public:
    Puzzle(const vector<int>& state, const vector<shared_ptr<PermutationElement>>& elements) : state(state), primordial_elements(elements) {

        if(false){
            // Verify that the set of primordial elements is closed under inversion
            for (const auto& elem : primordial_elements) {
                bool found_inverse = false;
                auto inverse_effect = ~elem->get_effect();
                for (const auto& check_elem : primordial_elements) {
                    if (check_elem->get_effect() == inverse_effect) {
                        found_inverse = true;
                        break;
                    }
                }
                if (!found_inverse) {
                    cout << "ERROR: Set of primordial elements is not closed under inversion. The following primordial has no inverse:" << endl;
                    elem->print();
                    exit(1);
                }
            }
        }

        // Verify that all primordial elements have unique names
        unordered_set<string> names;
        for (const auto& elem : primordial_elements) {
            string name = elem->get_name();
            if (names.find(name) != names.end()) {
                cout << "ERROR: Duplicate name found for primordial element: " << name << endl;
                exit(1);
            }
            names.insert(name);
        }
    }

    void print() const {
        cout << "=========" << endl;
        cout << "Puzzle state: ";
        for (int s : state) {
            cout << s << " ";
        }
        cout << endl;

        cout << "Primordial Elements:" << endl;
        for (const auto& elem : primordial_elements) {
            elem->print();
        }
        cout << "=========" << endl;
        cout << endl;

    }

    vector<shared_ptr<PermutationElement>> get_primordial_elements() const {
        return primordial_elements;
    }

    void apply(const string& name) {
        apply(*get_primordial(name));
    }

    void apply(const PermutationElement& elem) {
        state = state + elem.get_effect();
    }

    shared_ptr<PermutationElement> get_primordial(const string& name) {
        for (const auto& elem : primordial_elements) {
            if (elem->get_name() == name) {
                return elem;
            }
        }
        // If no element is found, you can return a nullptr or handle it as needed
        cout << "ERROR: No primordial element found with the name " << name << "." << endl;
        exit(1);
    }

    PieceSet scrambled_pieceset() const {
        vector<int> scrambled_indices;
        for (size_t i = 0; i < state.size(); ++i) {
            if (state[i] != static_cast<int>(i)) {
                scrambled_indices.push_back(i);
            }
        }
        return PieceSet(scrambled_indices);
    }

    double solvedness() const {
        size_t solved_count = 0;
        for (size_t i = 0; i < state.size(); ++i) {
            if (state[i] == static_cast<int>(i)) {
                ++solved_count;
            }
        }
        return static_cast<double>(solved_count) / state.size();
    }

    bool is_solved() const {
        return solvedness() == 1;
    }

    int size() const {
        return state.size();
    }

    void scramble() {
        // Seed the random number generator with the current time
        srand(static_cast<unsigned int>(time(nullptr)));

        for (int i = 0; i < 5000; i++) {
            apply_random_primordial();
        }
    }

private:
    vector<int> state;
    const vector<shared_ptr<PermutationElement>> primordial_elements;

    void apply_random_primordial() {
        int random_index = rand() % primordial_elements.size();
        apply(*primordial_elements[random_index]);
    }
};
