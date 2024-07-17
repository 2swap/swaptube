#pragma once

#include "PrimordialElement.cpp"
#include "Puzzle.cpp"

Puzzle create_s4_puzzle() {
    // Initial state of the puzzle
    vector<int> state = {0, 1, 2, 3};

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    // Swap 1 and 2
    elements.push_back(make_shared<PrimordialElement>("B", vector<int>{1, 0, 2, 3}));

    // Swap 1 and 3
    elements.push_back(make_shared<PrimordialElement>("C", vector<int>{2, 1, 0, 3}));

    // Swap 1 and 4
    elements.push_back(make_shared<PrimordialElement>("D", vector<int>{3, 1, 2, 0}));

    return Puzzle(state, elements);
}

Puzzle create_2x2x2() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 
                          4, 5, 6, 7,
                          8, 9,10,11,
                         12,13,14,15,
                         16,17,18,19,
                         20,21,22,23, };

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    vector<int> u = { 3, 0, 1, 2, 
                      8, 9, 6, 7,
                     12,13,10,11,
                     16,17,14,15,
                      4, 5,18,19,
                     20,21,22,23, };
    elements.push_back(make_shared<PrimordialElement>("U", u));

    vector<int> l = {18, 1, 2,17, 
                      7, 4, 5, 6,
                      0, 9,10, 3,
                     12,13,14,15,
                     16,23,20,19,
                      8,21,22,11, };
    elements.push_back(make_shared<PrimordialElement>("L", l));

    vector<int> f = { 0, 1, 5, 6, 
                      4,20,21, 7,
                     11, 8, 9,10,
                      3,13,14, 2,
                     16,17,18,19,
                     15,12,22,23, };
    elements.push_back(make_shared<PrimordialElement>("F", f));

    vector<int> r = { 0, 9,10, 3, 
                      4, 5, 6, 7,
                      8,21,22,11,
                     15,12,13,14,
                      2,17,18, 1,
                     20,19,16,23, };
    elements.push_back(make_shared<PrimordialElement>("R", r));

    vector<int> r2l2 = r+r+l+l;
    vector<int> d = r2l2+u+r2l2;
    elements.push_back(make_shared<PrimordialElement>("D", d));
    vector<int> b = r2l2+f+r2l2;
    elements.push_back(make_shared<PrimordialElement>("B", b));

    return Puzzle(state, elements);
}

Puzzle create_3x3x3() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 
                          4, 5, 6, 7,
                          8, 9,10,11,
                         12,13,14,15,
                         16,17,18,19,
                         20,21,22,23,

                         24,25,26,27,
                         28,29,30,31,
                         32,33,34,35,
                         36,37,38,39,
                         40,41,42,43,
                         44,45,46,47};

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    vector<int> u = { 3, 0, 1, 2, 
                      8, 9, 6, 7,
                     12,13,10,11,
                     16,17,14,15,
                      4, 5,18,19,
                     20,21,22,23,

                     27,24,25,26,
                     32,29,30,31,
                     36,33,34,35,
                     40,37,38,39,
                     28,41,42,43,
                     44,45,46,47, };
    elements.push_back(make_shared<PrimordialElement>("U", u));

    vector<int> l = {18, 1, 2,17, 
                      7, 4, 5, 6,
                      0, 9,10, 3,
                     12,13,14,15,
                     16,23,20,19,
                      8,21,22,11,

                     24,25,26,41,
                     31,28,29,30,
                     32,33,34,27,
                     36,37,38,39,
                     40,47,42,43,
                     44,45,46,35, };
    elements.push_back(make_shared<PrimordialElement>("L", l));

    vector<int> f = { 0, 1, 5, 6, 
                      4,20,21, 7,
                     11, 8, 9,10,
                      3,13,14, 2,
                     16,17,18,19,
                     15,12,22,23,

                     24,25,29,27,
                     28,44,30,31,
                     35,32,33,34,
                     36,37,38,26,
                     40,41,42,43,
                     39,45,46,47, };
    elements.push_back(make_shared<PrimordialElement>("F", f));

    vector<int> r = { 0, 9,10, 3, 
                      4, 5, 6, 7,
                      8,21,22,11,
                     15,12,13,14,
                      2,17,18, 1,
                     20,19,16,23,

                     24,33,26,27,
                     28,29,30,31,
                     32,45,34,35,
                     39,36,37,38,
                     40,41,42,25,
                     44,43,46,47, };
    elements.push_back(make_shared<PrimordialElement>("R", r));

    vector<int> d = { 0, 1, 2, 3,
                      4, 5,18,19,
                      8, 9, 6, 7,
                     12,13,10,11,
                     16,17,14,15,
                     23,20,21,22,

                     24,25,26,27,
                     28,29,42,31,
                     32,33,30,35,
                     36,37,34,39,
                     40,41,38,43,
                     47,44,45,46, };
    elements.push_back(make_shared<PrimordialElement>("D", d));

    vector<int> b = {13,14, 2, 3,
                      1, 5, 6, 0,
                      8, 9,10,11,
                     12,22,23,15,
                     19,16,17,18,
                     20,21, 7, 4,

                     37,25,26,27,
                     28,29,30,24,
                     32,33,34,35,
                     36,46,38,39,
                     43,40,41,42,
                     44,45,31,47};
    elements.push_back(make_shared<PrimordialElement>("B", b));



    return Puzzle(state, elements);
}

Puzzle create_swap_and_cycle_puzzle(int size) {
    // Initial state of the puzzle
    vector<int> state(size);
    for (int i = 0; i < size; ++i) {
        state[i] = i;
    }

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    { // Swap
        vector<int> effect(size);
        for (int j = 0; j < size; ++j) {
            effect[j] = j;
        }
        swap(effect[0], effect[1]);
        elements.push_back(make_shared<PrimordialElement>(string(1, 'S'), effect));
    }
    { // Cycle
        vector<int> effect(size);
        for (int j = 0; j < size; ++j) {
            effect[j] = (j+1)%size;
        }
        elements.push_back(make_shared<PrimordialElement>(string(1, 'C'), effect));
    }

    return Puzzle(state, elements);
}

Puzzle create_buffered_symmetric_group_puzzle(int size) {
    // Initial state of the puzzle
    vector<int> state(size);
    for (int i = 0; i < size; ++i) {
        state[i] = i;
    }

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    // Generate primordial elements for swaps between the first element and every other element
    char name = 'B';
    for (int i = 1; i < size; ++i) {
        vector<int> effect(size);
        for (int j = 0; j < size; ++j) {
            effect[j] = j;
        }
        swap(effect[0], effect[i]);
        elements.push_back(make_shared<PrimordialElement>(string(1, name), effect));
        name++;
    }

    return Puzzle(state, elements);
}
