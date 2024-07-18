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

Puzzle create_rectangle_slider() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 4,
                          5, 6, 7, 8, 9,
                         10,11,12,13,14,
                         15,16,17,18,19,
                         20,21,22,23,24, };

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    vector<int> a{ 4, 0, 1, 2, 3,
                   5, 6, 7, 8, 9,
                  10,11,12,13,14,
                  15,16,17,18,19,
                  20,21,22,23,24, };
    elements.push_back(make_shared<PrimordialElement>("A", a));
    elements.push_back(make_shared<PrimordialElement>("A'", ~a));

    vector<int> b{ 0, 1, 2, 3, 4,
                   9, 5, 6, 7, 8,
                  10,11,12,13,14,
                  15,16,17,18,19,
                  20,21,22,23,24, };
    elements.push_back(make_shared<PrimordialElement>("B", b));
    elements.push_back(make_shared<PrimordialElement>("B'", ~b));

    vector<int> c{ 0, 1, 2, 3, 4,
                   5, 6, 7, 8, 9,
                  14,10,11,12,13,
                  15,16,17,18,19,
                  20,21,22,23,24, };
    elements.push_back(make_shared<PrimordialElement>("C", c));
    elements.push_back(make_shared<PrimordialElement>("C'", ~c));

    vector<int> d{ 0, 1, 2, 3, 4,
                   5, 6, 7, 8, 9,
                  10,11,12,13,14,
                  19,15,16,17,18,
                  20,21,22,23,24, };
    elements.push_back(make_shared<PrimordialElement>("D", d));
    elements.push_back(make_shared<PrimordialElement>("D'", ~d));

    vector<int> e{ 0, 1, 2, 3, 4,
                   5, 6, 7, 8, 9,
                  10,11,12,13,14,
                  15,16,17,18,19,
                  24,20,21,22,23, };
    elements.push_back(make_shared<PrimordialElement>("E", e));
    elements.push_back(make_shared<PrimordialElement>("E'", ~e));

    elements.push_back(make_shared<PrimordialElement>("T", vector<int>{ 0, 5,10,15,20,
                                                                        1, 6,11,16,21,
                                                                        2, 7,12,17,22,
                                                                        3, 8,13,18,23,
                                                                        4, 9,14,19,24, }));

    return Puzzle(state, elements);
}

Puzzle create_nonsense_puzzle_five_moves() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 4,
                          5, 6, 7, 8, 9,
                         10,11,12,13,14,
                         15,16,17,18,19,
                         20,21,22,23,24, };

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    vector<int> a{ 21, 14, 20, 24, 7, 10, 16, 0, 13, 8, 11, 9, 3, 18, 6, 15, 1, 19, 5, 4, 17, 12, 2, 23, 22 };
    elements.push_back(make_shared<PrimordialElement>("A", a));
    elements.push_back(make_shared<PrimordialElement>("A'", ~a));

    vector<int> b{ 16, 23, 13, 5, 10, 20, 3, 14, 8, 21, 11, 24, 4, 6, 19, 9, 15, 2, 17, 0, 12, 18, 7, 1, 22 };
    elements.push_back(make_shared<PrimordialElement>("B", b));
    elements.push_back(make_shared<PrimordialElement>("B'", ~b));

    vector<int> c{ 22, 5, 12, 9, 15, 13, 8, 4, 0, 7, 6, 23, 16, 21, 20, 10, 14, 2, 11, 1, 18, 17, 3, 24, 19 };
    elements.push_back(make_shared<PrimordialElement>("C", c));
    elements.push_back(make_shared<PrimordialElement>("C'", ~c));

    vector<int> d{ 8, 0, 11, 20, 10, 18, 2, 12, 1, 9, 3, 21, 15, 16, 13, 7, 22, 24, 6, 23, 14, 4, 17, 5, 19 };
    elements.push_back(make_shared<PrimordialElement>("D", d));
    elements.push_back(make_shared<PrimordialElement>("D'", ~d));

    vector<int> e{ 24, 17, 4, 20, 6, 13, 18, 15, 0, 14, 9, 19, 12, 21, 7, 23, 1, 22, 11, 16, 5, 8, 3, 2, 10 };
    elements.push_back(make_shared<PrimordialElement>("E", e));
    elements.push_back(make_shared<PrimordialElement>("E'", ~e));

    return Puzzle(state, elements);
}

Puzzle create_cheese_puzzle() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9,10,11, };

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    elements.push_back(make_shared<PrimordialElement>("T" , vector<int>{ 5, 0, 1, 2, 3, 4,
                                                                         6, 7, 8, 9,10,11, }));

    elements.push_back(make_shared<PrimordialElement>("T'", vector<int>{ 1, 2, 3, 4, 5, 0,
                                                                         6, 7, 8, 9,10,11, }));

    elements.push_back(make_shared<PrimordialElement>("A" , vector<int>{ 8, 7, 6, 3, 4, 5,
                                                                         2, 1, 0, 9,10,11, }));

    elements.push_back(make_shared<PrimordialElement>("B" , vector<int>{ 0, 9, 8, 7, 4, 5,
                                                                         6, 3, 2, 1,10,11, }));

    elements.push_back(make_shared<PrimordialElement>("C" , vector<int>{ 0, 1,10, 9, 8, 5,
                                                                         6, 7, 4, 3, 2,11, }));

    return Puzzle(state, elements);
}

Puzzle create_nonsense_puzzle_two_moves() {
    // Initial state of the puzzle
    vector<int> state = { 0, 1, 2, 3, 4,
                          5, 6, 7, 8, 9,
                         10,11,12,13,14,
                         15,16,17,18,19,
                         20,21,22,23,24, };

    // Primordial elements representing swaps
    vector<shared_ptr<PermutationElement>> elements;

    vector<int> a{ 5, 14, 0, 16, 20, 21, 9, 12, 10, 15, 8, 13, 1, 24, 23, 11, 19, 17, 7, 18, 4, 2, 3, 6, 22 };
    elements.push_back(make_shared<PrimordialElement>("A", a));
    elements.push_back(make_shared<PrimordialElement>("A'", ~a));

    vector<int> b{ 12, 4, 14, 18, 24, 21, 5, 17, 22, 8, 0, 15, 9, 11, 10, 16, 20, 3, 6, 7, 23, 2, 1, 13, 19 };
    elements.push_back(make_shared<PrimordialElement>("B", b));
    elements.push_back(make_shared<PrimordialElement>("B'", ~b));

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
    elements.push_back(make_shared<PrimordialElement>("U'", ~u));

    vector<int> l = {18, 1, 2,17, 
                      7, 4, 5, 6,
                      0, 9,10, 3,
                     12,13,14,15,
                     16,23,20,19,
                      8,21,22,11, };
    elements.push_back(make_shared<PrimordialElement>("L", l));
    elements.push_back(make_shared<PrimordialElement>("L'", ~l));

    vector<int> f = { 0, 1, 5, 6, 
                      4,20,21, 7,
                     11, 8, 9,10,
                      3,13,14, 2,
                     16,17,18,19,
                     15,12,22,23, };
    elements.push_back(make_shared<PrimordialElement>("F", f));
    elements.push_back(make_shared<PrimordialElement>("F'", ~f));

    vector<int> r = { 0, 9,10, 3, 
                      4, 5, 6, 7,
                      8,21,22,11,
                     15,12,13,14,
                      2,17,18, 1,
                     20,19,16,23, };
    elements.push_back(make_shared<PrimordialElement>("R", r));
    elements.push_back(make_shared<PrimordialElement>("R'", ~r));


    vector<int> r2l2 = r+r+l+l;
    vector<int> d = r2l2+u+r2l2;
    elements.push_back(make_shared<PrimordialElement>("D", d));
    elements.push_back(make_shared<PrimordialElement>("D'", ~d));
    vector<int> b = r2l2+f+r2l2;
    elements.push_back(make_shared<PrimordialElement>("B", b));
    elements.push_back(make_shared<PrimordialElement>("B'", ~b));

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
    elements.push_back(make_shared<PrimordialElement>("U'", ~u));

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
    elements.push_back(make_shared<PrimordialElement>("L'", ~l));

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
    elements.push_back(make_shared<PrimordialElement>("F'", ~f));

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
    elements.push_back(make_shared<PrimordialElement>("R'", ~r));

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
    elements.push_back(make_shared<PrimordialElement>("D'", ~d));

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
    elements.push_back(make_shared<PrimordialElement>("B'", ~b));



    return Puzzle(state, elements);
}

Puzzle create_swap_and_cycle_puzzle() {
    int size = 20;
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
        elements.push_back(make_shared<PrimordialElement>("C'", ~effect));
    }

    return Puzzle(state, elements);
}

Puzzle create_buffered_symmetric_group_puzzle() {
    int size = 20;
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
