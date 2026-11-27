// Binary-symbol Turing machines, shared by the host and the GPU.
// Every machine in swaptube uses the symbols {0, 1}.

#pragma once
#include <cstdint>
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

// The transition for (state, symbol) lives at table index ("codon") 2*state + symbol.
// Each entry packs the whole transition into one int:
//     next_state * 4 + write_symbol * 2 + move   (move: 1 = right, 0 = left)
// next_state may be TM_HALT: the machine writes, moves, then halts.
// An entry of TM_UNDEFINED is a transition that has not been chosen.
inline constexpr int TM_CODONS = 60;      // room for 30 states
inline constexpr int TM_UNDEFINED = -1;
inline constexpr int TM_HALT = 31;        // a next_state value; larger than any real state

struct TuringMachine {
    int num_states = 0;
    int table[TM_CODONS];

    HOST_DEVICE TuringMachine() {
        for (int i = 0; i < TM_CODONS; i++) table[i] = TM_UNDEFINED;
    }
};

HOST_DEVICE inline int tm_codon(int state, int symbol) { return 2 * state + symbol; }
HOST_DEVICE inline int tm_pack(int next_state, int write_symbol, int move_right) { return next_state * 4 + write_symbol * 2 + move_right; }
HOST_DEVICE inline int tm_next_state(int tr) { return tr >> 2; }
HOST_DEVICE inline int tm_write_symbol(int tr) { return (tr >> 1) & 1; }
HOST_DEVICE inline int tm_move_right(int tr) { return tr & 1; }

HOST_DEVICE inline int tnf_slot_to_state(int slot, int S) { return slot == S - 1 ? TM_HALT : (slot == S ? S - 1 : slot); }
HOST_DEVICE inline int tnf_state_to_slot(int state, int S) { return state == TM_HALT ? S - 1 : (state == S - 1 ? S : state); }

// Results of tm_run other than a codon (which is always >= 0).
inline constexpr int TM_RAN_OUT = -1;   // out of steps, out of codons, or off the tape
inline constexpr int TM_HALTED = -2;    // reached a TM_HALT transition

// The tape tm_run uses: 64 cells, head starting mid-tape.
inline constexpr int TM_TAPE_LEN = 64;

// Runs the machine from a blank tape until one of:
//  - an undefined transition: returns its codon;
//  - a halting transition: returns TM_HALTED;
//  - otherwise TM_RAN_OUT.
HOST_DEVICE inline int tm_run(const int* table, int max_steps, int& steps) {
    uint64_t tape = 0;
    int head = TM_TAPE_LEN / 2;
    int state = 0;
    for (steps = 0; steps < max_steps; steps++) {
        const uint64_t mask = 1ull << head;
        const int codon = tm_codon(state, (tape & mask) != 0);
        if (codon >= TM_CODONS) break;
        const int tr = table[codon];
        if (tr == TM_UNDEFINED) return codon;
        if (tm_next_state(tr) == TM_HALT) return TM_HALTED;
        tape = tm_write_symbol(tr) ? (tape | mask) : (tape & ~mask);
        head += 2 * tm_move_right(tr) - 1;
        state = tm_next_state(tr);
        if (head < 0 || head >= TM_TAPE_LEN) break;
    }
    return TM_RAN_OUT;
}

SHARED_FILE_SUFFIX

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <stdexcept>
#include <string>

// Parses standard text notation, e.g. "1RB1LB_1LA0LC_1RZ1LD_1RD0RA": one
// underscore-separated group per state, with the transitions for symbols 0
// and 1 as (write, L/R, next state letter). 'Z' or '-' as the next state
// means halt; "---" means the transition is undefined.
inline TuringMachine tm_parse(const std::string& s) {
    if ((s.size() + 1) % 7 != 0) throw std::runtime_error("tm_parse: \"" + s + "\" is not groups of two 3-character transitions separated by '_'");
    TuringMachine tm;
    tm.num_states = (s.size() + 1) / 7;
    if (tm_codon(tm.num_states, 0) > TM_CODONS) throw std::runtime_error("tm_parse: \"" + s + "\" has too many states");
    for (int state = 0; state < tm.num_states; state++) {
        if (state > 0 && s[7 * state - 1] != '_') throw std::runtime_error("tm_parse: expected '_' between states in \"" + s + "\"");
        for (int symbol = 0; symbol < 2; symbol++) {
            const std::string t = s.substr(7 * state + 3 * symbol, 3);
            if (t == "---") continue;
            if (t[0] != '0' && t[0] != '1') throw std::runtime_error("tm_parse: write symbol in \"" + t + "\" is not 0 or 1");
            if (t[1] != 'L' && t[1] != 'R') throw std::runtime_error("tm_parse: move in \"" + t + "\" is not L or R");
            int next;
            if (t[2] == 'Z' || t[2] == '-') next = TM_HALT;
            else if (t[2] >= 'A' && t[2] - 'A' < tm.num_states) next = t[2] - 'A';
            else throw std::runtime_error("tm_parse: next state in \"" + t + "\" is not a state of \"" + s + "\"");
            tm.table[tm_codon(state, symbol)] = tm_pack(next, t[0] - '0', t[1] == 'R');
        }
    }
    return tm;
}
#endif
