const int CODON_MEM_LIMIT = 400;

struct TuringMachine {
    int num_symbols;
    int num_states;
    bool left_right[CODON_MEM_LIMIT];
    int write_symbol[CODON_MEM_LIMIT];
    int next_state[CODON_MEM_LIMIT];
};
