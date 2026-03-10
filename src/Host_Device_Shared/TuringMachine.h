struct TuringMachine {
    int num_symbols;
    int num_states;
    bool left_right[12];
    int write_symbol[12];
    int next_state[12];
};
