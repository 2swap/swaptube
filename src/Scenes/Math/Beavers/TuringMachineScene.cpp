#include "TuringMachineScene.h"
#include "../../../IO/Writer.h"
#include "../../../IO/SFX.h"
#include <vector>

extern "C" void draw_grid(uint32_t* pix, const ivec2& wh, const vec2& lx_ty, const vec2& rx_by, const uint32_t* grid, const ivec2& grid_wh, const vec2& grid_start);

void beep(int state, int symbol, int num_states){
    int tone_number = state + symbol * num_states;
    vector<double> notes{12, 5, 17, 22, 21, 15, 9, 29, 25, 27, 19};
    double tone = pow(2,notes[tone_number]/12.);
    sfx_boink(get_global_state("t"), tone * 440, .05, 1);
}

// For example, 1RB0LB_1LC0RE_1LA1LD_0LC---_0RB0RF_1RE1RB has 2 symbols and 5 states.
void parse_tm_from_string(char* s, int num_states, int num_symbols, TuringMachine& tm) {
    tm.num_symbols = num_symbols;
    tm.num_states = num_states;
    for(int state = 0; state < num_states; state++) {
        for(int symbol = 0; symbol < num_symbols; symbol++) {
            int action_layer = max(state, symbol) - 1;
            int action_side = (int)(state < symbol);
            int action_index = action_layer * action_layer + 2 * (state + symbol) + action_side - 1;
            if (action_index < CODON_MEM_LIMIT) {
                int string_index = state * (num_symbols * 3 + 1) + symbol * 3;
                tm.write_symbol[action_index] = s[string_index  ] - '0';
                tm.left_right  [action_index] = s[string_index+1] == 'R';
                char ns = s[string_index+2];
                if(ns == 'Z')
                    tm.next_state  [action_index] = -1;
                else
                    tm.next_state  [action_index] = ns - 'A';
            }
        }
    }
}

TuringMachineScene::TuringMachineScene(const TuringMachine& tm, const vec2& dimension)
: CoordinateScene(dimension), tm(tm), tape_length(301), tape(tape_length, 0), head_position(tape_length/2) {
    manager.set("iterations", "0");
    manager.set("ticks_opacity", "1");
    manager.set("zoom", "-1.5");
}

void TuringMachineScene::draw() {
    while (steps < state["iterations"]) {
        //int action_index = current_state * num_symbols + tape[head_position];
        int ls = tape[head_position];
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        int action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

	    beep(current_state, tape[head_position], tm.num_states);

        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        current_state = tm.next_state[action_index];

        if(current_state == -1) break; // halt
        steps++;

        grid.resize(tape_length + grid.size());
        for(int i = 0; i < tape_length; i++) {
            grid[grid.size() - tape_length + i] = tape[i] ? 0xffffffff : 0x00000000;
        }
        grid[grid.size() - tape_length + head_position] ^= 0x80ff0000;
    }

    const vec2 lx_ty(state["left_x"], state["top_y"]);
    const vec2 rx_by(state["right_x"], state["bottom_y"]);
    const vec2 grid_start(-tape_length/2., 0);
    draw_grid(gpu_pix->get_ptr(), get_width_height(), lx_ty, rx_by, grid.data(), ivec2(tape_length, (int)state["iterations"]), grid_start);

    CoordinateScene::draw();
}

const StateQuery TuringMachineScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, { "iterations" });
    return sq;
}
