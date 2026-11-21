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

TuringMachineScene::TuringMachineScene(const TuringMachine& tm, const vec2& dimension)
: CoordinateScene(dimension), tm(tm), tape_length(301), tape(tape_length, 0), head_position(tape_length/2) {
    manager.set("iterations", "0");
    manager.set("ticks_opacity", "1");
    manager.set("zoom", "-1.5");
}

void TuringMachineScene::draw() {
    while (steps < state["iterations"] && current_state != -1) {
        const int codon = tm_codon(current_state, tape[head_position]);
        if (codon >= TM_CODONS) break;
        const int tr = tm.table[codon];

	    beep(current_state, tape[head_position], tm.num_states);

        // an undefined transition halts without writing; a halting one writes and moves first
        if (tr == TM_UNDEFINED) { current_state = -1; break; }
        tape[head_position] = tm_write_symbol(tr);
        head_position += 2 * tm_move_right(tr) - 1;
        current_state = tm_next_state(tr) == TM_HALT ? -1 : tm_next_state(tr);

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
    draw_grid(gpu_pix.get_ptr(), get_width_height(), lx_ty, rx_by, grid.data(), ivec2(tape_length, (int)state["iterations"]), grid_start);

    CoordinateScene::draw();
}
