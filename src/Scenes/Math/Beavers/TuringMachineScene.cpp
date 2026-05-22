#include "TuringMachineScene.h"
#include "../../../IO/Writer.h"
#include "../../../IO/SFX.h"
#include <vector>

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
: CoordinateScene(dimension), tm(tm) {
    manager.set("iterations", "0");
    manager.set("ticks_opacity", "1");
    manager.set("zoom", "-1.5");
}

void TuringMachineScene::draw() {
    const int tape_length = 301;//iterations * 2 + 1;
    int iter_diffs = (int)state["iterations"] - last_iter;
    last_iter = state["iterations"];
    int tape[tape_length] = {0};
    int head_position = tape_length/2;
    int current_state = 0;
    int steps = -1;
    bool halted = false;
    int lowest_touched_index = head_position;
    int highest_touched_index = head_position;

    /*vec2 point0(head_position-tape_length/2. - .05, steps + .05);
    vec2 pix0 = point_to_pixel(point0);
    vec2 wh0 = point_to_pixel(point0 + vec2(1.1,-1.1)) - pix0;
    pix.fill_rect(pix0.x, pix0.y, wh0.x, wh0.y, 0xc0000000);*/
    vec2 point(head_position-tape_length/2. + .25, steps - .25);
    vec2 pix1 = point_to_pixel(point);
    vec2 wh = point_to_pixel(point + vec2(.5,-.5)) - pix1;
    pix.fill_rect(pix1.x, pix1.y, wh.x, wh.y, 0xff7f00ff);
    steps++;
    while (steps < state["iterations"]) {
        //int action_index = current_state * num_symbols + tape[head_position];
        int ls = tape[head_position];
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        int action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

        /*if (iter_diffs > 0 && steps >= state["iterations"] - 1)
	    beep(current_state, tape[head_position], tm.num_states);*/

        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        current_state = tm.next_state[action_index];
        lowest_touched_index = min(lowest_touched_index, head_position);
        highest_touched_index = max(highest_touched_index, head_position);

        if(current_state == -1) {
            halted = true;
            break;
        }
        for (int i = lowest_touched_index; i <= highest_touched_index; i++) {
	    /*vec2 point0(i-tape_length/2. - .05, steps + .05);
            vec2 pix0 = point_to_pixel(point0);
            vec2 wh0 = point_to_pixel(point0 + vec2(1.1,-1.1)) - pix0;
            pix.fill_rect(pix0.x, pix0.y, wh0.x, wh0.y, 0xc0000000);*/
            if(!tape[i]) continue;
            vec2 point(i-tape_length/2. + .1, steps-.1);
            vec2 pix1 = point_to_pixel(point);
            vec2 wh = point_to_pixel(point + vec2(.8,-.8)) - pix1;
            pix.fill_rect(pix1.x, pix1.y, wh.x, wh.y, 0xffffffff);
        }
        vec2 point(head_position-tape_length/2. + .25, steps - .25);
        vec2 pix1 = point_to_pixel(point);
        vec2 wh = point_to_pixel(point + vec2(.5,-.5)) - pix1;
        pix.fill_rect(pix1.x, pix1.y, wh.x, wh.y, 0xff7f00ff);
        steps++;
    }

    CoordinateScene::draw();
}

const StateQuery TuringMachineScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, { "iterations" });
    return sq;
}
