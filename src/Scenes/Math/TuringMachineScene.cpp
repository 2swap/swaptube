#include "TuringMachineScene.h"

// For example, 1RB0LB_1LC0RE_1LA1LD_0LC---_0RB0RF_1RE1RB has 2 symbols and 5 states.
void parse_tm_from_string(char* s, int num_states, int num_symbols, TuringMachine& tm) {
    tm.num_symbols = num_symbols;
    tm.num_states = num_states;
    for(int state = 0; state < num_states; state++) {
        for(int symbol = 0; symbol < num_symbols; symbol++) {
            int action_index = symbol * num_states + state;
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

TuringMachineScene::TuringMachineScene(const TuringMachine& tm, const vec2& dimension)
: CoordinateScene(dimension), tm(tm) {
    manager.set("iterations", "0");
    manager.set("ticks_opacity", "1");
    manager.set("zoom", "-3");
}

void TuringMachineScene::draw() {
    const int tape_length = 301;//iterations * 2 + 1;
    int tape[tape_length] = {0};
    int head_position = tape_length/2;
    int current_state = 0;
    int steps = 0;
    bool halted = false;
    while (steps < state["iterations"]) {
        //int action_index = current_state * num_symbols + tape[head_position];
        int action_index = current_state + tm.num_states * tape[head_position];

        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        current_state = tm.next_state[action_index];
	cout << "Current state: " << current_state << endl;
	cout << tm.write_symbol[action_index] << (tm.left_right[action_index]?"R":"L") << (char)('A' + tm.next_state[action_index]) << endl;

        if(current_state == -1) {
            halted = true;
            break;
        }
        for (int i = 0; i < tape_length; i++) {
            vec2 point(i-tape_length/2., steps);
            vec2 pix1 = point_to_pixel(point);
            vec2 wh = point_to_pixel(point + vec2(1,-1)) - pix1;
            if(tape[i])
                pix.fill_rect(pix1.x, pix1.y, wh.x, wh.y, 0xffffffff);
        }
            vec2 point(head_position-tape_length/2., steps);
            vec2 pix1 = point_to_pixel(point);
            vec2 wh = point_to_pixel(point + vec2(1,-1)) - pix1;
                pix.fill_rect(pix1.x, pix1.y, wh.x, wh.y, 0xff00ff00);
        steps++;
    }

    CoordinateScene::draw();
}

const StateQuery TuringMachineScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, { "iterations" });
    return sq;
}

void TuringMachineScene::mark_data_unchanged() { }
void TuringMachineScene::change_data() { }
bool TuringMachineScene::check_if_data_changed() const { return false; }
