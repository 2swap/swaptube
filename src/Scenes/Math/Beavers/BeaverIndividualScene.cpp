#include "BeaverIndividualScene.h"
#include "../../../IO/Writer.h"
#include "../../../IO/SFX.h"
#include <vector>

extern "C" void draw_individual_beaver(
    uint32_t* pixels, ivec2 wh, vec2 lx_ty, vec2 rx_by,
    uint32_t* grid, ivec2 grid_wh,
    uint32_t* icons, ivec2 icons_wh, int icons_len,
    TuringMachine tm, float iterations,
    float state_icon_scale, float vertical_step, float opacity_min, float opacity_dropoff,
    float dir_icon_scale, float current_tape_opacity, int rest,
    vec2 table_wh, vec2 table_wh0, float table_margin, float icon_border, float table_border, float table_glow
);

void boop(int state, int symbol, int num_states){
    int tone_number = state + symbol * num_states;
    vector<double> notes{12, 5, 17, 22, 21, 15, 9, 29, 25, 27, 19};
    double tone = pow(2,notes[tone_number]/12.);
    sfx_boink(get_global_state("t"), tone * 440, .05, 1);
}

BeaverIndividualScene::BeaverIndividualScene(const TuringMachine& tm, uint32_t* icons, ivec2& icons_wh, int& icons_len, const vec2& dimension)
: Scene(dimension), tm(tm), tape_length(31), tape(tape_length, 0), head_position(tape_length/2), icons(icons), icons_wh(icons_wh), icons_len(icons_len) {
    manager.set({
        // general simulation params
       	{"iterations", "0"},

        // spacetime diagram params
        {"state_icon_scale", "0"},
        {"vertical_step", "0"},
        {"opacity_min", "0"},
        {"opacity_dropoff", "0"},

        // current tape params
        {"dir_icon_scale", "0"},
        {"current_tape_opacity", "0"},
        {"sleep", "0"},

        // table params
        {"table_col_w", "0"},
        {"table_row_h", "0"},
        {"table_w0", "0"},
        {"table_h0", "0"},
        {"table_cell_margin", "0"},
        {"table_icon_border", "0"},
        {"table_border", "0"},
        {"table_line_glow", "0"},

        // camera/positioning params
        {"zoom", "0"},
        {"camera_x", "0"},
        {"camera_y", "0"},
    });
}

void BeaverIndividualScene::draw() {
    if (grid.size() == 0) {
        grid.resize(tape_length + grid.size());
        for (int i = 0; i < tape_length; i++) {
            grid[grid.size() - tape_length + i] = tape[i] << 16 | 0x0000ffff;
        }
        grid[grid.size() - tape_length + head_position] &= 0xffff0000 | current_state;
    }
    while (steps <= state["iterations"] && current_state != -1) {
        int ls = tape[head_position];
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        int action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

            boop(current_state, tape[head_position], tm.num_states);

        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        last_state = current_state;
        current_state = tm.next_state[action_index];

        if (current_state == -1) break; // halt
        steps++;

        // grid entry format: first 16 bits = symbol, last 16 bits = state
        grid.resize(tape_length + grid.size());
        for (int i = 0; i < tape_length; i++) {
            grid[grid.size() - tape_length + i] = tape[i] << 16 | 0x0000ffff;
        }
        grid[grid.size() - tape_length + head_position] &= 0xffff0000 | current_state;
    }

    ivec2 wh = get_width_height();
    ivec2 grid_wh = ivec2(tape_length, grid.size() / tape_length);
    float e = 2.718281828f;
    vec2 lx_ty(state["camera_x"] - wh.x / (2.0f * wh.y * pow(e, state["zoom"])), state["camera_y"] - 0.5f / pow(e, state["zoom"]));
    vec2 rx_by(state["camera_x"] + wh.x / (2.0f * wh.y * pow(e, state["zoom"])), state["camera_y"] + 0.5f / pow(e, state["zoom"]));
    draw_individual_beaver(
	gpu_pix->get_ptr(), wh, lx_ty, rx_by,
        grid.data(), ivec2(tape_length, grid.size() / tape_length),
        icons, icons_wh, icons_len,
        tm, fminf(state["iterations"], grid_wh.y - 1),
        state["state_icon_scale"], state["vertical_step"], state["opacity_min"], state["opacity_dropoff"],
        state["dir_icon_scale"], state["current_tape_opacity"], current_state == -1 ? 2 : (state["sleep"] > 0) - (state["sleep"] < 0) * (3 - last_state),
        vec2(state["table_col_w"], state["table_row_h"]), vec2(state["table_w0"], state["table_h0"]), state["table_cell_margin"], state["table_icon_border"], state["table_border"], state["table_line_glow"]
    );
}

const StateQuery BeaverIndividualScene::populate_state_query() const {
    StateQuery sq = {
	"iterations",
        "state_icon_scale", "vertical_step", "opacity_min", "opacity_dropoff",
        "dir_icon_scale", "current_tape_opacity", "sleep",
        "table_col_w", "table_row_h", "table_w0", "table_h0", "table_cell_margin", "table_icon_border", "table_border", "table_line_glow",
        "zoom", "camera_x", "camera_y"
    };
    return sq;
}

void BeaverIndividualScene::mark_data_unchanged() { }
void BeaverIndividualScene::change_data() { }
bool BeaverIndividualScene::check_if_data_changed() const { return false; }
