#include "BeaverTNF3DScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

extern "C" void beaver_TNF_3D_cuda(unsigned int* pixels, int w, int h, vec2 center, quat camera, float fov, vec3 lower, vec3 upper, std::vector<int> action, TuringMachine tm, float brightness_offset, float color_source_depth, /*float ancestor_offset,*/ vec3 highlight, float highlight_intensity, int max_steps);

BeaverTNF3DScene::BeaverTNF3DScene(const vec2& dimension) {
    manager.set({
        {"fov", "1"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"camera_distance", "1"},
        {"center_x", "0.5"},
        {"center_y", "0.5"},
        {"max_steps", "0"},
        {"target_x", "0"},
        {"target_y", "0"},
        {"target_z", "0"},
        {"scale_x", "1"},
        {"scale_y", "1"},
        {"scale_z", "1"},
        {"ancestor_offset", "0"},
        {"highlight_x", "-1"},
        {"highlight_y", "-1"},
        {"highlight_z", "-1"},
        {"highlight_intensity", "0"},
        {"brightness_offset", "0"},
	{"color_source_depth", "0"},
        {"max_tnf_depth", "15"}
    });
}

int halt_trans(TuringMachine tm, int max_steps) {
    int half_tape_length = 20;
    int tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    int action_index;
    while (steps < max_steps) {
        // the transitions are indexed like this (but continued up to CODON_MEM_LIMIT-1):
        // 0  2  5  10
        // 1  3  7  12
        // 4  6  8  14
        // 9  11 13 15
        int action_layer = max(current_state, tape[head_position]) - 1;
        int action_side = (int)(current_state < tape[head_position]);
        action_index = action_layer * action_layer + 2 * (current_state + tape[head_position]) + action_side - 1;
        if (action_index >= CODON_MEM_LIMIT) {
            break;
        }

        current_state = tm.next_state[action_index];
        if (current_state == -1) {
            break;
        }
        tape[head_position] = tm.write_symbol[action_index];
        head_position += 2 * tm.left_right[action_index] - 1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    return (current_state == -1) ? action_index : -1;
}

void get_context(std::vector<int>& action, TuringMachine& tm, vec3& lower, vec3& upper, vec3& highlight, double x, double y, double z, int max_steps, int max_tnf_depth) {
    for (int i=0; i<CODON_MEM_LIMIT; i++) tm.next_state[i] = -1;
    ivec3 scale(1);
    if (0 <= x && x <= 1 && 0 <= y && y <= 1 && 0 <= z && z <= 1) {
        int states = 2;
        int symbols = 2;
        for (int i=0; i<max_tnf_depth; i++) {
            action.push_back(halt_trans(tm, max_steps));
            int a = action[action.size()-1];
            if (a == -1) break;
            scale *= ivec3(states, symbols, 2);
            tm.next_state[a] = min((double)(states-1), max((double)(0), floor(states * x)));
            x = states * x - tm.next_state[a];
            states += tm.next_state[a] == states - 1;
            tm.write_symbol[a] = min((double)(symbols-1), max((double)(0), floor(symbols * y)));
            y = symbols * y - tm.write_symbol[a];
            symbols += tm.write_symbol[a] == symbols - 1;
            tm.left_right[a] = min((double)(1), max((double)(0), floor(2 * z)));
            z = 2 * z - tm.left_right[a];
        }
        action.pop_back();
    }
    lower = vec3(-x, -y, -z) / scale;
    upper = vec3(1-x, 1-y, 1-z) / scale;
}

void BeaverTNF3DScene::draw() {
    vec3 target = vec3(state["target_x"], state["target_y"], state["target_z"]);
    quat camera = quat(state["q1"], state["qi"], state["qj"], state["qk"]);
    float dist = state["camera_distance"];
    vec3 scale = vec3(state["scale_x"], state["scale_y"], state["scale_z"]);
    vec3 dir = normalize(rotate_vector(vec3(0, 0, 1), camera) / scale);
    double x = (double)(target.x) - (double)(dist * dir.x);
    double y = (double)(target.y) - (double)(dist * dir.y);
    double z = (double)(target.z) - (double)(dist * dir.z);
    vec3 highlight = vec3(state["highlight_x"], state["highlight_y"], state["highlight_z"]) - vec3(x,y,z);

    std::vector<int> action;
    TuringMachine tm;
    vec3 lower;
    vec3 upper;
    get_context(action, tm, lower, upper, highlight, x, y, z, state["max_steps"], state["max_tnf_depth"]);
    /*for (int i=0; i<action.size(); i++) {
        printf("\nTransition %d: (%d,%d,%d)", action[i], tm.write_symbol[action[i]], tm.left_right[action[i]], tm.next_state[action[i]]);
    }
    printf("\nCuboid: ((%f,%f,%f),(%f,%f,%f))\n", lower.x, lower.y, lower.z, upper.x, upper.y, upper.z);*/
    beaver_TNF_3D_cuda(
        gpu_pix->get_ptr(), get_width(), get_height(), vec2(state["center_x"], state["center_y"]),
        camera, state["fov"],
        lower*scale, upper*scale,
        action, tm,
        state["brightness_offset"], state["color_source_depth"],
        //state["ancestor_offset"],
        highlight*scale, state["highlight_intensity"],
        state["max_steps"]
    );
}

const StateQuery BeaverTNF3DScene::populate_state_query() const {
    StateQuery sq = {"fov", "q1", "qi", "qj", "qk", "camera_distance", "center_x", "center_y", "max_steps", "max_tnf_depth", "target_x", "target_y", "target_z", "scale_x", "scale_y", "scale_z", "ancestor_offset", "highlight_x", "highlight_y", "highlight_z", "highlight_intensity", "brightness_offset", "color_source_depth"};
    return sq;
}

void BeaverTNF3DScene::mark_data_unchanged() { }
void BeaverTNF3DScene::change_data() { }
bool BeaverTNF3DScene::check_if_data_changed() const { return false; }
