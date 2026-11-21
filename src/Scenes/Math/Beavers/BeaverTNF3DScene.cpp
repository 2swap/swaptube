#include "BeaverTNF3DScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

extern "C" void beaver_TNF_3D_cuda(unsigned int* pixels, const ivec2& wh, vec2 center, quat camera, float fov, vec3 lower, vec3 upper, const int* action, int pathlen, TuringMachine tm, float brightness_offset, int max_steps);

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
        {"brightness_offset", "0"},
        {"max_tnf_depth", "15"}
    });
}

// Runs tm from a blank tape. Returns the codon of the undefined transition it
// stops at, or -1 if it never reaches one (out of steps or off the tape).
int halt_trans(const TuringMachine& tm, int max_steps) {
    TuringMachineRun run;
    const int codon = tm_run(tm.table, run, max_steps);
    return codon >= 0 ? codon : -1;
}

void get_context(std::vector<int>& action, TuringMachine& tm, vec3& lower, vec3& upper, double x, double y, double z, int max_steps, int max_tnf_depth) {
    ivec3 scale(1);
    if (0 <= x && x <= 1 && 0 <= y && y <= 1 && 0 <= z && z <= 1) {
        int states = 2;
        for (int i=0; i<max_tnf_depth; i++) {
            action.push_back(halt_trans(tm, max_steps));
            int a = action[action.size()-1];
            if (a == -1) break;
            // children are laid out as (next state, write symbol, move) along (x, y, z)
            scale *= ivec3(states, 2, 2);
            const int next_state = min((double)(states-1), max((double)(0), floor(states * x)));
            x = states * x - next_state;
            const int write_symbol = min(1.0, max(0.0, floor(2 * y)));
            y = 2 * y - write_symbol;
            const int move_right = min(1.0, max(0.0, floor(2 * z)));
            z = 2 * z - move_right;
            tm.table[a] = tm_pack(next_state, write_symbol, move_right);
            states += next_state == states - 1;
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

    std::vector<int> action;
    TuringMachine tm;
    vec3 lower;
    vec3 upper;
    get_context(action, tm, lower, upper, x, y, z, state["max_steps"], state["max_tnf_depth"]);
    /*for (int i=0; i<action.size(); i++) {
        printf("\nTransition %d: (%d,%d,%d)", action[i], tm_write_symbol(tm.table[action[i]]), tm_move_right(tm.table[action[i]]), tm_next_state(tm.table[action[i]]));
    }
    printf("\nCuboid: ((%f,%f,%f),(%f,%f,%f))\n", lower.x, lower.y, lower.z, upper.x, upper.y, upper.z);*/
    beaver_TNF_3D_cuda(
        gpu_pix.get_ptr(), get_width_height(), vec2(state["center_x"], state["center_y"]),
        camera, state["fov"],
        lower*scale, upper*scale,
        action.data(), action.size(), tm,
        state["brightness_offset"],
        state["max_steps"]
    );
}
