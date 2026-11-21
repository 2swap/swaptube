#include "BeaverTNF3DScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

extern "C" void beaver_TNF_3D_cuda(unsigned int* pixels, const ivec2& wh, vec2 center, quat camera, float fov, vec3 lower, vec3 upper, const int* action, int pathlen, TuringMachine tm, float brightness_offset, int max_steps, int max_depth);

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

// Longest camera context path the kernel accepts: MAX_TNF_DEPTH - 2 in
// beaver_TNF_3d.cu (whose host wrapper throws if this is exceeded).
const int MAX_CONTEXT_PATH = 18;

// Runs tm from a blank tape. Returns the codon of the undefined transition it
// stops at, or -1 if it never reaches one (it halts, runs out of steps, or
// runs off the tape).
int halt_trans(const TuringMachine& tm, int max_steps) {
    TuringMachineRun run;
    const int codon = tm_run(tm.table, run, max_steps);
    return codon >= 0 ? codon : -1;
}

// Finds the tree path from the root cube [0,1]^3 down to the cube containing
// the camera at (x, y, z), at most max_tnf_depth levels deep, and that cube's
// bounds relative to the camera.
void get_context(std::vector<int>& action, TuringMachine& tm, vec3& lower, vec3& upper, double x, double y, double z, int max_steps, int max_tnf_depth) {
    double p[3] = {x, y, z};          // camera position inside the current cube, per axis, in [0,1]
    double scale[3] = {1, 1, 1};      // root cube size / current cube size, per axis
    if (0 <= x && x <= 1 && 0 <= y && y <= 1 && 0 <= z && z <= 1) {
        int states = 2;
        const int max_path = min(max_tnf_depth, MAX_CONTEXT_PATH);
        while ((int)action.size() < max_path) {
            const int a = halt_trans(tm, max_steps);
            if (a == -1) break;
            action.push_back(a);
            // children are laid out as (next state, write symbol, move) along (x, y, z);
            // the state axis has states + 1 slots: existing states, then halt
            scale[0] *= states + 1; scale[1] *= 2; scale[2] *= 2;
            const int slot = min(states, max(0, (int)floor((states + 1) * p[0])));
            p[0] = (states + 1) * p[0] - slot;
            const int write_symbol = min(1, max(0, (int)floor(2 * p[1])));
            p[1] = 2 * p[1] - write_symbol;
            const int move_right = min(1, max(0, (int)floor(2 * p[2])));
            p[2] = 2 * p[2] - move_right;
            tm.table[a] = tm_pack(slot == states ? TM_HALT : slot, write_symbol, move_right);
            states += slot == states - 1;
        }
    }
    lower = vec3(-p[0] / scale[0], -p[1] / scale[1], -p[2] / scale[2]);
    upper = vec3((1 - p[0]) / scale[0], (1 - p[1]) / scale[1], (1 - p[2]) / scale[2]);
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
        state["max_steps"], state["max_tnf_depth"]
    );
}
