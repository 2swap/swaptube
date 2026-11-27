#include "BeaverTNF3DScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

extern "C" void beaver_TNF_3D_cuda(unsigned int* pixels, const ivec2& wh, vec2 center, quat camera, float fov, vec3 lower, vec3 upper, const int* action, int pathlen, TuringMachine tm, int max_steps, int max_depth, int axis_cycle_start, int warp_level, float warp_t);

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
        {"max_tnf_depth", "15"},
        {"axis_cycle_start", "0"}
    });
}

// Longest camera context path the kernel accepts: MAX_TNF_DEPTH - 2 in
// beaver_TNF_3d.cu (whose host wrapper throws if this is exceeded).
const int MAX_CONTEXT_PATH = 29;

// Runs tm from a blank tape. Returns the codon of the undefined transition it
// stops at, or -1 if it never reaches one (it halts, runs out of steps, or
// runs off the tape).
int halt_trans(const TuringMachine& tm, int max_steps) {
    int steps;
    const int codon = tm_run(tm.table, max_steps, steps);
    return codon >= 0 ? codon : -1;
}

// Finds the tree path from the root cube [0,1]^3 down to the cube containing
// the camera at (x, y, z), at most max_tnf_depth levels deep, and that cube's
// bounds relative to the camera.
void get_context(std::vector<int>& action, TuringMachine& tm, vec3& lower, vec3& upper, double x, double y, double z, int max_steps, int max_tnf_depth, int axis_cycle_start, int warp_level, double warp_t) {
    double p[3] = {x, y, z};          // camera position inside the current cube, per axis, in [0,1]
    double scale[3] = {1, 1, 1};      // root cube size / current cube size, per axis
    double offset[3] = {0, 0, 0};     // camera pre-image minus camera, in root units
    if (0 <= x && x <= 1 && 0 <= y && y <= 1 && 0 <= z && z <= 1) {
        int states = 2;
        const int max_path = min(max_tnf_depth, MAX_CONTEXT_PATH);
        while ((int)action.size() < max_path) {
            const int a = halt_trans(tm, max_steps);
            if (a == -1) break;
            const int depth = action.size();
            if (depth == warp_level) {
                // pre-image of p under q -> (1-t) q + t (q.y, q.z, q.x); see warp_inverse in beaver_TNF_3d.cu
                const double a = 1 - warp_t, b = warp_t, det = a * a * a + b * b * b;
                double pre[3];
                bool inside = true;
                for (int i = 0; i < 3; i++) {
                    pre[i] = (a * a * p[i] - a * b * p[(i + 1) % 3] + b * b * p[(i + 2) % 3]) / det;
                    inside = inside && 0 <= pre[i] && pre[i] <= 1;
                }
                if (!inside) break;
                for (int i = 0; i < 3; i++) {
                    offset[i] = (pre[i] - p[i]) / scale[i];
                    p[i] = pre[i];
                }
            }
            const int rot = depth < axis_cycle_start ? 0 : (depth - axis_cycle_start) % 3;   // depth - axis_cycle_start >= 0 here
            const int aW = rot, aM = (rot + 1) % 3, aS = (rot + 2) % 3;   // write, move, state axes (see axis_rotation in beaver_TNF_3d.cu)
            action.push_back(a);
            scale[aS] *= states + 1; scale[aW] *= 2; scale[aM] *= 2;
            const int slot = min(states, max(0, (int)floor((states + 1) * p[aS])));
            p[aS] = (states + 1) * p[aS] - slot;
            const int write_symbol = min(1, max(0, (int)floor(2 * p[aW])));
            p[aW] = 2 * p[aW] - write_symbol;
            const int move_right = min(1, max(0, (int)floor(2 * p[aM])));
            p[aM] = 2 * p[aM] - move_right;
            tm.table[a] = tm_pack(tnf_slot_to_state(slot, states), write_symbol, move_right);
            states += slot == states;
        }
    }
    lower = vec3(offset[0] - p[0] / scale[0], offset[1] - p[1] / scale[1], offset[2] - p[2] / scale[2]);
    upper = vec3(offset[0] + (1 - p[0]) / scale[0], offset[1] + (1 - p[1]) / scale[1], offset[2] + (1 - p[2]) / scale[2]);
}

void BeaverTNF3DScene::draw() {
    const double target[3] = {state["target_x"], state["target_y"], state["target_z"]};
    quat camera = quat(state["q1"], state["qi"], state["qj"], state["qk"]);
    float dist = state["camera_distance"];
    vec3 scale = vec3(state["scale_x"], state["scale_y"], state["scale_z"]);
    vec3 dir = normalize(rotate_vector(vec3(0, 0, 1), camera) / scale);
    double x = target[0] - (double)(dist * dir.x);
    double y = target[1] - (double)(dist * dir.y);
    double z = target[2] - (double)(dist * dir.z);

    // fractional axis roll interpolates between the rolls at floor and
    // floor + 1, which differ only from depth floor + 1 down.
    const double roll = state["axis_cycle_start"];
    const int axis_cycle_start = (int)floor(roll);
    const double warp_t = roll - axis_cycle_start;
    const int warp_level = warp_t > 0 && axis_cycle_start >= -1 ? axis_cycle_start + 1 : -1;

    std::vector<int> action;
    TuringMachine tm;
    vec3 lower;
    vec3 upper;
    get_context(action, tm, lower, upper, x, y, z, state["max_steps"], state["max_tnf_depth"], axis_cycle_start, warp_level, warp_t);
    /*for (int i=0; i<action.size(); i++) {
        printf("\nTransition %d: (%d,%d,%d)", action[i], tm_write_symbol(tm.table[action[i]]), tm_move_right(tm.table[action[i]]), tm_next_state(tm.table[action[i]]));
    }
    printf("\nCuboid: ((%f,%f,%f),(%f,%f,%f))\n", lower.x, lower.y, lower.z, upper.x, upper.y, upper.z);*/
    beaver_TNF_3D_cuda(
        gpu_pix.get_ptr(), get_width_height(), vec2(state["center_x"], state["center_y"]),
        camera, state["fov"],
        lower*scale, upper*scale,
        action.data(), action.size(), tm,
        state["max_steps"], state["max_tnf_depth"], axis_cycle_start, warp_level, warp_t
    );
}
