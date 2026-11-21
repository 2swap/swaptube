#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"
#include <stdexcept>
#include <string>

const int MAX_TNF_DEPTH = 20;
const float INF = 10000000000000.0f;

// The camera's context path (transition indices from the root down to the cube
// containing the camera). Passed by value as a kernel argument, so no device
// allocation is needed per frame.
struct ActionPath {
    int a[MAX_TNF_DEPTH];
};

struct RayState {
    Cuda::vec3 dir;
    float len = 0;
    float thickness = 0;
    int face = -2;

    int table[Cuda::TM_CODONS];   // packed transitions of the machine being explored (see TuringMachine.h)
    int max_steps;
    int max_depth;
    int axis_cycle_start;
    bool hit = false;    // the ray ended on a halting (solid) cube
    int halt_time = 0;   // steps the hit machine ran before halting

    int action[MAX_TNF_DEPTH];
    int states[MAX_TNF_DEPTH] = {2};
    Cuda::vec3 l[MAX_TNF_DEPTH];   // lower corner
    Cuda::vec3 u[MAX_TNF_DEPTH];   // upper corner
    int plen;

    bool print = false;
};

__device__ bool in_block(Cuda::vec3 pos, Cuda::vec3 l, Cuda::vec3 u, int skip_axis = 0) {
    return (
        ((skip_axis & 1) == 1 || (l.x <= pos.x && pos.x <= u.x)) &&
        ((skip_axis & 2) == 2 || (l.y <= pos.y && pos.y <= u.y)) &&
        ((skip_axis & 4) == 4 || (l.z <= pos.z && pos.z <= u.z))
    );
}

__device__ void raytrace_block(RayState& r, bool inside) {
    //if (r.print) printf("Block: ((%f,%f,%f),(%f,%f,%f))\n", r.l[r.plen - 1].x, r.l[r.plen - 1].y, r.l[r.plen - 1].z, r.u[r.plen - 1].x, r.u[r.plen - 1].y, r.u[r.plen - 1].z);
    Cuda::vec3 whichface = Cuda::vec3((r.dir.x > 0) == inside, (r.dir.y > 0) == inside, (r.dir.z > 0) == inside);
    Cuda::vec3 dists = (r.l[r.plen-1] + whichface * (r.u[r.plen-1] - r.l[r.plen-1])) / r.dir;
    dists *= Cuda::vec3(dists.x >= 0, dists.y >= 0, dists.z >= 0);
    //if (r.print) printf("Dists: (%f,%f,%f)\n", dists.x, dists.y, dists.z);
    r.face = -1;
    r.face += (r.face == -1 && in_block(r.dir * dists.x, r.l[r.plen-1], r.u[r.plen-1], 1)) * (((r.dir.x > 0) == inside) + 1);
    r.face += (r.face == -1 && in_block(r.dir * dists.y, r.l[r.plen-1], r.u[r.plen-1], 2)) * (((r.dir.y > 0) == inside) + 3);
    r.face += (r.face == -1 && in_block(r.dir * dists.z, r.l[r.plen-1], r.u[r.plen-1], 4)) * (((r.dir.z > 0) == inside) + 5);
    Cuda::vec3 tolerance = (r.u[r.plen-1] - r.l[r.plen-1]) * 0.00001;
    r.face += (r.face == -1 && in_block(r.dir * dists.x, r.l[r.plen-1]-tolerance, r.u[r.plen-1]+tolerance, 1)) * (((r.dir.x > 0) == inside) + 1);
    r.face += (r.face == -1 && in_block(r.dir * dists.y, r.l[r.plen-1]-tolerance, r.u[r.plen-1]+tolerance, 2)) * (((r.dir.y > 0) == inside) + 3);
    r.face += (r.face == -1 && in_block(r.dir * dists.z, r.l[r.plen-1]-tolerance, r.u[r.plen-1]+tolerance, 4)) * (((r.dir.z > 0) == inside) + 5);
    r.len = (r.face >> 1 == 0) * fminf(INF,fmaxf(-INF,dists.x)) + (r.face >> 1 == 1) * fminf(INF,fmaxf(-INF,dists.y)) + (r.face >> 1 == 2) * fminf(INF,fmaxf(-INF,dists.z)) + (r.face == -1) * fminf(dists.x, fminf(dists.y, dists.z));
}

// Axis cycling. A branch at depth k (k = 0 for the root's split) lays its
// children out with the write symbol on axis rot, the move on axis (rot+1)%3
// and the next state on axis (rot+2)%3, where rot is 0 above depth
// axis_cycle_start and cycles 0, 1, 2, ... from there down.
__device__ int axis_rotation(int depth, int axis_cycle_start) {
    return depth < axis_cycle_start ? 0 : (depth - axis_cycle_start) % 3;
}

// The vector with w on axis rot, m on axis (rot+1)%3 and st on axis (rot+2)%3.
__device__ Cuda::vec3 place(int rot, float st, float w, float m) {
    if (rot == 0) return Cuda::vec3(w, m, st);
    if (rot == 1) return Cuda::vec3(st, w, m);
    return Cuda::vec3(m, st, w);
}

__device__ float axis_component(const Cuda::vec3& v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__device__ void get_child_block(RayState& r) {
    // children are laid out as (next state, write symbol, move) along the axes
    // given by this depth's rotation. The state axis has S+1 slots: the S
    // existing states (the last of which is new to this machine), then halt.
    const int rot = axis_rotation(r.plen - 1, r.axis_cycle_start);
    const int S = r.states[r.plen - 1];
    Cuda::vec3 child_size = (r.u[r.plen] - r.l[r.plen]) / place(rot, S + 1, 2, 2);
    Cuda::vec3 fc = (r.len * r.dir - r.l[r.plen]) / child_size;
    const int write_symbol = min(1, max(0, (int)floorf(axis_component(fc, rot))));
    const int move_right = min(1, max(0, (int)floorf(axis_component(fc, (rot + 1) % 3))));
    const int slot = min(S, max(0, (int)floorf(axis_component(fc, (rot + 2) % 3))));
    //if (r.print) printf("Child: (%d,%d,%d)\n", slot, write_symbol, move_right);
    r.l[r.plen] += child_size * place(rot, slot, write_symbol, move_right);
    r.u[r.plen] = r.l[r.plen] + child_size;
    r.table[r.action[r.plen - 1]] = Cuda::tm_pack(slot == S ? Cuda::TM_HALT : slot, write_symbol, move_right);
    r.states[r.plen] = S + (slot == S - 1);
}

__device__ void print_things(RayState& r) {
    printf("\nPath Length: %d\nRay Length: %f\nRay Pos: (%f,%f,%f)\nLast Block: ((%f,%f,%f),(%f,%f,%f))\nFace: %d\n", r.plen, r.len, r.len*r.dir.x, r.len*r.dir.y, r.len*r.dir.z, r.l[r.plen - 1].x, r.l[r.plen - 1].y, r.l[r.plen - 1].z, r.u[r.plen - 1].x, r.u[r.plen - 1].y, r.u[r.plen - 1].z, r.face);
    for (int i=0; i<r.plen; i++) printf("Cuboid %d: ((%f,%f,%f),(%f,%f,%f))\n", i, r.l[i].x, r.l[i].y, r.l[i].z, r.u[i].x, r.u[i].y, r.u[i].z);
    for (int i=0; i<r.plen-1; i++) printf("Transition %d: (%d,%d,%d)\n", r.action[i], Cuda::tm_write_symbol(r.table[r.action[i]]), Cuda::tm_move_right(r.table[r.action[i]]), Cuda::tm_next_state(r.table[r.action[i]]));
}

__device__ void print_init(RayState& r) {
    printf("\nDirection: (%f,%f,%f)\n", r.dir.x, r.dir.y, r.dir.z);
    //for (int i=0; i<r.plen; i++) printf("Cuboid %d: ((%f,%f,%f),(%f,%f,%f))\n", i, r.l[i].x, r.l[i].y, r.l[i].z, r.u[i].x, r.u[i].y, r.u[i].z);
}



// Leave the current cube through its exit face into the neighbouring cube,
// climbing up the tree while that neighbour lies outside the parent.
// Returns false if the ray leaves the whole fractal (or hits a numerical dead end).
__device__ bool step_out(RayState& r) {
    raytrace_block(r, true);
    if (r.face < 0) return false;
    bool out = true;
    while (out) {
        r.plen--;
        if (r.plen == 0) return false;
        // move to the sibling across the exit face by changing this level's transition.
        // On the state axis, slot S (one past the last state) is the halt slot.
        const int a = r.action[r.plen - 1];
        const int S = r.states[r.plen - 1];
        const int tr = r.table[a];
        const int slot = Cuda::tm_next_state(tr) == Cuda::TM_HALT ? S : Cuda::tm_next_state(tr);
        // which of (write, move, state) the crossed face's axis carries at this depth
        const int role = ((r.face >> 1) - axis_rotation(r.plen - 1, r.axis_cycle_start) + 3) % 3;
        const int sgn = (r.face & 1) ? 1 : -1;
        const int ws = Cuda::tm_write_symbol(tr) + (role == 0) * sgn;
        const int lr = Cuda::tm_move_right(tr) + (role == 1) * sgn;
        const int ns = slot + (role == 2) * sgn;
        out = ns < 0 || ns > S || ws < 0 || ws > 1 || lr < 0 || lr > 1;
        if (out) {
            // no sibling that way: undo this level's choice and keep climbing
            r.table[a] = Cuda::TM_UNDEFINED;
        } else {
            r.table[a] = Cuda::tm_pack(ns == S ? Cuda::TM_HALT : ns, ws, lr);
            r.states[r.plen] = S + (ns == S - 1);
        }
    }
    Cuda::vec3 shift = (r.u[r.plen] - r.l[r.plen]) * Cuda::vec3((r.face == 1) - (r.face == 0), (r.face == 3) - (r.face == 2), (r.face == 5) - (r.face == 4));
    r.l[r.plen] += shift;
    r.u[r.plen] += shift;
    r.plen++;
    return true;
}

// Flat shading up to which face was entered
__device__ float face_shade(int axis) {
    if (axis == 1) return 0.85f;
    if (axis == 2) return 0.7f;
    return 1.0f; // Camera in cube
}

// One step of the traversal, in the cube at the end of the current path.
// Returns true to keep going, false once the ray has hit a halting cube
// (r.hit) or left the fractal.
__device__ bool mainloop(RayState& r) {
    Cuda::vec3 cubsize = r.u[r.plen - 1] - r.l[r.plen - 1];
    const bool cut_off = r.plen >= r.max_depth || (cubsize.x + cubsize.y + cubsize.z) < 0.3 * r.thickness * r.len;
    if (!cut_off) {
        //if (r.print) print_things(r);
        Cuda::TuringMachineRun run;
        const int result = Cuda::tm_run(r.table, run, r.max_steps);
        if (result >= 0) {
            // undefined transition: branch on it, descending into the child the ray is in
            r.action[r.plen - 1] = result;
            r.l[r.plen] = r.l[r.plen - 1];
            r.u[r.plen] = r.u[r.plen - 1];
            get_child_block(r);
            r.plen++;
            return true;
        }
        if (result == Cuda::TM_HALTED) {
            // a halting machine is a solid cube
            r.hit = true;
            r.halt_time = run.steps + 1;
            return false;
        }
    }
    // non-halting and cut-off cubes are transparent: pass straight through
    return step_out(r);
}



__global__ void beaver_tnf_3D_kernel(unsigned int* pixels, const Cuda::ivec2 wh, Cuda::vec2 center, Cuda::quat camera, float fov, Cuda::vec3 lower, Cuda::vec3 upper, ActionPath path, int pathlen, Cuda::TuringMachine tm, int max_steps, int max_depth, int axis_cycle_start) {
    const int* action = path.a;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= wh.x || idy >= wh.y) {
        return;
    }
    int pixel_index = idy * wh.x + idx;
    int pos_x = floor(idx + (0.5f - center.x) * wh.x);
    int pos_y = floor(idy + (0.5f - center.y) * wh.y);

    RayState r;
    //r.print = idx == 313 && idy == 193;
    //r.print = idx == 64 && idy == 48;

    r.dir = normalize(Cuda::get_raymarch_vector(Cuda::ivec2(pos_x, pos_y), wh, fov, camera));
    r.thickness = 20 * fov / (wh.x + wh.y);
    r.max_steps = max_steps;
    r.max_depth = min(MAX_TNF_DEPTH - 1, max_depth);
    r.axis_cycle_start = axis_cycle_start;

    for (int i=0; i<Cuda::TM_CODONS; i++) r.table[i] = Cuda::TM_UNDEFINED;
    r.plen = pathlen + 1;
    for (int i=0; i<r.plen-1; i++) {
        r.table[action[i]] = tm.table[action[i]];
        r.action[i] = action[i];
        r.states[i+1] = r.states[i] + (Cuda::tm_next_state(tm.table[action[i]]) == r.states[i] - 1);
    }

    r.l[r.plen - 1] = lower;
    r.u[r.plen - 1] = upper;
    for (int i=r.plen-2; i>=0; i--) {
        const int tr = tm.table[action[i]];
        const int slot = Cuda::tm_next_state(tr) == Cuda::TM_HALT ? r.states[i] : Cuda::tm_next_state(tr);
        const int rot = axis_rotation(i, axis_cycle_start);
        r.l[i] = r.l[i+1] - (r.u[i+1] - r.l[i+1]) * place(rot, slot, Cuda::tm_write_symbol(tr), Cuda::tm_move_right(tr));
        r.u[i] = r.l[i] + (r.u[i+1] - r.l[i+1]) * place(rot, r.states[i] + 1, 2, 2);
    }

    //if (r.print) print_init(r);
    //if (r.print) print_things(r);
    bool started_inside = in_block(Cuda::vec3(0), lower, upper);
    if (!started_inside) raytrace_block(r, false);
    if (started_inside || r.face >= 0) {
        while (mainloop(r)) {}
    }

    // a miss leaves the pixel as the scene cleared it
    if (!r.hit) return;

    // color by the hit machine's halt time, shaded by which face was hit
    int haltcol = Cuda::rainbow(atan(r.halt_time / 4.0f) / 1.57079632679f);
    Cuda::vec3 col = face_shade(r.face >> 1) * Cuda::vec3((haltcol >> 16) & 255, (haltcol >> 8) & 255, haltcol & 255);

    pixels[pixel_index] = 0xff000000 | ((unsigned int)(floor(col.x)) << 16) | ((unsigned int)(floor(col.y)) << 8) | (unsigned int)(floor(col.z));
    if (r.print) pixels[pixel_index] = 0xff00ff00;
    //if ((idx & idy & 3) == 3) pixels[pixel_index] = 0xffff00ff;
    //if ((idx & idy & 15) == 15) pixels[pixel_index] = 0xffe0e000;
}

extern "C" void beaver_TNF_3D_cuda(
    unsigned int* d_pixels, const Cuda::ivec2& wh, Cuda::vec2 center,
    Cuda::quat camera, float fov,
    Cuda::vec3 lower, Cuda::vec3 upper,
    const int* action, int pathlen, Cuda::TuringMachine tm,
    int max_steps, int max_depth, int axis_cycle_start
) {
    if (pathlen > MAX_TNF_DEPTH - 2) {
        throw runtime_error("beaver_TNF_3D_cuda: context path of length " + to_string(pathlen) + " exceeds MAX_TNF_DEPTH - 2");
    }
    ActionPath path;
    for (int i=0; i<pathlen; i++) path.a[i] = action[i];

    dim3 threads(16, 16);
    dim3 block((wh.x + threads.x - 1) / threads.x, (wh.y + threads.y - 1) / threads.y);

    beaver_tnf_3D_kernel<<<block, threads>>>(d_pixels, wh, center, normalize(camera), fov, lower, upper, path, pathlen, tm, max_steps, max_depth, axis_cycle_start);
}
