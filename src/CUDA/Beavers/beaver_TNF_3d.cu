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
    int max_depth = MAX_TNF_DEPTH - 1;
    int halt_time = 0;   // steps until the most recently run machine reached an undefined transition

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

__device__ void get_child_block(RayState& r) {
    // children are laid out as (next state, write symbol, move) along (x, y, z)
    Cuda::vec3 child_size = (r.u[r.plen] - r.l[r.plen]) / Cuda::vec3(r.states[r.plen - 1], 2, 2);
    Cuda::vec3 thingy = r.len * r.dir - r.l[r.plen];
    Cuda::ivec3 child = floor(thingy / child_size);
    child.x = min(r.states[r.plen - 1] - 1, max(0, child.x));
    child.y = min(1, max(0, child.y));
    child.z = min(1, max(0, child.z));
    //if (r.print) printf("Child: (%d,%d,%d)\n", child.x, child.y, child.z);
    r.l[r.plen] += child_size * child;
    r.u[r.plen] = r.l[r.plen] + child_size;
    r.table[r.action[r.plen - 1]] = Cuda::tm_pack(child.x, child.y, child.z);
    r.states[r.plen] = r.states[r.plen - 1] + (child.x == r.states[r.plen - 1] - 1);
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



/*
-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------



the loop:

- run TM
- raytrace to the block whose TM you just ran, from the outside if the TM halted and from the inside if the TM didn't halt
- use raytracing info to find next block to enter

alternatively:

- run TM
- if nonhalt, raytrace out of the block, use face id to find next block and modify path and TM accordingly
- if halt, use already computed ray position to find which child the ray passes through, modify path and TM accordingly

not sure if the branching is worth skipping the raytrace. idk exactly the mechanism through which branching loses time on gpu.



-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
*/



__device__ bool mainloop(RayState& r) {
    // end ray if it covers the whole block or it's past max depth
    Cuda::vec3 cubsize = r.u[r.plen - 1] - r.l[r.plen - 1];
    if (r.plen >= r.max_depth || (cubsize.x + cubsize.y + cubsize.z) < 0.3 * r.thickness * r.len || r.face == -1) return false;

    //if (r.print) print_things(r);

    // run TM from a blank tape until it reaches an undefined transition ("halts") or gives up
    Cuda::TuringMachineRun run;
    int action_index = Cuda::tm_run(r.table, run, r.max_steps);
    bool halted = action_index >= 0;   // >= 0 is a codon: the machine reached an undefined transition
    r.halt_time = halted ? run.steps + 1 : 0;

    //if (r.print) printf("\nHalted? %d\n", halted);

    // modify path
    if (halted) {
        r.action[r.plen - 1] = action_index;
        r.l[r.plen] = r.l[r.plen - 1];
        r.u[r.plen] = r.u[r.plen - 1];
        get_child_block(r);
        r.plen++;
    } else {
        raytrace_block(r, true);
        bool out = true;
        while (out) {
            r.plen--;
            if (r.plen == 0) {r.len = INF; return false;}
            // move to the sibling across the exit face by changing this level's transition
            const int a = r.action[r.plen - 1];
            const int S = r.states[r.plen - 1];
            const int tr = r.table[a];
            const int ns = Cuda::tm_next_state(tr) + (r.face == 1) - (r.face == 0);
            const int ws = Cuda::tm_write_symbol(tr) + (r.face == 3) - (r.face == 2);
            const int lr = Cuda::tm_move_right(tr) + (r.face == 5) - (r.face == 4);
            out = ns < 0 || ns >= S || ws < 0 || ws > 1 || lr < 0 || lr > 1;
            if (out) {
                // no sibling that way: undo this level's choice and keep climbing
                r.table[a] = Cuda::TM_UNDEFINED;
            } else {
                r.table[a] = Cuda::tm_pack(ns, ws, lr);
                r.states[r.plen] = S + (ns == S - 1);
            }
        }
        Cuda::vec3 shift = (r.u[r.plen] - r.l[r.plen]) * Cuda::vec3((r.face == 1) - (r.face == 0), (r.face == 3) - (r.face == 2), (r.face == 5) - (r.face == 4));
        r.l[r.plen] += shift;
        r.u[r.plen] += shift;
        r.plen++;
    }

    return true;
}



__global__ void beaver_tnf_3D_kernel(unsigned int* pixels, const Cuda::ivec2 wh, Cuda::vec2 center, Cuda::quat camera, float fov, Cuda::vec3 lower, Cuda::vec3 upper, ActionPath path, int pathlen, Cuda::TuringMachine tm, float brightness_offset, int max_steps) {
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
        r.l[i] = r.l[i+1] - (r.u[i+1] - r.l[i+1]) * Cuda::vec3(Cuda::tm_next_state(tr), Cuda::tm_write_symbol(tr), Cuda::tm_move_right(tr));
        r.u[i] = r.l[i] + (r.u[i+1] - r.l[i+1]) * Cuda::vec3(r.states[i], 2, 2);
    }

    //if (r.print) print_init(r);
    //if (r.print) print_things(r);
    bool started_inside = in_block(Cuda::vec3(0), lower, upper);
    if (!started_inside) raytrace_block(r, false);
    if (started_inside || r.face >= 0) {
        int time = 0;
        while (mainloop(r)) {
            //if (r.print) print_things(r);
            time++;
        }
    } else {
        r.len = INF;
    }

    float b = fmaxf(0,fminf(1, -logf(r.len)/3+brightness_offset));
    int haltcol = Cuda::rainbow(atan(r.halt_time / 4.0f) / 1.57079632679f);
    Cuda::vec3 col = (b+0.25f)/1.25f*Cuda::vec3((haltcol >> 16) & 255, (haltcol >> 8) & 255, haltcol & 255);

    pixels[pixel_index] = (0xff000000 * (r.len < INF)) | ((unsigned int)(floor(col.x)) << 16) | ((unsigned int)(floor(col.y)) << 8) | (unsigned int)(floor(col.z));
    if (r.print) pixels[pixel_index] = 0xff00ff00;
    //if ((idx & idy & 3) == 3) pixels[pixel_index] = 0xffff00ff;
    //if ((idx & idy & 15) == 15) pixels[pixel_index] = 0xffe0e000;
}

extern "C" void beaver_TNF_3D_cuda(
    unsigned int* d_pixels, const Cuda::ivec2& wh, Cuda::vec2 center,
    Cuda::quat camera, float fov,
    Cuda::vec3 lower, Cuda::vec3 upper,
    const int* action, int pathlen, Cuda::TuringMachine tm,
    float brightness_offset,
    int max_steps
) {
    if (pathlen > MAX_TNF_DEPTH - 2) {
        throw runtime_error("beaver_TNF_3D_cuda: context path of length " + to_string(pathlen) + " exceeds MAX_TNF_DEPTH - 2");
    }
    ActionPath path;
    for (int i=0; i<pathlen; i++) path.a[i] = action[i];

    dim3 threads(16, 16);
    dim3 block((wh.x + threads.x - 1) / threads.x, (wh.y + threads.y - 1) / threads.y);

    beaver_tnf_3D_kernel<<<block, threads>>>(d_pixels, wh, center, normalize(camera), fov, lower, upper, path, pathlen, tm, brightness_offset, max_steps);
}
