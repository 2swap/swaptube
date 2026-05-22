#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"
#include <vector>

const int half_tape_length = 20;
const int maxt = -1;

struct RayState {
    Cuda::vec3 raypos;
    Cuda::vec3 raydir;
    Cuda::vec4 raycol;
    Cuda::vec4 cubcol;
    float cubalpha2;
    TuringMachine tm;
    int max_steps;

    int action[CODON_MEM_LIMIT] = {0};
    int states[CODON_MEM_LIMIT] = {2};
    int symbols[CODON_MEM_LIMIT] = {2};
    int haltcol[CODON_MEM_LIMIT];
    int pathlen = 0;

    Cuda::vec3 lower[2*CODON_MEM_LIMIT];
    Cuda::vec3 upper[2*CODON_MEM_LIMIT];
    int cplen = 0;

    int looppart;
    int face_id;
    Cuda::vec3 target;
    Cuda::vec3 startpos;
    float ray_thickness;
    Cuda::vec3 scale;
    float shell_border;
    float core_border;
    Cuda::vec3 highlight;
    float highlight_intensity;

    float ancestor_offset;

    bool init;
    int t;
    bool skip = false;
    bool print = false;
};

__device__ void raytrace_aligned(RayState& r, bool inside) {
    float dists[6] = {-1, -1, -1, -1, -1, -1};
    bool checkface[6] = {
        r.raydir.x != 0 && ((r.raydir.x < 0) == inside),
        r.raydir.x != 0 && ((r.raydir.x > 0) == inside),
        r.raydir.y != 0 && ((r.raydir.y < 0) == inside),
        r.raydir.y != 0 && ((r.raydir.y > 0) == inside),
        r.raydir.z != 0 && ((r.raydir.z < 0) == inside),
        r.raydir.z != 0 && ((r.raydir.z > 0) == inside)
    };
    if (checkface[0]) dists[0] = (r.lower[r.cplen - 1].x - r.raypos.x) / r.raydir.x;
    if (checkface[1]) dists[1] = (r.upper[r.cplen - 1].x - r.raypos.x) / r.raydir.x;
    if (checkface[2]) dists[2] = (r.lower[r.cplen - 1].y - r.raypos.y) / r.raydir.y;
    if (checkface[3]) dists[3] = (r.upper[r.cplen - 1].y - r.raypos.y) / r.raydir.y;
    if (checkface[4]) dists[4] = (r.lower[r.cplen - 1].z - r.raypos.z) / r.raydir.z;
    if (checkface[5]) dists[5] = (r.upper[r.cplen - 1].z - r.raypos.z) / r.raydir.z;
    float dist = -1;
    r.face_id = -1;
    for (int i=0; i<6; i++) {
        int axis = i / 2;
        Cuda::vec3 newpos = r.raypos + dists[i] * r.raydir;
        if (dists[i] >= 0 && ((newpos.x >= r.lower[r.cplen - 1].x && newpos.x <= r.upper[r.cplen - 1].x) || axis == 0) && ((newpos.y >= r.lower[r.cplen - 1].y && newpos.y <= r.upper[r.cplen - 1].y) || axis == 1) && ((newpos.z >= r.lower[r.cplen - 1].z && newpos.z <= r.upper[r.cplen - 1].z) || axis == 2)) {
            dist = dists[i];
            r.face_id = i;
	    break;
        }
    }
    if (r.face_id != -1) {
        r.raypos += dist * r.raydir;
    }
    Cuda::vec3 thingy = (r.raypos - (r.lower[r.cplen - 1] + r.upper[r.cplen - 1]) / 2) / (r.upper[r.cplen - 1] - r.lower[r.cplen - 1]);
    r.cubalpha2 = 2 * (thingy.x * thingy.x + thingy.y * thingy.y + thingy.z * thingy.z - 0.25);
}

__device__ void add_color_layer(RayState& r) {
    r.cubcol = Cuda::vec4(((r.haltcol[r.pathlen - 1] >> 24) & 0x000000ff) / 255.0f, ((r.haltcol[r.pathlen - 1] >> 16) & 0x000000ff) / 255.0f, ((r.haltcol[r.pathlen - 1] >> 8) & 0x000000ff) / 255.0f, (r.haltcol[r.pathlen - 1] & 0x000000ff) / 255.0f);

    int ca = 0;
    int nonca = r.cplen / 2;
    while (nonca > ca + 1) {
        int cid = (ca + nonca) / 2;
        bool targeted = (
            r.lower[2 * cid].x < r.target.x && r.target.x < r.upper[2 * cid].x &&
            r.lower[2 * cid].y < r.target.y && r.target.y < r.upper[2 * cid].y &&
            r.lower[2 * cid].z < r.target.z && r.target.z < r.upper[2 * cid].z
        );
        ca += (int)(targeted) * (cid - ca);
        nonca -= (int)(!targeted) * (nonca - cid);
    }
    float fractionca = nonca;
    if (nonca < r.cplen / 2) {
        Cuda::vec3 dists = Cuda::vec3(
            fmaxf(0.0f, fmaxf(r.lower[2 * nonca].x - r.target.x, r.target.x - r.upper[2 * nonca].x)),
            fmaxf(0.0f, fmaxf(r.lower[2 * nonca].y - r.target.y, r.target.y - r.upper[2 * nonca].y)),
            fmaxf(0.0f, fmaxf(r.lower[2 * nonca].z - r.target.z, r.target.z - r.upper[2 * nonca].z))
        );
        float distnonca = sqrtf(dists.x * dists.x + dists.y * dists.y + dists.z * dists.z);
        dists = Cuda::vec3(
            fmaxf(r.lower[2 * ca].x - r.target.x, r.target.x - r.upper[2 * ca].x),
            fmaxf(r.lower[2 * ca].y - r.target.y, r.target.y - r.upper[2 * ca].y),
            fmaxf(r.lower[2 * ca].z - r.target.z, r.target.z - r.upper[2 * ca].z)
        );
        float distca = sqrtf(dists.x * dists.x + dists.y * dists.y + dists.z * dists.z);
        fractionca += distca / (distca + distnonca);
    }

    float front_opacity = r.raycol.x;

    float depth_offset = 1.3f + r.cplen / 2 - fractionca;
    float c = atanf((fractionca - r.ancestor_offset - depth_offset * depth_offset) / 4.) / 1.57079632679f + 1;
    if (c < 0.01f && r.looppart == 4) r.skip = true;
    float back_opacity = r.cubcol.x * r.cubalpha2 * c * 0.5 /* (0.3f + (int)(nonca == r.cplen / 2))*/;

    float total_opacity = 1 - (1 - front_opacity) * (1 - back_opacity);
    float front_weight = front_opacity / total_opacity;
    Cuda::vec3 rgb_lerp = front_weight * Cuda::vec3(r.raycol.y, r.raycol.z, r.raycol.w) + (1 - front_weight) * Cuda::vec3(r.cubcol.y, r.cubcol.z, r.cubcol.w);
    r.raycol = Cuda::vec4(total_opacity, rgb_lerp.x, rgb_lerp.y, rgb_lerp.z);
}

__device__ void get_child_3d(RayState& r) {
    Cuda::vec3 child_size = (r.upper[r.cplen - 1] - r.lower[r.cplen - 1]) / Cuda::vec3(r.tm.num_states, r.tm.num_symbols, 2);

    /* ivecs don't exist yet and some things need to be changed anyway
    Cuda::ivec3 child_pos = floor((raypos - lower_xyz) / child_size);
    lower_xyz += child_size * child_pos;
    upper_xyz = lower_xyz + child_size;
    tm.left_right[action_index] = child_pos.z;
    tm.write_symbol[action_index] = child_pos.y;
    tm.next_state[action_index] = child_pos.x;
    tm.num_symbols += (int)(tm.write_symbol[action_index] == tm.num_symbols-1);
    tm.num_states += (int)(tm.next_state[action_index] == tm.num_states-1);
    */

    int child_x = min(r.tm.num_states-1, max(0, (int)(floor((r.raypos.x - r.lower[r.cplen - 1].x) / child_size.x))));
    int child_y = min(r.tm.num_symbols-1, max(0, (int)(floor((r.raypos.y - r.lower[r.cplen - 1].y) / child_size.y))));
    int child_z = min(1, max(0, (int)(floor((r.raypos.z - r.lower[r.cplen - 1].z) / child_size.z))));
    r.lower[r.cplen - 1] += child_size * Cuda::vec3(child_x, child_y, child_z);
    r.upper[r.cplen - 1] = r.lower[r.cplen - 1] + child_size;
    r.tm.left_right[r.action[r.pathlen - 1]] = (bool)(child_z);
    r.tm.write_symbol[r.action[r.pathlen - 1]] = child_y;
    r.tm.next_state[r.action[r.pathlen - 1]] = child_x;
    r.tm.num_symbols += (int)(r.tm.write_symbol[r.action[r.pathlen - 1]] == r.tm.num_symbols-1);
    r.tm.num_states += (int)(r.tm.next_state[r.action[r.pathlen - 1]] == r.tm.num_states-1);
}

__device__ void print_everything(RayState& r) {
    printf("Looppart: %d    Pos: (%f,%f,%f)    Cuboid: ((%f,%f,%f),(%f,%f,%f))    Color: (%f,%f,%f,%f)\n",
        r.looppart,
        r.raypos.x / r.scale.x, r.raypos.y / r.scale.y, r.raypos.z / r.scale.z,
        r.lower[r.cplen - 1].x / r.scale.x, r.lower[r.cplen - 1].y / r.scale.y, r.lower[r.cplen - 1].z / r.scale.z, r.upper[r.cplen - 1].x / r.scale.x, r.upper[r.cplen - 1].y / r.scale.y, r.upper[r.cplen - 1].z / r.scale.z,
        r.raycol.x, r.raycol.y, r.raycol.z, r.raycol.w
    );
}



/*
-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
the 5 parts of the loop:
1. raytrace out of shell, figure out next shell to enter if any (go to 2. if next shell exists else 5.)
2. raytrace to core (go to 3. if hit else 1.)
3. run TM, add info to path if halt (go to 4. if halt else 1.)
4. add color layer of face through which you entered, raytrace to the union of shells to figure out inner shell to enter, define transition if hit (go to 2. if hit else 5.)
5. raytrace out of core, add color layer of face through which you exited, undefine transition, and remove info from path (go to 1. if path is nonempty else break)

raypos changes only via raytracing
raydir doesn't change
raycol changes only via add_color_layer
cubcol doesn't change
face_id changes only via raytracing
looppart changes according to the if/else things as described above

cuboid path changes like this:
the last cuboid's index is cp.pathlen - 1
entering 1., the last cuboid in cp is the shell out of which we are raytracing
entering 2., the last cuboid in cp is the shell containing the core to which we are raytracing (the start of loop2 appends the core itself to the end)
entering 3., the last cuboid in cp is the core whose visibility we are checking (to which we just raytraced in 2.)
entering 4., the last cuboid in cp is the core whose visibility we just verified and whose face's color we are adding as a layer to raycol
entering 5., the last cuboid in cp is the core out of which we are raytracing

tm and its transition path changes like this:
the last transition's path index is p.pathlen - 1. the last transition's num_states and num_symbols are the ranges from which values are picked to define that transition
in 1., we change the last transition to the one corresponding to the core of the shell we're switching to. if necessary, we also increase the tm's (NOT the path's) num_states or num_symbols
in 2., we don't change anything. last transition is the one added by the core we are raytracing to, which was already set in 1. or 4.
in 3., in the halting case, we add the new transition to path (including its action index, num_states and num_symbols)
in 4., inside get_child, we define the new transition in the tm (and possibly increase num_states or num_symbols), whose info we just added to path in 3.
in 5., we remove the last transition from path and undefine it in the tm, and revert the tm's num_states and num_symbols
-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
*/



__device__ void loop1(RayState& r) {
    if (r.print) print_everything(r);

    raytrace_aligned(r, true);
    if (r.print && r.face_id == -1) printf("missed\n");

    if (r.face_id == 0) r.tm.next_state[r.action[r.pathlen - 1]]--;
    if (r.face_id == 1) r.tm.next_state[r.action[r.pathlen - 1]]++;
    if (r.face_id == 2) r.tm.write_symbol[r.action[r.pathlen - 1]]--;
    if (r.face_id == 3) r.tm.write_symbol[r.action[r.pathlen - 1]]++;
    bool should_leave_core = (r.tm.next_state[r.action[r.pathlen - 1]] < 0 || r.tm.next_state[r.action[r.pathlen - 1]] >= r.states[r.pathlen - 1] || r.tm.write_symbol[r.action[r.pathlen - 1]] < 0 || r.tm.write_symbol[r.action[r.pathlen - 1]] >= r.symbols[r.pathlen - 1]);
    if (r.face_id >= 4) {
        should_leave_core = (r.tm.left_right[r.action[r.pathlen - 1]] == (r.face_id == 5));
        r.tm.left_right[r.action[r.pathlen - 1]] = !r.tm.left_right[r.action[r.pathlen - 1]];
    }

    if (r.face_id != -1 && !should_leave_core) {
        int axis = r.face_id / 2;
        Cuda::vec3 diff = ((r.face_id & 1) * 2 - 1) * Cuda::vec3((int)(axis == 0), (int)(axis == 1), (int)(axis == 2)) * (r.upper[r.cplen - 1] - r.lower[r.cplen - 1]);
        r.lower[r.cplen - 1] += diff;
        r.upper[r.cplen - 1] += diff;
	if (r.tm.next_state[r.action[r.pathlen - 1]] == r.tm.num_states - 1) r.tm.num_states++;
	if (r.tm.write_symbol[r.action[r.pathlen - 1]] == r.tm.num_symbols - 1) r.tm.num_symbols++;
        r.looppart = 2;
    } else {
        r.cplen--;
        r.looppart = 5;
    }
}

__device__ void loop2(RayState& r) {
    if (r.print) print_everything(r);

    Cuda::vec3 border = r.shell_border * (r.upper[r.cplen - 1] - r.lower[r.cplen - 1]);
    r.lower[r.cplen] = r.lower[r.cplen - 1] + border;
    r.upper[r.cplen] = r.upper[r.cplen - 1] - border;
    if (!r.init) {
        r.cplen++;
        raytrace_aligned(r, false);
    } else {
        if (!(
          r.raypos.x >= r.lower[r.cplen].x && r.raypos.x <= r.upper[r.cplen].x &&
          r.raypos.y >= r.lower[r.cplen].y && r.raypos.y <= r.upper[r.cplen].y &&
          r.raypos.z >= r.lower[r.cplen].z && r.raypos.z <= r.upper[r.cplen].z
        )) {
            r.looppart = 0;
            return;
        }
        r.cplen++;
        r.looppart = 3;
        return;
    }
    Cuda::vec3 curray = r.raypos - r.startpos;
    Cuda::vec3 cubsize = r.upper[r.cplen - 1] - r.lower[r.cplen - 1];

    if (r.face_id == -1 || (cubsize.x + cubsize.y + cubsize.z) < 3 * r.ray_thickness * sqrtf(curray.x * curray.x + curray.y * curray.y + curray.z * curray.z)) {
        r.cplen--;
        r.looppart = 1;
    } else {
        r.looppart = 3;
    }
}

__device__ void loop3(RayState& r) {
    if (r.print) print_everything(r);

    int tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    int action_index;

    while (steps < r.max_steps) {
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

        current_state = r.tm.next_state[action_index];
        if(current_state == -1) {
            break;
        }
        tape[head_position] = r.tm.write_symbol[action_index];
        head_position += r.tm.left_right[action_index] ? 1 : -1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    bool halted = current_state == -1;

    r.skip = !halted && (
      r.highlight.x >= r.lower[r.cplen - 1].x && r.highlight.x <= r.upper[r.cplen - 1].x &&
      r.highlight.y >= r.lower[r.cplen - 1].y && r.highlight.y <= r.upper[r.cplen - 1].y &&
      r.highlight.z >= r.lower[r.cplen - 1].z && r.highlight.z <= r.upper[r.cplen - 1].z
    ); // using r.skip for highlighting, since they affect the flow of the program the same way
    
    if (!(halted || r.skip)) {
        r.cplen--;
        r.looppart = 1;
    } else {
        r.haltcol[r.pathlen] = d_rainbow(atan(steps / 4.0) / 1.57079632679f) - 0xd0000000;
        r.haltcol[r.pathlen] += (int)(r.skip) * (0x00ffffff - r.haltcol[r.pathlen] + ((int)(floor(r.highlight_intensity * 255.99)) << 24));
        //r.cubcol = Cuda::vec4(((r.haltcol[r.pathlen] >> 24) & 0x000000ff) / 255.0f, ((r.haltcol[r.pathlen] >> 16) & 0x000000ff) / 255.0f, ((r.haltcol[r.pathlen] >> 8) & 0x000000ff) / 255.0f, (r.haltcol[r.pathlen] & 0x000000ff) / 255.0f);
        r.action[r.pathlen] = action_index + (int)(r.skip)*(CODON_MEM_LIMIT - action_index - 1);
        r.states[r.pathlen] = r.tm.num_states;
        r.symbols[r.pathlen] = r.tm.num_symbols;
        r.pathlen++;
        r.looppart = 4;
        r.skip = false;
    }
}

__device__ void loop4(RayState& r) {
    if (r.print) print_everything(r);

    Cuda::vec3 border = (r.upper[r.cplen - 1] - r.lower[r.cplen - 1]) * r.core_border;
    r.lower[r.cplen] = r.lower[r.cplen - 1] + border;
    r.upper[r.cplen] = r.upper[r.cplen - 1] - border;
    if (!r.init) {
        add_color_layer(r);
        r.cplen++;
        raytrace_aligned(r, false);
    } else {
        if (r.skip || !(
          r.raypos.x >= r.lower[r.cplen].x && r.raypos.x <= r.upper[r.cplen].x &&
          r.raypos.y >= r.lower[r.cplen].y && r.raypos.y <= r.upper[r.cplen].y &&
          r.raypos.z >= r.lower[r.cplen].z && r.raypos.z <= r.upper[r.cplen].z
        )) {
            r.looppart = -2;
            return;
        }
        r.cplen++;
    }

    if (r.face_id == -1 || r.skip) {
        r.skip = false;
        r.cplen--;
        r.looppart = 5;
    } else {
        get_child_3d(r);
        r.looppart = 2;
    }
}

__device__ void loop5(RayState& r) {
    if (r.print) print_everything(r);

    raytrace_aligned(r, true);
    add_color_layer(r);
    r.pathlen--;
    r.tm.write_symbol[r.action[r.pathlen]] = 1;
    r.tm.left_right[r.action[r.pathlen]] = true;
    r.tm.next_state[r.action[r.pathlen]] = -1;
    r.tm.num_states = r.states[r.pathlen];
    r.tm.num_symbols = r.symbols[r.pathlen];
    r.cplen--;
    if (r.cplen > 0) {
        r.looppart = 1;
    } else {
        r.looppart = 0;
    }
}



__global__ void beaver_raytrace_kernel(unsigned int* pixels, int w, int h, Cuda::vec2 center, Cuda::vec3 raypos, Cuda::quat camera, float fov, int max_steps, float shell_border, float core_border, Cuda::vec3 scale, Cuda::vec3 target, float ancestor_offset, Cuda::vec3 highlight, float highlight_intensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w || idy >= h) {
        return;
    }
    int pixel_index = idy * w + idx;
    int pos_x = floor(idx + (0.5f - center.x) * w);
    int pos_y = floor(idy + (0.5f - center.y) * h);

    RayState r;
    //r.print = (idx == w / 2) && (idy == h / 2);
    //r.print = (idx == 80) && (idy == 20);
    //r.print = (idx == 358) && (idy == 178);

    r.raypos = /*Cuda::vec3(0)*/ raypos;
    r.startpos = /*Cuda::vec3(0)*/ raypos;
    r.max_steps = max_steps;
    r.target = /*target - raypos*/ Cuda::vec3(0);
    r.scale = scale;
    r.shell_border = shell_border;
    r.core_border = core_border;
    r.ancestor_offset = ancestor_offset;
    r.highlight = highlight - /*raypos*/ target;
    r.highlight_intensity = highlight_intensity;

    Cuda::vec2 wh(w, h);
    r.raydir = Cuda::get_raymarch_vector(Cuda::vec2(pos_x, pos_y), wh, fov, camera);
    Cuda::vec3 turn = Cuda::get_raymarch_vector(Cuda::vec2(pos_x+1, pos_y), wh, fov, camera) - r.raydir;
    r.ray_thickness = sqrtf(turn.x * turn.x + turn.y * turn.y + turn.z * turn.z);
    r.raycol = Cuda::vec4(0);
    r.cubcol = Cuda::vec4(0.1, 1, 1, 1);

    r.tm.num_states = 2;
    r.tm.num_symbols = 2;
    for (int i=0; i<CODON_MEM_LIMIT; i++) {
        r.tm.write_symbol[i] = 1;
        r.tm.left_right[i] = true;
        r.tm.next_state[i] = -1;
    }
    r.lower[0] = /*-raypos*/ -target;
    r.upper[0] = scale - /*raypos*/ target;
    r.cplen = 1;

    r.looppart = 3;
    r.init = true;
    r.face_id = -2;

    if (r.print) printf("\n");
    //if (r.print) printf("Ray Position: (%f,%f,%f)\n", r.raypos.x / scale.x, r.raypos.y / scale.y, r.raypos.z / scale.z);

    if (r.raypos.x >= r.lower[0].x && r.raypos.x <= r.upper[0].x && r.raypos.y >= r.lower[0].y && r.raypos.y <= r.upper[0].y && r.raypos.z >= r.lower[0].z && r.raypos.z <= r.upper[0].z) {
        while (r.looppart > 1) {
            if (r.looppart == 2) loop2(r);
            if (r.looppart == 3) loop3(r);
            if (r.looppart == 4) loop4(r);
        }
        r.looppart = 2 - r.looppart;
    } else {
        raytrace_aligned(r, false);
        r.looppart = 3;
    }
    r.init = false;
    if (r.print) printf("Init done!\n");
    if (r.face_id != -1) {
        r.t = 0;
        while (r.raycol.x < 0.9f && r.t != maxt) {
            if (r.looppart == 0) break;
            if (r.looppart == 1) loop1(r);
            if (r.looppart == 2) loop2(r);
            if (r.looppart == 3) loop3(r);
            if (r.looppart == 4) loop4(r);
            if (r.looppart == 5) loop5(r);
            r.t++;
        }
    }
    if (r.print) print_everything(r);

    /*float front_opacity = r.raycol.x;
    float back_opacity = 1;
    float total_opacity = 1;
    float front_weight = front_opacity;
    Cuda::vec3 rgb_lerp = front_weight * Cuda::vec3(r.raycol.y, r.raycol.z, r.raycol.w) + (1 - front_weight) * Cuda::vec3(1);
    r.raycol = Cuda::vec4(total_opacity, rgb_lerp.x, rgb_lerp.y, rgb_lerp.z);*/

    pixels[pixel_index] = ((unsigned int)(floor(r.raycol.x * 255)) << 24) | ((unsigned int)(floor(r.raycol.y * 255)) << 16) | ((unsigned int)(floor(r.raycol.z * 255)) << 8) | (unsigned int)(floor(r.raycol.w * 255));
    if (r.print) pixels[pixel_index] = 0xffff0000;
    //if (r.t == maxt) pixels[pixel_index] = 0xffffffff;
    //if (idx == 358 || idy == 178) pixels[pixel_index] = 0xffff00ff;
}

extern "C" void beaver_grid_TNF_3D_cuda(
    unsigned int* pixels,
    int w, int h, Cuda::vec2 center,
    float dist, Cuda::quat camera, float fov, Cuda::vec3 target,
    float shell_border, float core_border,
    Cuda::vec3 scale, float ancestor_offset,
    Cuda::vec3 highlight, float highlight_intensity,
    int max_steps
){
    unsigned int* d_pixels;

    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));
    dim3 threads(16, 16);
    dim3 block((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    //pos *= scale;
    target *= scale;
    highlight *= scale;
    Cuda::vec3 pos = /*target +*/ dist * rotate_vector(Cuda::vec3(0, 0, -1), conjugate(camera));

    beaver_raytrace_kernel<<<block, threads>>>(d_pixels, w, h, center, pos, normalize(camera), fov, max_steps, shell_border, core_border, scale, target, ancestor_offset, highlight, highlight_intensity);

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
}

/*
extern "C" void beaver_grid_TNF_3D_cuda(
    unsigned int* pixels,
    int w, int h,
    Cuda::vec3 pos, Cuda::quat camera, float fov,
    Cuda::vec3 target, Cuda::vec3 up, float use_quat_camera,
    float shell_border, float core_border,
    Cuda::vec3 scale, float ancestor_offset,
    int max_steps
){
    unsigned int* d_pixels;

    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));
    dim3 threads(16, 16);
    dim3 block((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    pos *= scale;
    target *= scale;
    Cuda::vec3 forward = target - pos;

    camera = normalize(use_quat_camera * normalize(camera) + (1 - use_quat_camera) * Cuda::get_quat(forward, up));

    beaver_raytrace_kernel<<<block, threads>>>(d_pixels, w, h, pos, camera, fov, max_steps, shell_border, core_border, scale, target, ancestor_offset);

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
}
*/
