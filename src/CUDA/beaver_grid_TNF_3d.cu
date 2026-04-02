#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/TuringMachine.h"
#include "color.cuh"
#include "common_graphics.cuh"
#include <vector>

const int half_tape_length = 20;

struct Path {
    int action[CODON_MEM_LIMIT] = {0};
    int states[CODON_MEM_LIMIT] = {2};
    int symbols[CODON_MEM_LIMIT] = {2};
    int pathlen = 0;
};

struct CuboidPath {
    Cuda::vec3 lower[2*CODON_MEM_LIMIT];
    Cuda::vec3 upper[2*CODON_MEM_LIMIT];
    int pathlen = 0;
};

__device__ int raytrace_aligned(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec3 lower_xyz, Cuda::vec3 upper_xyz, bool inside) {
    float dists[6] = {-1, -1, -1, -1, -1, -1};
    bool checkface[6] = {
        raydir.x != 0 && ((raydir.x < 0) == inside),
        raydir.x != 0 && ((raydir.x > 0) == inside),
        raydir.y != 0 && ((raydir.y < 0) == inside),
        raydir.y != 0 && ((raydir.y > 0) == inside),
        raydir.z != 0 && ((raydir.z < 0) == inside),
        raydir.z != 0 && ((raydir.z > 0) == inside)
    };
    if (checkface[0]) dists[0] = (lower_xyz.x - raypos.x) / raydir.x;
    if (checkface[1]) dists[1] = (upper_xyz.x - raypos.x) / raydir.x;
    if (checkface[2]) dists[2] = (lower_xyz.y - raypos.y) / raydir.y;
    if (checkface[3]) dists[3] = (upper_xyz.y - raypos.y) / raydir.y;
    if (checkface[4]) dists[4] = (lower_xyz.z - raypos.z) / raydir.z;
    if (checkface[5]) dists[5] = (upper_xyz.z - raypos.z) / raydir.z;
    float dist = -1;
    int face_id = -1;
    for (int i=0; i<6; i++) {
        int axis = i / 2;
        Cuda::vec3 newpos = raypos + dists[i] * raydir;
        if (dists[i] >= 0 && ((newpos.x >= lower_xyz.x && newpos.x <= upper_xyz.x) || axis == 0) && ((newpos.y >= lower_xyz.y && newpos.y <= upper_xyz.y) || axis == 1) && ((newpos.z >= lower_xyz.z && newpos.z <= upper_xyz.z) || axis == 2)) {
            dist = dists[i];
            face_id = i;
        }
    }
    if (face_id != -1) {
        raypos += dist * raydir;
    }
    return face_id;
}

__device__ void add_color_layer(Cuda::vec3 raypos, Cuda::vec4& raycol, Cuda::vec4 cubcol, Cuda::vec3 lower_xyz, Cuda::vec3 upper_xyz, int face_id, Cuda::vec3 target) {
    int axis = face_id / 2;
    float c = (lower_xyz.x <= target.x && target.x <= upper_xyz.x && lower_xyz.y <= target.y && target.y <= upper_xyz.y && lower_xyz.z <= target.z && target.z <= upper_xyz.z) ? 1 : 0.05;
    float front_opacity = raycol.x;
    float back_opacity = cubcol.x * c;
    float total_opacity = 1 - (1 - front_opacity) * (1 - back_opacity);
    float front_weight = front_opacity / total_opacity;
    Cuda::vec3 facecol = Cuda::vec3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    Cuda::vec3 rgb_lerp = front_weight * Cuda::vec3(raycol.y, raycol.z, raycol.w) + (1 - front_weight) * Cuda::vec3(cubcol.y, cubcol.z, cubcol.w) * facecol;
    raycol = Cuda::vec4(total_opacity, rgb_lerp.x, rgb_lerp.y, rgb_lerp.z);
}

__device__ void get_child_3d(TuringMachine& tm, int action_index, const Cuda::vec3& raypos, Cuda::vec3& lower_xyz, Cuda::vec3& upper_xyz) {
    Cuda::vec3 child_size = (upper_xyz - lower_xyz) / Cuda::vec3(tm.num_states, tm.num_symbols, 2);

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

    int child_x = min(tm.num_states-1, max(0, (int)(floor((raypos.x - lower_xyz.x) / child_size.x))));
    int child_y = min(tm.num_symbols-1, max(0, (int)(floor((raypos.y - lower_xyz.y) / child_size.y))));
    int child_z = min(1, max(0, (int)(floor((raypos.z - lower_xyz.z) / child_size.z))));
    lower_xyz += child_size * Cuda::vec3(child_x, child_y, child_z);
    upper_xyz = lower_xyz + child_size;
    tm.left_right[action_index] = (bool)(child_z);
    tm.write_symbol[action_index] = child_y;
    tm.next_state[action_index] = child_x;
    tm.num_symbols += (int)(tm.write_symbol[action_index] == tm.num_symbols-1);
    tm.num_states += (int)(tm.next_state[action_index] == tm.num_states-1);
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



__device__ void loop1(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec4& raycol, Cuda::vec4& cubcol, TuringMachine& tm, int max_steps, Path& p, CuboidPath& cp, int& looppart, int& face_id, Cuda::vec3 target) {
    face_id = raytrace_aligned(raypos, raydir, cp.lower[cp.pathlen - 1], cp.upper[cp.pathlen - 1], true);

    if (face_id == 0) tm.next_state[p.action[p.pathlen - 1]]--;
    if (face_id == 1) tm.next_state[p.action[p.pathlen - 1]]++;
    if (face_id == 2) tm.write_symbol[p.action[p.pathlen - 1]]--;
    if (face_id == 3) tm.write_symbol[p.action[p.pathlen - 1]]++;
    bool should_leave_core = (tm.next_state[p.action[p.pathlen - 1]] < 0 || tm.next_state[p.action[p.pathlen - 1]] >= p.states[p.pathlen - 1] || tm.write_symbol[p.action[p.pathlen - 1]] < 0 || tm.write_symbol[p.action[p.pathlen - 1]] >= p.symbols[p.pathlen - 1]);
    if (face_id >= 4) {
        should_leave_core = (tm.left_right[p.action[p.pathlen - 1]] == (face_id == 5));
        tm.left_right[p.action[p.pathlen - 1]] = !tm.left_right[p.action[p.pathlen - 1]];
    }

    if (!should_leave_core) {
        int axis = face_id / 2;
        Cuda::vec3 diff = ((face_id & 1) * 2 - 1) * Cuda::vec3((int)(axis == 0), (int)(axis == 1), (int)(axis == 2)) * (cp.upper[cp.pathlen - 1] - cp.lower[cp.pathlen - 1]);
        cp.lower[cp.pathlen - 1] += diff;
        cp.upper[cp.pathlen - 1] += diff;
	if (tm.next_state[p.action[p.pathlen - 1]] == tm.num_states - 1) tm.num_states++;
	if (tm.write_symbol[p.action[p.pathlen - 1]] == tm.num_symbols - 1) tm.num_symbols++;
        looppart = 2;
    } else {
        cp.pathlen--;
        looppart = 5;
    }
}

__device__ void loop2(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec4& raycol, Cuda::vec4& cubcol, TuringMachine& tm, int max_steps, Path& p, CuboidPath& cp, int& looppart, int& face_id, Cuda::vec3 target, float shell_border, bool init = false) {
    Cuda::vec3 border = shell_border * (cp.upper[cp.pathlen - 1] - cp.lower[cp.pathlen - 1]);
    cp.lower[cp.pathlen] = cp.lower[cp.pathlen - 1] + border;
    cp.upper[cp.pathlen] = cp.upper[cp.pathlen - 1] - border;
    if (!init) {
        face_id = raytrace_aligned(raypos, raydir, cp.lower[cp.pathlen], cp.upper[cp.pathlen], false);
    } else {
        if (!(raypos.x >= cp.lower[cp.pathlen].x && raypos.x <= cp.upper[cp.pathlen].x && raypos.y >= cp.lower[cp.pathlen].y && raypos.y <= cp.upper[cp.pathlen].y && raypos.z >= cp.lower[cp.pathlen].z && raypos.z <= cp.upper[cp.pathlen].z)) {
            looppart = 0;
            return;
        }
    }

    if (face_id == -1) {
        looppart = 1;
    } else {
        cp.pathlen++;
        looppart = 3;
    }
}

__device__ void loop3(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec4& raycol, Cuda::vec4& cubcol, TuringMachine& tm, int max_steps, Path& p, CuboidPath& cp, int& looppart, int& face_id, Cuda::vec3 target, bool init = false) {
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
        if(current_state == -1) {
            break;
        }
        tape[head_position] = tm.write_symbol[action_index];
        head_position += tm.left_right[action_index] ? 1 : -1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    bool halted = current_state == -1;
    
    if (!halted) {
        cp.pathlen--;
        looppart = 1;
    } else {
        p.action[p.pathlen] = action_index;
        p.states[p.pathlen] = tm.num_states;
        p.symbols[p.pathlen] = tm.num_symbols;
        p.pathlen++;
        looppart = 4;
    }
}

__device__ void loop4(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec4& raycol, Cuda::vec4& cubcol, TuringMachine& tm, int max_steps, Path& p, CuboidPath& cp, int& looppart, int& face_id, Cuda::vec3 target, float core_border, bool init = false) {
    Cuda::vec3 border = (cp.upper[cp.pathlen - 1] - cp.lower[cp.pathlen - 1]) * core_border;
    cp.lower[cp.pathlen] = cp.lower[cp.pathlen - 1] + border;
    cp.upper[cp.pathlen] = cp.upper[cp.pathlen - 1] - border;
    if (!init) {
        add_color_layer(raypos, raycol, cubcol, cp.lower[cp.pathlen - 1], cp.upper[cp.pathlen - 1], face_id, target);
        face_id = raytrace_aligned(raypos, raydir, cp.lower[cp.pathlen], cp.upper[cp.pathlen], false);
    } else {
        if (!(raypos.x >= cp.lower[cp.pathlen].x && raypos.x <= cp.upper[cp.pathlen].x && raypos.y >= cp.lower[cp.pathlen].y && raypos.y <= cp.upper[cp.pathlen].y && raypos.z >= cp.lower[cp.pathlen].z && raypos.z <= cp.upper[cp.pathlen].z)) {
            looppart = -2;
            return;
        }
    }

    if (face_id == -1) {
        looppart = 5;
    } else {
        cp.pathlen++;
        get_child_3d(tm, p.action[p.pathlen - 1], raypos, cp.lower[cp.pathlen - 1], cp.upper[cp.pathlen - 1]);
        looppart = 2;
    }
}

__device__ void loop5(Cuda::vec3& raypos, Cuda::vec3 raydir, Cuda::vec4& raycol, Cuda::vec4& cubcol, TuringMachine& tm, int max_steps, Path& p, CuboidPath& cp, int& looppart, int& face_id, Cuda::vec3 target) {
    face_id = raytrace_aligned(raypos, raydir, cp.lower[cp.pathlen - 1], cp.upper[cp.pathlen - 1], true);
    add_color_layer(raypos, raycol, cubcol, cp.lower[cp.pathlen - 1], cp.upper[cp.pathlen - 1], face_id, target);
    p.pathlen--;
    tm.write_symbol[p.action[p.pathlen]] = 1;
    tm.left_right[p.action[p.pathlen]] = true;
    tm.next_state[p.action[p.pathlen]] = -1;
    tm.num_states = p.states[p.pathlen];
    tm.num_symbols = p.symbols[p.pathlen];
    cp.pathlen--;
    if (cp.pathlen > 0) {
        looppart = 1;
    } else {
        looppart = 0;
    }
}



__global__ void beaver_raytrace_kernel(unsigned int* pixels, int w, int h, Cuda::vec3 raypos, Cuda::quat camera, float fov, int max_steps, float shell_border, float core_border, Cuda::vec3 scale, Cuda::vec3 target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w || idy >= h) {
        return;
    }
    int pixel_index = idy * w + idx;

    Cuda::vec3 raydir = Cuda::get_raymarch_vector(idx, idy, w, h, fov, camera);
    Cuda::vec4 raycol = Cuda::vec4(0);
    Cuda::vec4 cubcol = Cuda::vec4(0.1, 1, 1, 1);

    TuringMachine tm;
    tm.num_states = 2;
    tm.num_symbols = 2;
    for (int i=0; i<CODON_MEM_LIMIT; i++) {
        tm.write_symbol[i] = 1;
        tm.left_right[i] = true;
        tm.next_state[i] = -1;
    }
    Path p;
    CuboidPath cp;
    cp.lower[0] = Cuda::vec3(0);
    cp.upper[0] = scale;
    cp.pathlen = 1;

    int looppart = 3;
    bool init = true;
    int face_id = -2;
    if (raypos.x >= cp.lower[0].x && raypos.x <= cp.upper[0].x && raypos.y >= cp.lower[0].y && raypos.y <= cp.upper[0].y && raypos.z >= cp.lower[0].z && raypos.z <= cp.upper[0].z) {
        while (looppart > 1) {
            if (looppart == 2) loop2(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target, shell_border, init);
            if (looppart == 3) loop3(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target, init);
            if (looppart == 4) loop4(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target, core_border, init);
        }
        looppart = 2 - looppart;
    } else {
        face_id = raytrace_aligned(raypos, raydir, cp.lower[0], cp.upper[0], false);
        looppart = 3;
    }
    if (face_id != -1) {
        int t = 0;
        while (raycol.x < 0.998f && t < 50) {
            if (looppart == 0) break;
            if (looppart == 1) loop1(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target);
            if (looppart == 2) loop2(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target, shell_border);
            if (looppart == 3) loop3(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target);
            if (looppart == 4) loop4(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target, core_border);
            if (looppart == 5) loop5(raypos, raydir, raycol, cubcol, tm, max_steps, p, cp, looppart, face_id, target);
            t++;
        }
    }
    pixels[pixel_index] = ((unsigned int)(floor(raycol.x * 255)) << 24) | ((unsigned int)(floor(raycol.y * 255)) << 16) | ((unsigned int)(floor(raycol.z * 255)) << 8) | (unsigned int)(floor(raycol.w * 255));
}

extern "C" void beaver_grid_TNF_3D_cuda(
    unsigned int* pixels,
    int w, int h,
    Cuda::vec3 pos, Cuda::quat camera, float fov,
    Cuda::vec3 target, Cuda::vec3 up, float use_quat_camera,
    float shell_border, float core_border,
    Cuda::vec3 scale,
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

    beaver_raytrace_kernel<<<block, threads>>>(d_pixels, w, h, pos, camera, fov, max_steps, shell_border, core_border, scale, target);

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
}