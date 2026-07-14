#include <cuda_runtime.h>
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/TuringMachine.h"
#include "../color.cuh"
#include "../common_graphics.cuh"
#include "../edge_detect.cuh"
#include <vector>

extern "C" void cuda_overlay (
    uint32_t* background, const Cuda::ivec2& b_wh,
    const uint32_t* foreground, const Cuda::ivec2& f_wh,
    const Cuda::vec2& center, const float opacity, const float angle_rad);

const int MAX_TNF_DEPTH = 20;
const float INF = 10000000000000.0f;
const int half_tape_length = 20;

struct RayState {
    Cuda::vec3 dir;
    float len = 0;
    float thickness = 0;
    int face = -2;
    Cuda::vec3 highlight;
    float highlight_hit = 0;

    TuringMachine tm;
    int max_steps;
    int max_depth = MAX_TNF_DEPTH - 1;
    int halt_time_low = 0;
    int halt_time_high = 0;
    int low_time_depth = 0;

    int action[MAX_TNF_DEPTH];
    int states[MAX_TNF_DEPTH] = {2};
    int symbols[MAX_TNF_DEPTH] = {2};
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
    Cuda::vec3 child_size = (r.u[r.plen] - r.l[r.plen]) / Cuda::vec3(r.states[r.plen - 1], r.symbols[r.plen - 1], 2);
    Cuda::vec3 thingy = r.len * r.dir - r.l[r.plen];
    Cuda::ivec3 child = floor(thingy / child_size);
    child.x = min(r.states[r.plen - 1] - 1, max(0, child.x));
    child.y = min(r.symbols[r.plen - 1] - 1, max(0, child.y));
    child.z = min(1, max(0, child.z));
    //if (r.print) printf("Child: (%d,%d,%d)\n", child.x, child.y, child.z);
    r.l[r.plen] += child_size * child;
    r.u[r.plen] = r.l[r.plen] + child_size;
    r.tm.next_state[r.action[r.plen - 1]] = child.x;
    r.tm.write_symbol[r.action[r.plen - 1]] = child.y;
    r.tm.left_right[r.action[r.plen - 1]] = child.z;
    r.states[r.plen] = r.states[r.plen - 1] + (child.x == r.states[r.plen - 1] - 1);
    r.symbols[r.plen] = r.symbols[r.plen - 1] + (child.y == r.symbols[r.plen - 1] - 1);
}

__device__ void print_things(RayState& r) {
    printf("\nPath Length: %d\nRay Length: %f\nRay Pos: (%f,%f,%f)\nLast Block: ((%f,%f,%f),(%f,%f,%f))\nFace: %d\n", r.plen, r.len, r.len*r.dir.x, r.len*r.dir.y, r.len*r.dir.z, r.l[r.plen - 1].x, r.l[r.plen - 1].y, r.l[r.plen - 1].z, r.u[r.plen - 1].x, r.u[r.plen - 1].y, r.u[r.plen - 1].z, r.face);
    for (int i=0; i<r.plen; i++) printf("Cuboid %d: ((%f,%f,%f),(%f,%f,%f))\n", i, r.l[i].x, r.l[i].y, r.l[i].z, r.u[i].x, r.u[i].y, r.u[i].z);
    for (int i=0; i<r.plen-1; i++) printf("Transition %d: (%d,%d,%d)\n", r.action[i], r.tm.write_symbol[r.action[i]], r.tm.left_right[r.action[i]], r.tm.next_state[r.action[i]]);
}

__device__ void print_init(RayState& r) {
    printf("\nDirection: (%f,%f,%f)\n", r.dir.x, r.dir.y, r.dir.z);
    //for (int i=0; i<r.plen; i++) printf("Cuboid %d: ((%f,%f,%f),(%f,%f,%f))\n", i, r.l[i].x, r.l[i].y, r.l[i].z, r.u[i].x, r.u[i].y, r.u[i].z);
    //for (int i=0; i<r.plen-1; i++) printf("Transition %d: (%d,%d,%d)\n", r.action[i], r.tm.write_symbol[r.action[i]], r.tm.left_right[r.action[i]], r.tm.next_state[r.action[i]]);
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

    // run TM
    int tape[2 * half_tape_length + 1] = {0};
    int head_position = half_tape_length;
    int current_state = 0;
    int steps = 0;
    int action_index;
    r.halt_time_low = 0;
    r.halt_time_high = 0;
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
        r.halt_time_low += (r.halt_time_low == 0 && (current_state == -1 || (action_index == r.action[r.low_time_depth] && r.low_time_depth < r.plen - 1))) * (steps + 1);
        r.halt_time_high += (r.halt_time_high == 0 && (current_state == -1 || (action_index == r.action[r.low_time_depth + 1] && r.low_time_depth < r.plen - 2))) * (steps + 1);
        if (current_state == -1) {
            break;
        }
        tape[head_position] = r.tm.write_symbol[action_index];
        head_position += 2 * r.tm.left_right[action_index] - 1;
        if (head_position < 0 || head_position > 2 * half_tape_length) {
            break;
        }

        steps++;
    }
    bool halted = current_state == -1;

    //if (r.print) printf("\nHalted? %d\n", halted);

    // modify path
    if (halted) {
        r.action[r.plen - 1] = action_index;
        r.l[r.plen] = r.l[r.plen - 1];
        r.u[r.plen] = r.u[r.plen - 1];
        get_child_block(r);
        r.plen++;
    } else {
        r.highlight_hit += (int)(in_block(r.highlight, r.l[r.plen - 1], r.u[r.plen - 1]));
        raytrace_block(r, true);
        bool out = true;
        while (out) {
            r.plen--;
            if (r.plen == 0) {r.len = INF; return false;}
            r.tm.next_state[r.action[r.plen - 1]] += (r.face == 1) - (r.face == 0);
            r.tm.write_symbol[r.action[r.plen - 1]] += (r.face == 3) - (r.face == 2);
            int lr = r.tm.left_right[r.action[r.plen - 1]] + (r.face == 5) - (r.face == 4);
            out = r.tm.next_state[r.action[r.plen - 1]] < 0 || r.tm.next_state[r.action[r.plen - 1]] >= r.states[r.plen - 1] || r.tm.write_symbol[r.action[r.plen - 1]] < 0 || r.tm.write_symbol[r.action[r.plen - 1]] >= r.symbols[r.plen - 1] || lr < 0 || lr > 1;
            r.states[r.plen] += r.tm.next_state[r.action[r.plen - 1]] == r.states[r.plen - 1] - 1 && r.face == 1;
            r.symbols[r.plen] += r.tm.write_symbol[r.action[r.plen - 1]] == r.symbols[r.plen - 1] - 1 && r.face == 3;
            r.tm.left_right[r.action[r.plen - 1]] = lr == 1;
            r.tm.next_state[r.action[r.plen - 1]] -= out * (r.tm.next_state[r.action[r.plen - 1]] + 1);
        }
        Cuda::vec3 shift = (r.u[r.plen] - r.l[r.plen]) * Cuda::vec3((r.face == 1) - (r.face == 0), (r.face == 3) - (r.face == 2), (r.face == 5) - (r.face == 4));
        r.l[r.plen] += shift;
        r.u[r.plen] += shift;
        r.plen++;
    }

    return true;
}



__global__ void beaver_tnf_3D_kernel(unsigned int* pixels, unsigned int* highlight_pixels, float* depth_buffer, int w, int h, Cuda::vec2 center, Cuda::quat camera, float fov, Cuda::vec3 lower, Cuda::vec3 upper, int* action, int pathlen, TuringMachine tm, float brightness_offset, float color_source_depth, Cuda::vec3 highlight, float highlight_intensity, int max_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w || idy >= h) {
        return;
    }
    int pixel_index = idy * w + idx;
    int pos_x = floor(idx + (0.5f - center.x) * w);
    int pos_y = floor(idy + (0.5f - center.y) * h);

    RayState r;
    //r.print = idx == 313 && idy == 193;
    //r.print = idx == 64 && idy == 48;

    Cuda::ivec2 wh(w, h);
    r.dir = normalize(Cuda::get_raymarch_vector(Cuda::ivec2(pos_x, pos_y), wh, fov, camera));
    r.thickness = 20 * fov / (w + h);
    r.low_time_depth = floor(color_source_depth);
    r.highlight = highlight;
    r.max_steps = max_steps;

    for (int i=0; i<CODON_MEM_LIMIT; i++) r.tm.next_state[i] = -1;
    r.plen = pathlen + 1;
    for (int i=0; i<r.plen-1; i++) {
        r.tm.next_state[action[i]] = tm.next_state[action[i]];
        r.tm.write_symbol[action[i]] = tm.write_symbol[action[i]];
        r.tm.left_right[action[i]] = tm.left_right[action[i]];
        r.action[i] = action[i];
        r.states[i+1] = r.states[i] + (tm.next_state[action[i]] == r.states[i] - 1);
        r.symbols[i+1] = r.symbols[i] + (tm.write_symbol[action[i]] == r.symbols[i] - 1);
    }

    r.l[r.plen - 1] = lower;
    r.u[r.plen - 1] = upper;
    for (int i=r.plen-2; i>=0; i--) {
        r.l[i] = r.l[i+1] - (r.u[i+1] - r.l[i+1]) * Cuda::vec3(tm.next_state[action[i]], tm.write_symbol[action[i]], tm.left_right[action[i]]);
        r.u[i] = r.l[i] + (r.u[i+1] - r.l[i+1]) * Cuda::vec3(r.states[i], r.symbols[i], 2);
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
    float t = color_source_depth - r.low_time_depth;
    int haltcollow = Cuda::rainbow(atan(r.halt_time_low / 4.0f) / 1.57079632679f);
    int haltcolhigh = Cuda::rainbow(atan(r.halt_time_high / 4.0f) / 1.57079632679f);
    Cuda::vec3 collow = (b+0.25f)/1.25f*Cuda::vec3((haltcollow >> 16) & 255, (haltcollow >> 8) & 255, haltcollow & 255);
    Cuda::vec3 colhigh = (b+0.25f)/1.25f*Cuda::vec3((haltcolhigh >> 16) & 255, (haltcolhigh >> 8) & 255, haltcolhigh & 255);
    Cuda::vec3 col = collow * (1-t) + colhigh * t;
    //col += highlight_intensity * r.highlight_hit * (Cuda::vec3(255.5) - col);

    pixels[pixel_index] = (0xff000000 * (r.len < INF)) | ((unsigned int)(floor(col.x)) << 16) | ((unsigned int)(floor(col.y)) << 8) | (unsigned int)(floor(col.z));
    if (r.print) pixels[pixel_index] = 0xff00ff00;
    //if ((idx & idy & 3) == 3) pixels[pixel_index] = 0xffff00ff;
    //if ((idx & idy & 15) == 15) pixels[pixel_index] = 0xffe0e000;
    highlight_pixels[pixel_index] = ((unsigned int)(floor(highlight_intensity * r.highlight_hit * 255)) << 24) | 0x00ffffff;
    r.len = 5*sqrtf(sqrtf(r.len));
    depth_buffer[pixel_index] = r.len;
}

extern "C" void beaver_TNF_3D_cuda(
    unsigned int* pixels, int w, int h, Cuda::vec2 center,
    Cuda::quat camera, float fov,
    Cuda::vec3 lower, Cuda::vec3 upper,
    std::vector<int> action, TuringMachine tm,
    float brightness_offset, float color_source_depth,
    Cuda::vec3 highlight, float highlight_intensity,
    int max_steps
) {
    unsigned int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(unsigned int));
    unsigned int* d_highlight;
    cudaMalloc(&d_highlight, w * h * sizeof(unsigned int));
    float* d_depth_buffer;
    cudaMalloc(&d_depth_buffer, w * h * sizeof(float));
    int* d_action_path;
    cudaMalloc(&d_action_path, action.size() * sizeof(int));

    int action_path[action.size()];
    for (int i=0; i<action.size(); i++) action_path[i] = action[i];
    cudaMemcpy(d_action_path, action_path, action.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 block((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    beaver_tnf_3D_kernel<<<block, threads>>>(d_pixels, d_highlight, d_depth_buffer, w, h, center, normalize(camera), fov, lower, upper, d_action_path, action.size(), tm, brightness_offset, color_source_depth, highlight, highlight_intensity, max_steps);
    cuda_edge_detect(d_pixels, d_depth_buffer, Cuda::ivec2(w, h), 0xff000000);

    Cuda::ivec2 wh(w, h);
    cuda_overlay(d_pixels, wh, d_highlight, wh, Cuda::vec2(0,0), 1, 0);

    cudaMemcpy(pixels, d_pixels, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_highlight);
    cudaFree(d_depth_buffer);
    cudaFree(d_action_path);
}