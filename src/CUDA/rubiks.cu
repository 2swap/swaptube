#include "../Host_Device_Shared/vec.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <valarray>
#include "../Host_Device_Shared/ThreeDimensionStructs.h"
#include "color.cuh" // Contains overlay_pixel and set_pixel
#include "common_graphics.cuh" // Contains get_raymarch_vector

// TODO maybe when the pixel is transparent, we can check next intersection with the cube and draw the next sticker behind it, to give a more realistic look to the cube




__device__ __forceinline__ Cuda::vec3 rotate_vector(const Cuda::quat& q, const Cuda::vec3& v) {
    Cuda::vec3 q_vec(q.i, q.j, q.k);
    Cuda::vec3 t = Cuda::cross(q_vec, v) * 2.0f;
    return v + t * q.u + Cuda::cross(q_vec, t);
}




__device__ __forceinline__ bool ray_cube_intersect(
    const Cuda::vec3& ray_origin, const Cuda::vec3& ray_dir, Cuda::vec2& uv_coords, Cuda::vec3& hit_point, char& face_name) {

    Cuda::vec3 oc = ray_origin;
    const char faces[6] = {'L', 'R', 'D', 'U', 'B', 'F'}; // this should probably be outside of the function for efficiency
    // we will probably need to change the order of the faces to match the definition in Rubiks.cpp

    // normal vector for each face of the cube
    Cuda::vec3 normals[6] = {
        Cuda::vec3(-1, 0, 0),  // right
        Cuda::vec3(1, 0, 0), // left
        Cuda::vec3(0, -1, 0), // down
        Cuda::vec3(0, 1, 0),  // up
        Cuda::vec3(0, 0, 1), // back
        Cuda::vec3(0, 0, -1),   // front
    };

    // distance of each face from the origin
    float distances[6] = {1, 1, 1, 1, 1, 1};


    // dot product of ray direction and normal for each face
    float dot_products[6];
    for (int i = 0; i < 6; ++i) {
        dot_products[i] = Cuda::dot(ray_dir, normals[i]);
    }

    // find the intersection with each face if dot product is not zero (ray is not parallel to the face)
    float t_values[6];
    for (int i = 0; i < 6; ++i) {
        if (dot_products[i] != 0) {
            t_values[i] = (distances[i] - Cuda::dot(oc, normals[i])) / dot_products[i];
        } else {
            t_values[i] = -1; // no intersection
        }
    }

    // tolerance to account for floating point inaccuracies
    const float eps = 1e-3f; 

    // keep only t values that end up on a face of the cube
    for (int i = 0; i < 6; ++i) {
        if (t_values[i] > 0) {
            Cuda::vec3 hit_point_temp = oc + t_values[i] * ray_dir;
            if (hit_point_temp.x < -1.0f - eps || hit_point_temp.x > 1.0f + eps ||
                hit_point_temp.y < -1.0f - eps || hit_point_temp.y > 1.0f + eps ||
                hit_point_temp.z < -1.0f - eps || hit_point_temp.z > 1.0f + eps) {
                t_values[i] = -1; // intersection is outside the cube
            }
        }
    }

    // find the closest intersection
    // if all t_values are negative, there is no intersection
    if (t_values[0] < 0 && t_values[1] < 0 && t_values[2] < 0 &&
        t_values[3] < 0 && t_values[4] < 0 && t_values[5] < 0) {
        return false;
    }

    // take the smallest non negative value
    float min_t = 1e30f;
    int min_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (t_values[i] > 0 && t_values[i] < min_t) {
            min_t = t_values[i];
            min_index = i;
        }
    }

    if (min_index < 0) return false; //sanity check, should not happen

    face_name = '?';

    // determine which face was hit based on the index of the minimum t value
    if (min_index >= 0 && min_index < 6) {
        face_name = faces[min_index];
    } else {
        face_name = '?';
    }

    hit_point = oc + min_t * ray_dir;
    
    //uv coordinates for the face hit, normalized to [0, 1]
    float u = 0.0f, v = 0.0f;

    switch (min_index) { //might have to adjut inverted axes depending on the convention used for the cube's orientation
        case 0:
            u = (1.0f - hit_point.z) * 0.5f; 
            v = (1.0f - hit_point.y) * 0.5f; 
            break;
        case 1:
            u = (hit_point.z + 1.0f) * 0.5f; 
            v = (1.0f - hit_point.y) * 0.5f; 
            break;
        case 2:
            u = (1.0f - hit_point.x) * 0.5f; 
            v = (1.0f - hit_point.z) * 0.5f; 
            break;
        case 3:
            u = (1.0f - hit_point.x) * 0.5f; 
            v = (hit_point.z + 1.0f) * 0.5f; 
            break;
        case 4:
            u = (hit_point.x + 1.0f) * 0.5f; 
            v = (1.0f - hit_point.y) * 0.5f; 
            break;
        case 5:
            u = (1.0f - hit_point.x) * 0.5f; 
            v = (1.0f - hit_point.y) * 0.5f; 
            break;
    }

    // Secure the uv coordinates to be within [0, 1] range, with a small epsilon to avoid edge cases
    uv_coords.x = fmaxf(0.0f, fminf(1.0f - 1e-5f, u));
    uv_coords.y = fmaxf(0.0f, fminf(1.0f - 1e-5f, v));
    return true;
}


__global__ void render_cube_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    float geom_mean_size,
    const Cuda::quat camera_direction, const Cuda::vec3 camera_pos, float fov, 
    int cube_size, float turn_fraction, Cuda::quat rotation_quat, const Cuda::vec3 slice_axis, float slice_dist, char d_stickers[6][11][11])
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::ivec2 pixel(px, py);

    rotation_quat = normalize(Cuda::quat(1.0f, 0.0f, 0.0f, 0.0f) * (1-turn_fraction) + rotation_quat * turn_fraction);


    Cuda::vec3 ray_dir = get_raymarch_vector(pixel, wh, fov, conjugate(camera_direction));

    Cuda::vec2 uv;
    Cuda::vec3 hit_point;
    char face_name = '?';
    bool have_hit = false;
    int col = -1, row = -1;
    float best_dist = 1e30f;


    // static hit
    Cuda::vec2 uv_s; Cuda::vec3 hp_s; char face_s = '?'; bool some_collision = false;
    bool hit_s = ray_cube_intersect(camera_pos, ray_dir, uv_s, hp_s, face_s);
    if (hit_s) {
        float axis_pos = Cuda::dot(hp_s, slice_axis);
        if (axis_pos <= slice_dist) {
            Cuda::vec3 d = hp_s - camera_pos;
            some_collision = true;
            float dist = Cuda::dot(d, d);
            if (dist < best_dist) { // so the slice doesn't clip through the cube
                best_dist = dist;
                uv = uv_s; hit_point = hp_s; face_name = face_s;
                have_hit = true;
            }
        }
    }


    // moving hit
    Cuda::quat inv_rot = conjugate(rotation_quat);
    Cuda::vec3 rotated_origin = rotate_vector(inv_rot, camera_pos);
    Cuda::vec3 rotated_dir = rotate_vector(inv_rot, ray_dir);

    Cuda::vec2 uv_m; Cuda::vec3 hp_m; char face_m = '?';
    bool hit_m = ray_cube_intersect(rotated_origin, rotated_dir, uv_m, hp_m, face_m);
    if (hit_m) {
        float axis_pos_local = Cuda::dot(hp_m, slice_axis);
        if (axis_pos_local > slice_dist) {
            Cuda::vec3 d = hp_m - rotated_origin;
            some_collision = true;
            float dist = Cuda::dot(d, d);
            if (dist < best_dist) {
                best_dist = dist;
                uv = uv_m; hit_point = hp_m; face_name = face_m;
                have_hit = true;
            }
        }
    }

    if (!have_hit) {
        pixels[pixel.x + wh.x * pixel.y] = some_collision ? 0xFF000000 : 0x00000000;
        return;
    }


    

    switch (face_name) {
        case 'U': row = cube_size - 1 - (int)(uv.y * cube_size); col = cube_size - 1 - (int)(uv.x * cube_size); break;
        case 'L': row = (int)(uv.y * cube_size); col = (int)(uv.x * cube_size); break;
        case 'F': row = (int)(uv.y * cube_size); col = cube_size - 1 - (int)(uv.x * cube_size); break;
        case 'R': row = (int)(uv.y * cube_size); col = (int)(uv.x * cube_size); break;
        case 'B': row = (int)(uv.y * cube_size); col = cube_size - 1 - (int)(uv.x * cube_size); break;
        case 'D': row = cube_size - 1 - (int)(uv.y * cube_size); col = cube_size - 1 - (int)(uv.x * cube_size); break;
        default: row = (int)(uv.y * cube_size); col = (int)(uv.x * cube_size); break;
    }

    // change row and col, interpolating to make it work with all sizes under 11
    float temp_row = row * 10. / (cube_size - 1);
    float temp_col = col * 10. / (cube_size - 1);
    // round to nearest integer
    row = (int)(temp_row + 0.5f);
    col = (int)(temp_col + 0.5f);
    


    // coordinates inside the sticker, normalized to [0, 1]
    float local_u = fmodf(uv.x * cube_size, 1.0f);
    float local_v = fmodf(uv.y * cube_size, 1.0f);

    // normalize to [-1, 1] for the purpose of calculating the 8-norm
    float x = local_u * 2.0f - 1.0f;
    float y = local_v * 2.0f - 1.0f;

    // scale factor to slightly enlarge the coordinates, making the stickers appear smaller and more distinct
    float scale = 1.1f; 
    x *= scale;
    y *= scale;


    uint32_t default_color = 0xFF000000; // default color
    uint32_t plastic_color = 0xFF000000; // plastic color

    // Calculate the 8-norm of the coordinates to determine if the pixel is within the sticker's bounds
    float x2 = x * x; float x4 = x2 * x2; float x8 = x4 * x4;
    float y2 = y * y; float y4 = y2 * y2; float y8 = y4 * y4;

    if (x8 + y8 >= 1.0f) {
        pixels[pixel.x + wh.x * pixel.y] = plastic_color; // transparent plastic
        return;
    }

    uint32_t color = default_color;

    int i;

    switch (face_name) {
        case 'U': i = 0; break;
        case 'L': i = 1; break;
        case 'F': i = 2; break;
        case 'R': i = 3; break;
        case 'B': i = 4; break;
        case 'D': i = 5; break;
        
        default:  color = 0xFFC92AAC; break;
    }

    switch (d_stickers[i][row][col]) {
        case 'R': color = 0xFFC21D1D; break;
        case 'G': color = 0xFF1DC249; break;
        case 'B': color = 0xFF251BB3; break;
        case 'W': color = 0xFFFFFFFF; break;
        case 'Y': color = 0xFFDED82A; break;
        case 'O': color = 0xFFF7A31B; break;
    }

    pixels[pixel.x + wh.x * pixel.y] = color;
}

extern "C" void cuda_render_cube(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    float geom_mean_size,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov, float turn_fraction, Cuda::quat rotation_quat, Cuda::vec3 slice_plane, float slice_dist, char (*d_stickers)[6][11][11], int cube_size)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    render_cube_kernel<<<gridSize, blockSize>>>(
        d_pixels, wh,
        geom_mean_size,
        camera_direction, camera_pos, fov, cube_size, turn_fraction, rotation_quat, slice_plane, slice_dist, *d_stickers);
    cudaDeviceSynchronize();
}

extern "C" void allocate_stickers(char (*d_stickers)[6][11][11], int num_stickers) {
    char* temp;
    cudaMalloc(&temp, num_stickers*sizeof(char));
    d_stickers = (char (*)[6][11][11])temp;
}


extern "C" void copy_stickers(char (*d_stickers)[6][11][11], char* h_stickers, int num_stickers) {
    cudaMemcpy(d_stickers, h_stickers, num_stickers*sizeof(char), cudaMemcpyHostToDevice);
}



