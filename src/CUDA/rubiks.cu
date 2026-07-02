#include "../Host_Device_Shared/vec.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include "../Host_Device_Shared/ThreeDimensionStructs.h"
#include "color.cuh" // Contains overlay_pixel and set_pixel
#include "common_graphics.cuh" // Contains get_raymarch_vector


// for each pixel, check if it intersects with the cube, and then identify color

__device__ __forceinline__ bool ray_cube_intersect(
    const Cuda::vec3& ray_origin, const Cuda::vec3& ray_dir, Cuda::vec3& out_hit_point
    , char& face_name, Cuda::vec2& uv_coords) {
    Cuda::vec3 oc = ray_origin;
    const char faces[6] = {'R', 'L', 'D', 'U', 'B', 'F'}; // this should probably be outside of the function for efficiency
    // we will probably need to change the order of the faces to match the definition in Rubiks.cpp

    // normal vector for each face of the cube
    Cuda::vec3 normals[6] = {
        Cuda::vec3(-1, 0, 0), // left
        Cuda::vec3(1, 0, 0),  // right
        Cuda::vec3(0, -1, 0), // down
        Cuda::vec3(0, 1, 0),  // up
        Cuda::vec3(0, 0, -1), // back
        Cuda::vec3(0, 0, 1)   // front
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

    // determine which face was hit based on the index of the minimum t value
    if (min_index >= 0 && min_index < 6) {
        face_name = faces[min_index];
    } else {
        face_name = '?';
    }

    out_hit_point = oc + min_t * ray_dir;
    
    //uv coordinates for the face hit, normalized to [0, 1]
    float u = 0.0f, v = 0.0f;

    switch (min_index) { //might have to adjut inverted axes depending on the convention used for the cube's orientation
        case 0:
            u = (out_hit_point.z + 1.0f) * 0.5f; 
            v = (out_hit_point.y + 1.0f) * 0.5f; 
            break;
        case 1:
            u = (1.0f - out_hit_point.z) * 0.5f; 
            v = (out_hit_point.y + 1.0f) * 0.5f; 
            break;
        case 2:
            u = (out_hit_point.x + 1.0f) * 0.5f; 
            v = (out_hit_point.z + 1.0f) * 0.5f; 
            break;
        case 3:
            u = (out_hit_point.x + 1.0f) * 0.5f; 
            v = (1.0f - out_hit_point.z) * 0.5f; 
            break;
        case 4:
            u = (1.0f - out_hit_point.x) * 0.5f; 
            v = (out_hit_point.y + 1.0f) * 0.5f; 
            break;
        case 5:
            u = (out_hit_point.x + 1.0f) * 0.5f; 
            v = (out_hit_point.y + 1.0f) * 0.5f; 
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
    int cube_size)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::ivec2 pixel(px, py);

    Cuda::vec3 ray_dir = get_raymarch_vector(pixel, wh, fov, conjugate(camera_direction));
    Cuda::vec3 hit_point;
    char face_name_char;
    Cuda::vec2 sticker_coords;
    bool collision = ray_cube_intersect(camera_pos, ray_dir, hit_point, face_name_char, sticker_coords);
    if(!collision) {
        pixels[pixel.x + wh.x * pixel.y] = 0x00000000;
        return;
    }


    // Determine which sticker the pixel belongs to based on the uv coordinates and cube size
    int col = (int)(sticker_coords.x * cube_size);
    int row = (int)(sticker_coords.y * cube_size);

    // coordinates inside the sticker, normalized to [0, 1]
    float local_u = fmodf(sticker_coords.x * cube_size, 1.0f);
    float local_v = fmodf(sticker_coords.y * cube_size, 1.0f);

    // normalize to [-1, 1] for the purpose of calculating the 8-norm
    float x = local_u * 2.0f - 1.0f;
    float y = local_v * 2.0f - 1.0f;

    // scale factor to slightly enlarge the coordinates, making the stickers appear smaller and more distinct
    float scale = 1.15f; 
    x *= scale;
    y *= scale;

    // Calculate the 8-norm of the coordinates to determine if the pixel is within the sticker's bounds
    float x2 = x * x; float x4 = x2 * x2; float x8 = x4 * x4;
    float y2 = y * y; float y4 = y2 * y2; float y8 = y4 * y4;

    if (x8 + y8 >= 1.0f) {
        pixels[pixel.x + wh.x * pixel.y] = 0x00000000; // transparent plastic
        return;
    }


    uint32_t color = 0xFFFFFFFF; // default color
    
    // For now, checker pattern for the stickers
    if (1==2){                                      //((row + col) % 2 == 0) {
        color = 0xFF000000; // black sticker
    } else {
        switch (face_name_char) {
            case 'R': color = 0xFFC21D1D; break; // Red
            case 'F': color = 0xFF1DC249; break; // Green
            case 'B': color = 0xFF251BB3; break; // Blue
            case 'U': color = 0xFFFFFFFF; break; // White
            case 'D': color = 0xFFDED82A; break; // Yellow
            case 'L': color = 0xFFF7A31B; break; // Orange
            case '?': color = 0x00FFFFFF; break; // Transparent for unknown face 
        }
    }
    
    pixels[pixel.x + wh.x * pixel.y] = color;
}

extern "C" void cuda_render_cube(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    float geom_mean_size,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((wh.x + blockSize.x - 1) / blockSize.x, (wh.y + blockSize.y - 1) / blockSize.y);
    render_cube_kernel<<<gridSize, blockSize>>>(
        d_pixels, wh,
        geom_mean_size,
        camera_direction, camera_pos, fov, 5);// Assuming a 3x3 cube for now
    cudaDeviceSynchronize();
}



