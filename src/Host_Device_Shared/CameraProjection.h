#pragma once
#include "vec.h"
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

HOST_DEVICE inline float behind_camera_sentinel() { return -1000.0f; }

HOST_DEVICE inline vec3 coordinate_to_pixel(
    vec3 coordinate,
    const quat& camera_direction,
    const vec3& camera_pos,
    float fov,
    float geom_mean_size,
    const vec2& wh)
{
    const vec3 rotated = rotate_vector(coordinate - camera_pos, camera_direction);
    if (rotated.z <= 0.0f) {
        const float s = behind_camera_sentinel();
        return vec3(s, s, rotated.z);
    }
    const float scale = (geom_mean_size * fov) / rotated.z;
    return vec3(
         scale * rotated.x + wh.x * 0.5f,
        -scale * rotated.y + wh.y * 0.5f,
        rotated.z);
}

SHARED_FILE_SUFFIX
