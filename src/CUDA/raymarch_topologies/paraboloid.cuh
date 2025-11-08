// PARABOLOID

#pragma once

__device__ glm::vec4 surface(glm::vec3 v, float intensity) {
    float w = v.x * v.x + v.y * v.y + v.z * v.z;
    return glm::vec4(v.x, v.y, v.z, w * intensity);
}

static __device__ void dsurface_dv(glm::vec3 v, glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz, float intensity) {
    d_dx = glm::vec4(1.0f, 0.0f, 0.0f, 2.0f * v.x * intensity);
    d_dy = glm::vec4(0.0f, 1.0f, 0.0f, 2.0f * v.y * intensity);
    d_dz = glm::vec4(0.0f, 0.0f, 1.0f, 2.0f * v.z * intensity);
}

// Metric tensor is given by:
// 1+4x^2    4xy        4xz
// 4xy      1+4y^2      4yz
// 4xz        4yz      1+4z^2

static __device__ void dmetric_dv(glm::vec3 v, float dg_dx[3][3], float dg_dy[3][3], float dg_dz[3][3], float intensity) {
    dg_dx[0][0] = 8.0f * v.x * intensity;
    dg_dx[0][1] = 4.0f * v.y * intensity;
    dg_dx[0][2] = 4.0f * v.z * intensity;
    dg_dx[1][0] = 4.0f * v.y * intensity;
    dg_dx[1][1] = 0.0f;
    dg_dx[1][2] = 0.0f;
    dg_dx[2][0] = 4.0f * v.z * intensity;
    dg_dx[2][1] = 0.0f;
    dg_dx[2][2] = 0.0f;

    dg_dy[0][0] = 0.0f;
    dg_dy[0][1] = 4.0f * v.x * intensity;
    dg_dy[0][2] = 0.0f;
    dg_dy[1][0] = 4.0f * v.x * intensity;
    dg_dy[1][1] = 8.0f * v.y * intensity;
    dg_dy[1][2] = 4.0f * v.z * intensity;
    dg_dy[2][0] = 0.0f;
    dg_dy[2][1] = 4.0f * v.z * intensity;
    dg_dy[2][2] = 0.0f;

    dg_dz[0][0] = 0.0f;
    dg_dz[0][1] = 0.0f;
    dg_dz[0][2] = 4.0f * v.x * intensity;
    dg_dz[1][0] = 0.0f;
    dg_dz[1][1] = 0.0f;
    dg_dz[1][2] = 4.0f * v.y * intensity;
    dg_dz[2][0] = 4.0f * v.x * intensity;
    dg_dz[2][1] = 4.0f * v.y * intensity;
    dg_dz[2][2] = 8.0f * v.z * intensity;
}
