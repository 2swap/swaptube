// EUCLIDEAN SPACE

#pragma once

__device__ glm::vec4 surface_flat(glm::vec3 v) {
    return glm::vec4(v.x, v.y, v.z, 0);
}

static __device__ void dsurface_dv_flat(glm::vec3 v, glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz, float intensity) {
    d_dx = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f) * intensity;
    d_dy = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f) * intensity;
    d_dz = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f) * intensity;
}

// Metric tensor is given by:
// [ 1 0 0 ]
// [ 0 1 0 ]
// [ 0 0 1 ] * intensity^2

static __device__ void dmetric_dv_flat(glm::vec3 v, float dg_dx[3][3], float dg_dy[3][3], float dg_dz[3][3], float intensity) {
    dg_dx[0][0] = 0.0f;
    dg_dx[0][1] = 0.0f;
    dg_dx[0][2] = 0.0f;
    dg_dx[1][0] = 0.0f;
    dg_dx[1][1] = 0.0f;
    dg_dx[1][2] = 0.0f;
    dg_dx[2][0] = 0.0f;
    dg_dx[2][1] = 0.0f;
    dg_dx[2][2] = 0.0f;

    dg_dy[0][0] = 0.0f;
    dg_dy[0][1] = 0.0f;
    dg_dy[0][2] = 0.0f;
    dg_dy[1][0] = 0.0f;
    dg_dy[1][1] = 0.0f;
    dg_dy[1][2] = 0.0f;
    dg_dy[2][0] = 0.0f;
    dg_dy[2][1] = 0.0f;
    dg_dy[2][2] = 0.0f;

    dg_dz[0][0] = 0.0f;
    dg_dz[0][1] = 0.0f;
    dg_dz[0][2] = 0.0f;
    dg_dz[1][0] = 0.0f;
    dg_dz[1][1] = 0.0f;
    dg_dz[1][2] = 0.0f;
    dg_dz[2][0] = 0.0f;
    dg_dz[2][1] = 0.0f;
    dg_dz[2][2] = 0.0f;
}
