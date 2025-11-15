// Witch of Agnesi surface

#pragma once

__device__ glm::vec4 surface_witch(glm::vec3 v) {
    float r2 = 1 + glm::dot(v, v);
    float w = 1.0f / r2;
    return glm::vec4(v.x, v.y, v.z, w);
}

static __device__ void dsurface_dv_witch(glm::vec3 v, glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz, float intensity) {
    float r2 = 1 + glm::dot(v, v);
    float denom = 2.0f / (r2 * r2);
    d_dx += glm::vec4(1.0f, 0.0f, 0.0f, v.x * denom) * intensity;
    d_dy += glm::vec4(0.0f, 1.0f, 0.0f, v.y * denom) * intensity;
    d_dz += glm::vec4(0.0f, 0.0f, 1.0f, v.z * denom) * intensity;
}

// Metric tensor
//
// [ 1 + 4x^2 / (-x^2-y^2-z^2)^4    4xy / (-x^2-y^2-z^2)^4        4xz / (-x^2-y^2-z^2)^4      ]
// [ 4xy / (-x^2-y^2-z^2)^4        1 + 4y^2 / (-x^2-y^2-z^2)^4    4yz / (-x^2-y^2-z^2)^4      ]
// [ 4xz / (-x^2-y^2-z^2)^4        4yz / (-x^2-y^2-z^2)^4        1 + 4z^2 / (-x^2-y^2-z^2)^4  ] * intensity^2

static __device__ void dmetric_dv_witch(glm::vec3 v,
                                  float dg_dx[3][3],
                                  float dg_dy[3][3],
                                  float dg_dz[3][3], float intensity) {
    float r2 = 1 + glm::dot(v, v);
    float denom = r2;
    float denom2 = denom * denom;
    float denom3 = denom2 * denom;
    float denom4 = denom3 * denom;

    float d_denom4_dx = 8.0f * v.x / denom3;
    float d_denom4_dy = 8.0f * v.y / denom3;
    float d_denom4_dz = 8.0f * v.z / denom3;

    float denom8 = denom4 * denom4;

    float constant_term = intensity * intensity / denom8;

    dg_dx[0][0] += (8.0f * v.x * denom4 - 4.0f * v.x * v.x * d_denom4_dx) * constant_term;
    dg_dx[0][1] += (4.0f * v.y * denom4 - 4.0f * v.x * v.y * d_denom4_dx) * constant_term;
    dg_dx[0][2] += (4.0f * v.z * denom4 - 4.0f * v.x * v.z * d_denom4_dx) * constant_term;
    dg_dx[1][0] += dg_dx[0][1];
    dg_dx[1][1] += (-4.0f * v.y * v.y * d_denom4_dx) * constant_term;
    dg_dx[1][2] += (-4.0f * v.y * v.z * d_denom4_dx) * constant_term;
    dg_dx[2][0] += dg_dx[0][2];
    dg_dx[2][1] += dg_dx[1][2];
    dg_dx[2][2] += (-4.0f * v.z * v.z * d_denom4_dx) * constant_term;

    dg_dy[0][0] += (-4.0f * v.x * v.x * d_denom4_dy) * constant_term;
    dg_dy[0][1] += (4.0f * v.x * denom4 - 4.0f * v.x * v.y * d_denom4_dy) * constant_term;
    dg_dy[0][2] += (-4.0f * v.x * v.z * d_denom4_dy) * constant_term;
    dg_dy[1][0] += dg_dy[0][1];
    dg_dy[1][1] += (8.0f * v.y * denom4 - 4.0f * v.y * v.y * d_denom4_dy) * constant_term;
    dg_dy[1][2] += (4.0f * v.z * denom4 - 4.0f * v.y * v.z * d_denom4_dy) * constant_term;
    dg_dy[2][0] += dg_dy[0][2];
    dg_dy[2][1] += dg_dy[1][2];
    dg_dy[2][2] += (-4.0f * v.z * v.z * d_denom4_dy) * constant_term;

    dg_dz[0][0] += (-4.0f * v.x * v.x * d_denom4_dz) * constant_term;
    dg_dz[0][1] += (-4.0f * v.x * v.y * d_denom4_dz) * constant_term;
    dg_dz[0][2] += (4.0f * v.x * denom4 - 4.0f * v.x * v.z * d_denom4_dz) * constant_term;
    dg_dz[1][0] += dg_dz[0][1];
    dg_dz[1][1] += (-4.0f * v.y * v.y * d_denom4_dz) * constant_term;
    dg_dz[1][2] += (4.0f * v.y * denom4 - 4.0f * v.y * v.z * d_denom4_dz) * constant_term;
    dg_dz[2][0] += dg_dz[0][2];
    dg_dz[2][1] += dg_dz[1][2];
    dg_dz[2][2] += (8.0f * v.z * denom4 - 4.0f * v.z * v.z * d_denom4_dz) * constant_term;
}
