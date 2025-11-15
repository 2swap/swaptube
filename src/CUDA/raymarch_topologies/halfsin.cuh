// Sine Moguls

#pragma once

__device__ glm::vec4 surface(glm::vec3 v, const float intensity) {
    float w = v.z > 4 ? 0 : sin(v.x) + sin(v.y) + sin(v.z);
    return glm::vec4(v.x, v.y, v.z, w * intensity);
}

static __device__ void dsurface_dv(const glm::vec3& v, glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz, const float intensity) {
    d_dx = glm::vec4(1.0f, 0.0f, 0.0f, v.z > 4 ? 0 : cos(v.x) * intensity);
    d_dy = glm::vec4(0.0f, 1.0f, 0.0f, v.z > 4 ? 0 : cos(v.y) * intensity);
    d_dz = glm::vec4(0.0f, 0.0f, 1.0f, v.z > 4 ? 0 : cos(v.z) * intensity);
}

// Metric tensor
// =
// [ 1 + cos^2(x)      0               0            ]
// [ 0                 1 + cos^2(y)    0            ]
// [ 0                 0               1 + cos^2(z) ] * intensity^2

static __device__ void dmetric_dv(const glm::vec3& v,
                                  float dg_dx[3][3],
                                  float dg_dy[3][3],
                                  float dg_dz[3][3],
                                  const float intensity) {
    float intensity_sq = intensity * intensity;
    dg_dx[0][0] = v.z > 4 ? 0 : -2.0f * sin(v.x) * cos(v.x) * intensity_sq;
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
    dg_dy[1][1] = v.z > 4 ? 0 : -2.0f * sin(v.y) * cos(v.y) * intensity_sq;
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
    dg_dz[2][2] = v.z > 4 ? 0 : -2.0f * sin(v.z) * cos(v.z) * intensity_sq;
}
