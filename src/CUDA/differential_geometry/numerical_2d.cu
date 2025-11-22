#include <cuda_runtime.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "../../Host_Device_Shared/ManifoldData.h"
#include "../calculator.cuh"
#include "../common_graphics.cuh"

#define numeric_delta 1e-4f

__constant__ char d_x_equation[256];
__constant__ char d_y_equation[256];
__constant__ char d_z_equation[256];

__device__ glm::vec3 surface(glm::vec2 v) {
    char x_inserted[256];
    char y_inserted[256];
    char z_inserted[256];
    insert_tags(d_x_equation, v.x, v.y, x_inserted, 256);
    insert_tags(d_y_equation, v.x, v.y, y_inserted, 256);
    insert_tags(d_z_equation, v.x, v.y, z_inserted, 256);
    double x, y, z;
    if(!calculator(x_inserted, &x)) printf("Error calculating manifold x at (%f,%f): %s\n", v.x, v.y, x_inserted);
    if(!calculator(y_inserted, &y)) printf("Error calculating manifold y at (%f,%f): %s\n", v.x, v.y, y_inserted);
    if(!calculator(z_inserted, &z)) printf("Error calculating manifold z at (%f,%f): %s\n", v.x, v.y, z_inserted);
    return glm::vec3(x, y, z);
}

// Compute partial derivatives of the surface embedding wrt parameter axes
static __device__ void dsurface_dv_numerical(glm::vec2 v, glm::vec3& d_dx, glm::vec3& d_dy) {
    glm::vec3 here   = surface(v);
    glm::vec3 plus_x = surface(v + glm::vec2(numeric_delta, 0.0f));
    glm::vec3 plus_y = surface(v + glm::vec2(0.0f, numeric_delta));
    d_dx = (plus_x - here) / numeric_delta;
    d_dy = (plus_y - here) / numeric_delta;
}

static __device__ void metric_tensor(glm::vec2 v, float g[2][2]) {
    glm::vec3 d_dx, d_dy;
    dsurface_dv_numerical(v, d_dx, d_dy);

    g[0][0] = glm::dot(d_dx, d_dx);
    g[0][1] = glm::dot(d_dx, d_dy);

    g[1][0] = glm::dot(d_dy, d_dx);
    g[1][1] = glm::dot(d_dy, d_dy);
}

static __device__ void dmetric_dv_numerical(glm::vec2 v, float dg[2][2][2]) {
    float g_pu[2][2], g_mu[2][2], g_pv[2][2], g_mv[2][2];
    metric_tensor(v + glm::vec2( numeric_delta, 0.0f), g_pu);
    metric_tensor(v + glm::vec2(-numeric_delta, 0.0f), g_mu);

    metric_tensor(v + glm::vec2(0.0f,  numeric_delta), g_pv);
    metric_tensor(v + glm::vec2(0.0f, -numeric_delta), g_mv);

    float g_u[2][2], g_v[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            g_u[i][j] = (g_pu[i][j] - g_mu[i][j]) / (2.0f * numeric_delta);
            g_v[i][j] = (g_pv[i][j] - g_mv[i][j]) / (2.0f * numeric_delta);
        }

    for (int l = 0; l < 2; ++l)
        for (int j = 0; j < 2; ++j) {
            dg[0][l][j] = g_u[l][j];
            dg[1][l][j] = g_v[l][j];
        }
}

static __device__ float determinant_2x2(const float m[2][2]) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

static __device__ bool invert2x2(const float m[2][2], float invOut[2][2]) {
    float det = determinant_2x2(m);
    if (fabs(det) < 1e-12) {
        return false; // Singular matrix
    }
    float invDet = 1.0f / det;

    invOut[0][0] =  m[1][1] * invDet;
    invOut[0][1] = -m[0][1] * invDet;
    invOut[1][0] = -m[1][0] * invDet;
    invOut[1][1] =  m[0][0] * invDet;

    return true;
}

// Christoffel symbols Γ^i_jk at parameter-space point v using central differences.
// Gamma is output as Gamma[i][j][k]
static __device__ bool christoffel_symbols(glm::vec2 v, float Gamma[2][2][2]) {
    // compute metric at center
    float g[2][2];
    metric_tensor(v, g);

    // invert metric
    float g_inv[2][2];
    if (!invert2x2(g, g_inv)) return false;

    float dg[2][2][2];
    for( int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                dg[i][j][k] = 0.0f;
            }
        }
    }
    dmetric_dv_numerical(v, dg);

    // Γ^i_jk = 1/2 g^{i l} ( ∂_j g_{l k} + ∂_k g_{l j} - ∂_l g_{j k} )
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                float sum = 0.0f;
                for (int l = 0; l < 2; ++l) {
                    float term = dg[j][l][k] + dg[k][l][j] - dg[l][j][k];
                    sum += 0.5f * g_inv[i][l] * term;
                }
                Gamma[i][j][k] = sum;
            }
        }
    }
    return true;
}

// Geodesic RHS: input Y[4] = [u, v, up, vp] -> outputs dY[4]
static __device__ bool geodesic_rhs(const float Y[4], float dY[4]) {
    glm::vec2 pos = glm::vec2(Y[0], Y[1]);
    float vel[2] = { Y[2], Y[3] };

    float Gamma[2][2][2];
    if (!christoffel_symbols(pos, Gamma)) return false;

    // d position = velocity
    dY[0] = vel[0];
    dY[1] = vel[1];

    // d velocity = - Γ^i_jk v^j v^k
    for (int i = 0; i < 2; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                acc += Gamma[i][j][k] * vel[j] * vel[k];
        dY[2 + i] = -acc;
    }
    return true;
}

// Single RK4 step for 6-D state Y with step dt
static __device__ bool rk4_step_geodesic(float Y[4], float dt) {
    float k1[4], k2[4], k3[4], k4[4], temp[4];

    if (!geodesic_rhs(Y, k1)) return false;
    for (int i = 0; i < 4; ++i) temp[i] = Y[i] + 0.5f * dt * k1[i];

    if (!geodesic_rhs(temp, k2)) return false;
    for (int i = 0; i < 4; ++i) temp[i] = Y[i] + 0.5f * dt * k2[i];

    if (!geodesic_rhs(temp, k3)) return false;
    for (int i = 0; i < 4; ++i) temp[i] = Y[i] + dt * k3[i];

    if (!geodesic_rhs(temp, k4)) return false;
    for (int i = 0; i < 4; ++i)
        Y[i] = Y[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);

    return true;
}

__global__ void geodesics_2d_kernel(
    uint32_t* pixels, const int w, const int h,
    const glm::vec2 start_position, const glm::vec2 start_velocity,
    const int num_geodesics, const int num_steps, const float spread_angle,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float u_min, const float u_max, const float v_min, const float v_max
) {
    // Determine which geodesic this thread is responsible for
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_geodesics) return;

    // Set up initial state
    float rotation_angle = num_geodesics > 1 ? (index / float(num_geodesics - 1)) - 0.5f : 0.0f;
    rotation_angle *= spread_angle;
    glm::vec2 rotated_start_velocity = glm::vec2(
        start_velocity.x * cosf(rotation_angle) - start_velocity.y * sinf(rotation_angle),
        start_velocity.x * sinf(rotation_angle) + start_velocity.y * cosf(rotation_angle)
    );
    float state[4];
    state[0] = start_position.x;
    state[1] = start_position.y;
    state[2] = rotated_start_velocity.x;
    state[3] = rotated_start_velocity.y;

    // Iterate geodesic curve
    for(int i = 0; i < num_steps; ++i) {
        if (!rk4_step_geodesic(state, 1)) return;
        if (state[0] < u_min || state[0] > u_max || state[1] < v_min || state[1] > v_max) return;

        bool behind_camera = false;
        float out_x, out_y, out_z;
        d_coordinate_to_pixel(
            surface(glm::vec2(state[0], state[1])),
            behind_camera,
            camera_direction,
            camera_pos,
            conjugate_camera_direction,
            fov,
            geom_mean_size,
            w,
            h,
            out_x, out_y, out_z
        );
        if (behind_camera) continue;
        int pixel_x = static_cast<int>(out_x);
        int pixel_y = static_cast<int>(out_y);

        if (pixel_x < 0 || pixel_x >= w || pixel_y < 0 || pixel_y >= h) continue;

        pixels[pixel_y * w + pixel_x] = 0xffff0000;
    }
}

extern "C" void cuda_render_geodesics_2d(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData& manifold,
    const glm::vec2 start_position, const glm::vec2 start_velocity,
    const int num_geodesics, const int num_steps, const float spread_angle,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov
) {
    // Allocate and copy pixels to device
    uint32_t* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(uint32_t));
    cudaMemcpy(d_pixels, pixels, w * h * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Copy string expressions to device constants
    cudaMemcpyToSymbol(d_x_equation, manifold.x_eq, 256);
    cudaMemcpyToSymbol(d_y_equation, manifold.y_eq, 256);
    cudaMemcpyToSymbol(d_z_equation, manifold.z_eq, 256);

    // Launch kernel: one thread per geodesic
    int blockSize = 256;
    int gridSize = (num_geodesics + blockSize - 1) / blockSize;
    geodesics_2d_kernel<<<gridSize, blockSize>>>(
        d_pixels, w, h,
        start_position, start_velocity,
        num_geodesics, num_steps, spread_angle,
        camera_pos, camera_direction, conjugate_camera_direction,
        geom_mean_size, fov,
        manifold.u_min, manifold.u_max, manifold.v_min, manifold.v_max
    );
    cudaDeviceSynchronize();

    // Copy pixels back to host
    cudaMemcpy(pixels, d_pixels, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
}
