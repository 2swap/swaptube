#include <cuda_runtime.h>
#include "color.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "calculator.cuh"
#include <stdint.h>

// Simple paraboloid surface function
/*
__device__ glm::vec4 surface(glm::vec3 v) {
    float w = v.x * v.x + v.y * v.y + v.z * v.z;
    return glm::vec4(v.x, v.y, v.z, w * 2);
}
*/

__constant__ char d_w_equation[256];

__device__ glm::vec4 surface(glm::vec3 v) {
    char w_inserted[256];
    insert_tags_xyz(d_w_equation, v.x, v.y, v.z, w_inserted, 256);
    double w = 0;
    if(!calculator(w_inserted, &w)) printf("Error calculating manifold w at (%f,%f,%f): %s\n", v.x, v.y, v.z, w_inserted);
    return glm::vec4(v.x, v.y, v.z, w);
}

// Compute partial derivatives of the surface embedding wrt parameter axes
static __device__ void dsurface_dv(glm::vec3 v, float delta,
                                   glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz) {
    glm::vec4 here   = surface(v);
    glm::vec4 plus_x = surface(v + glm::vec3(delta, 0.0f, 0.0f));
    glm::vec4 plus_y = surface(v + glm::vec3(0.0f, delta, 0.0f));
    glm::vec4 plus_z = surface(v + glm::vec3(0.0f, 0.0f, delta));
    d_dx = (plus_x - here) / delta;
    d_dy = (plus_y - here) / delta;
    d_dz = (plus_z - here) / delta;
}

static __device__ void metric_tensor(glm::vec3 v, float g[3][3], float delta) {
    glm::vec4 dx, dy, dz;
    dsurface_dv(v, delta, dx, dy, dz);

    g[0][0] = glm::dot(dx, dx);
    g[0][1] = glm::dot(dx, dy);
    g[0][2] = glm::dot(dx, dz);
    g[1][0] = g[0][1];
    g[1][1] = glm::dot(dy, dy);
    g[1][2] = glm::dot(dy, dz);
    g[2][0] = g[0][2];
    g[2][1] = g[1][2];
    g[2][2] = glm::dot(dz, dz);
}

static __device__ float determinant_3x3(const float m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

static __device__ bool invert3x3(const float m[3][3], float invOut[3][3]) {
    float det = determinant_3x3(m);
    if (fabs(det) < 1e-12) {
        return false; // Singular matrix
    }
    float invDet = 1.0f / det;

    invOut[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
    invOut[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * invDet;
    invOut[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;

    invOut[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * invDet;
    invOut[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    invOut[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * invDet;

    invOut[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
    invOut[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * invDet;
    invOut[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;

    return true;
}

// Christoffel symbols Γ^i_jk at parameter-space point v using central differences.
// Gamma is output as Gamma[i][j][k]
static __device__ bool christoffel_symbols(glm::vec3 v, float delta, float Gamma[3][3][3]) {
    // compute metric at center
    float g[3][3];
    metric_tensor(v, g, delta);

    // invert metric
    float g_inv[3][3];
    if (!invert3x3(g, g_inv)) return false;

    // compute metric at neighboring points for central differences
    float g_pu[3][3], g_mu[3][3], g_pv[3][3], g_mv[3][3], g_pw[3][3], g_mw[3][3];

    metric_tensor(v + glm::vec3( delta, 0.0f, 0.0f), g_pu, delta);
    metric_tensor(v + glm::vec3(-delta, 0.0f, 0.0f), g_mu, delta);

    metric_tensor(v + glm::vec3(0.0f,  delta, 0.0f), g_pv, delta);
    metric_tensor(v + glm::vec3(0.0f, -delta, 0.0f), g_mv, delta);

    metric_tensor(v + glm::vec3(0.0f, 0.0f,  delta), g_pw, delta);
    metric_tensor(v + glm::vec3(0.0f, 0.0f, -delta), g_mw, delta);

    float g_u[3][3], g_v[3][3], g_w[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            g_u[i][j] = (g_pu[i][j] - g_mu[i][j]) / (2.0f * delta);
            g_v[i][j] = (g_pv[i][j] - g_mv[i][j]) / (2.0f * delta);
            g_w[i][j] = (g_pw[i][j] - g_mw[i][j]) / (2.0f * delta);
        }

    // dg[a][l][j] where a=0->u,1->v,2->w
    float dg[3][3][3];
    for (int l = 0; l < 3; ++l)
        for (int j = 0; j < 3; ++j) {
            dg[0][l][j] = g_u[l][j];
            dg[1][l][j] = g_v[l][j];
            dg[2][l][j] = g_w[l][j];
        }

    // Γ^i_jk = 1/2 g^{i l} ( ∂_j g_{l k} + ∂_k g_{l j} - ∂_l g_{j k} )
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                float sum = 0.0f;
                for (int l = 0; l < 3; ++l) {
                    float term = dg[j][l][k] + dg[k][l][j] - dg[l][j][k];
                    sum += 0.5f * g_inv[i][l] * term;
                }
                Gamma[i][j][k] = sum;
            }
        }
    }
    return true;
}

// Geodesic RHS: input Y[6] = [u, v, w, up, vp, wp] -> outputs dY[6]
static __device__ bool geodesic_rhs(const float Y[6], float dY[6], float fd_delta) {
    glm::vec3 pos = glm::vec3(Y[0], Y[1], Y[2]);
    float vel[3] = { Y[3], Y[4], Y[5] };

    float Gamma[3][3][3];
    if (!christoffel_symbols(pos, fd_delta, Gamma)) return false;

    // d position = velocity
    dY[0] = vel[0];
    dY[1] = vel[1];
    dY[2] = vel[2];

    // d velocity = - Γ^i_jk v^j v^k
    for (int i = 0; i < 3; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                acc += Gamma[i][j][k] * vel[j] * vel[k];
        dY[3 + i] = -acc;
    }
    return true;
}

// Single RK4 step for 6-D state Y with step dt
static __device__ bool rk4_step_geodesic(float Y[6], float dt, float fd_delta) {
    float k1[6], k2[6], k3[6], k4[6], temp[6];

    if (!geodesic_rhs(Y, k1, fd_delta)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k1[i];

    if (!geodesic_rhs(temp, k2, fd_delta)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k2[i];

    if (!geodesic_rhs(temp, k3, fd_delta)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + dt * k3[i];

    if (!geodesic_rhs(temp, k4, fd_delta)) return false;
    for (int i = 0; i < 6; ++i)
        Y[i] = Y[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);

    return true;
}

// Rotate vec by quat (device compatible)
static __device__ glm::vec3 quat_rotate(const glm::quat& q, const glm::vec3& v) {
    // v' = q * (v,0) * q^{-1}
    glm::quat vq(0.0f, v.x, v.y, v.z);
    glm::quat res = q * vq * glm::conjugate(q);
    return glm::vec3(res.x, res.y, res.z);
}

// Kernel: trace one ray per pixel, integrate geodesic in parameter-space
__global__ void cuda_surface_raymarch_kernel(uint32_t* d_pixels, int w, int h,
                                                        glm::quat camera_orientation,
                                                        glm::vec3 camera_position,
                                                        float fov) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    // NDC coordinates [-1,1]
    float ndc_x = ((px + 0.5f) / float(w)) * 2.0f - 1.0f;
    float ndc_y = ((py + 0.5f) / float(h)) * 2.0f - 1.0f;

    float aspect = float(w) / float(h);
    float px_cam = ndc_x * tanf(fov * 0.5f) * aspect;
    float py_cam = -ndc_y * tanf(fov * 0.5f); // negative to flip Y to image coords
    glm::vec3 dir_cam = glm::vec3(px_cam, py_cam, -1.0f);
    glm::vec3 dir_world = quat_rotate(camera_orientation, dir_cam);
    dir_world = glm::normalize(dir_world);

    // Initialize state in parameter-space (we treat param-space coords directly)
    float Y[6];
    Y[0] = camera_position.x;
    Y[1] = camera_position.y;
    Y[2] = camera_position.z;
    Y[3] = dir_world.x;
    Y[4] = dir_world.y;
    Y[5] = dir_world.z;

    // integration parameters (tune these)
    const float dt = 0.01f;       // step in affine parameter
    const float fd_delta = 1e-1f; // finite-difference delta for metric
    const int max_steps = 1000;    // as requested
    const float escape_bound = 1.0f;

    bool escaped = false;
    bool failure = false;
    int step = 0;
    for (step = 0; step < max_steps; ++step) {
        // escape check
        if (fabsf(Y[0]) > escape_bound || fabsf(Y[1]) > escape_bound || fabsf(Y[2]) > escape_bound) {
            escaped = true;
            break;
        }
        bool ok = rk4_step_geodesic(Y, dt, fd_delta);
        if (!ok) {
            // numerical trouble (singular metric inversion), break - treat as non-escaped
            failure = true;
            break;
        }
    }

    uint32_t out = 0xFF000000u;
    if (failure) {
        out = 0xffff0000; // red for failure
    } else if (escaped) {
        // Color checkerboard pattern based on final position
        int check_u = int(floorf(Y[0] * 10 / escape_bound));
        int check_v = int(floorf(Y[1] * 10 / escape_bound));
        int check_w = int(floorf(Y[2] * 10 / escape_bound));
        out = (( (check_u + check_v + check_w) % 2) == 0) ?
              0xff00ff00 : // green
              0xff0000ff;  // blue
        out = d_colorlerp(out, 0xff000000, float(step) / float(max_steps) ); // fade to black based on steps
    } else {
        out = 0xff000000; // black for non-escaped
    }

    d_pixels[py * w + px] = out;
}

// Host-facing launcher
extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
                                             glm::quat camera_orientation, glm::vec3 camera_position,
                                             float fov_rad, const char* w_eq) {
    uint32_t* d_pixels;
    size_t pixel_buffer_size = w * h * sizeof(uint32_t);
    cudaMalloc(&d_pixels, pixel_buffer_size);

    dim3 block(16, 16);
    dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y );
    cudaMemcpyToSymbol(d_w_equation, w_eq, strlen(w_eq) + 1);
    cuda_surface_raymarch_kernel<<<grid, block>>>(d_pixels, w, h, camera_orientation, camera_position, fov_rad);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pixel_buffer_size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
