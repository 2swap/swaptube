#include <cuda_runtime.h>
#include "color.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "calculator.cuh"
#include <stdint.h>
#include "raymarch_topologies/flat.cuh"
#include "raymarch_topologies/sin.cuh"
#include "raymarch_topologies/parabola.cuh"
#include "raymarch_topologies/blackhole.cuh"
#include "raymarch_topologies/witch.cuh"

static __device__ void metric_tensor(glm::vec3 v, float g[3][3], float* d_intensities) {
    glm::vec4 dx, dy, dz;
    if (d_intensities[0]>0.001) dsurface_dv_flat(v, dx, dy, dz, d_intensities[0]);
    if (d_intensities[1]>0.001) dsurface_dv_sin(v, dx, dy, dz, d_intensities[1]);
    if (d_intensities[2]>0.001) dsurface_dv_parabola(v, dx, dy, dz, d_intensities[2]);
    if (d_intensities[3]>0.001) dsurface_dv_blackhole(v, dx, dy, dz, d_intensities[3]);
    if (d_intensities[4]>0.001) dsurface_dv_witch(v, dx, dy, dz, d_intensities[4]);

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
static __device__ bool christoffel_symbols(glm::vec3 v, float Gamma[3][3][3], float* d_intensities) {
    // compute metric at center
    float g[3][3];
    metric_tensor(v, g, d_intensities);

    // invert metric
    float g_inv[3][3];
    if (!invert3x3(g, g_inv)) return false;

    float dg[3][3][3];
    for( int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                dg[i][j][k] = 0.0f;
            }
        }
    }
    // Flat metric is zero anyways. dmetric_dv_flat(v, dg[0], dg[1], dg[2], d_intensities[0]);
    if (d_intensities[1]>0.001) dmetric_dv_sin(v, dg[0], dg[1], dg[2], d_intensities[1]);
    if (d_intensities[2]>0.001) dmetric_dv_parabola(v, dg[0], dg[1], dg[2], d_intensities[2]);
    if (d_intensities[3]>0.001) dmetric_dv_blackhole(v, dg[0], dg[1], dg[2], d_intensities[3]);
    if (d_intensities[4]>0.001) dmetric_dv_witch(v, dg[0], dg[1], dg[2], d_intensities[4]);

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
static __device__ bool geodesic_rhs(const float Y[6], float dY[6], float* d_intensities) {
    glm::vec3 pos = glm::vec3(Y[0], Y[1], Y[2]);
    float vel[3] = { Y[3], Y[4], Y[5] };

    float Gamma[3][3][3];
    if (!christoffel_symbols(pos, Gamma, d_intensities)) return false;

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
static __device__ bool rk4_step_geodesic(float Y[6], float dt, float* d_intensities) {
    float k1[6], k2[6], k3[6], k4[6], temp[6];

    if (!geodesic_rhs(Y, k1, d_intensities)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k1[i];

    if (!geodesic_rhs(temp, k2, d_intensities)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k2[i];

    if (!geodesic_rhs(temp, k3, d_intensities)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + dt * k3[i];

    if (!geodesic_rhs(temp, k4, d_intensities)) return false;
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
                                             float fov, float* d_intensities, float floor_distort,
                                             float step_size, int step_count,
                                             float floor_y, float ceiling_y, float grid_opacity, float zaxis) {
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

    int step = 0;
    float grid_accumulator = 1.0f;
    uint32_t out = 0xFF000000u;
    for (step = 0; step < step_count; ++step) {
        float floor_y_here = floor_y;
        if (fabsf(floor_distort) > 0.001f) floor_y_here += floor_distort * (sin(Y[0] * 5) + sin(Y[2] * 5)) * 0.2f;
        if (Y[1] < floor_y_here) { // Floor Pattern
            int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[2]+.5));
            out = square_num % 2 ?
                0xff00bb00 : // green
                0xff009900;  // dark green
            break;
        }
        if (Y[1] > ceiling_y) { // Ceiling Pattern
            int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[2]+.5));
            out = square_num % 2 ?
                0xff87ceeb : // light blue
                0xff4682b4;  // steel blue
            break;
        }
        /*
        if(fabsf(Y[0]) > 1.0f || fabsf(Y[2]) > 1.0f) { // Side Walls
            int red = 0xffbc573b;
            int white = 0xffd5d6da;
            out = red;
            int upshift = int((Y[1] + escape_bound)) % 10;
            if(int((Y[1] + escape_bound) * 10) % 10 == 0) out = white;
            else if(int((Y[2] + upshift * 5 + escape_bound) * 10) % 20 == 2 || int((Y[0] + upshift * 5 + escape_bound) * 10) % 20 == 2) {
                out = white;
            }
            break;
        }
        */

        if (zaxis > 0.001f) {
            bool on_z_axis = ( (fabsf(Y[0] - .5) < 0.02f) && (fabsf(Y[1] - .5) < 0.02f) );
            if (on_z_axis) {
                grid_accumulator *= (1.0f - zaxis);
                if (grid_accumulator < 0.01f) break;
            }
        }
        if (grid_opacity > 0.001f) {
            int spacing = 5;
            bool on_x_line = Y[0] / spacing + 1000.5 - floorf(Y[0] / spacing + 1000.5f) < 0.02f;
            bool on_y_line = Y[1] / spacing + 1000.5 - floorf(Y[1] / spacing + 1000.5f) < 0.02f;
            bool on_z_line = Y[2] / spacing + 1000.5 - floorf(Y[2] / spacing + 1000.5f) < 0.02f;
            int num_axes = int(on_x_line) + int(on_y_line) + int(on_z_line);
            if (num_axes >= 2) {
                grid_accumulator *= (1.0f - grid_opacity);
                if (grid_accumulator < 0.01f) break;
            }
        }

        // In Black Hole
        /*if (glm::length(glm::vec3(Y[0], Y[1], Y[2])) < 1.33f) {
            out = 0xff000000; // black
            break;
        }*/

        /*
        // Red Cubes
        int check_u_2 = int(floorf(Y[0] * 20 / escape_bound));
        int check_v_2 = int(floorf(Y[1] * 20 / escape_bound));
        int check_w_2 = int(floorf(Y[2] * 20 / escape_bound));
        if ( ( (check_u_2 % 4) == 0) &&
             ( (check_v_2 % 4) == 0) &&
             ( (check_w_2 % 4) == 0) ) {
            out = 0xffffa500; // orange
            break;
        }*/
        bool ok = rk4_step_geodesic(Y, step_size, d_intensities);
        if (!ok) {
            // numerical trouble (singular metric inversion), break - treat as non-escaped
            out = 0xffff0000; // red for failure
            break;
        }
    }

    out = d_colorlerp(0xffffffff, out, grid_accumulator);
    out = d_colorlerp(out, 0xff000000, float(step) / float(step_count) ); // fade to black based on steps
    d_pixels[py * w + px] = out;
}

// Host-facing launcher
extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
                                             glm::quat camera_orientation, glm::vec3 camera_position,
                                             float fov_rad, float* intensities, float floor_distort,
                                             float step_size, int step_count,
                                             float floor_y, float ceiling_y, float grid_opacity, float zaxis) {
    uint32_t* d_pixels;
    size_t pixel_buffer_size = w * h * sizeof(uint32_t);
    cudaMalloc(&d_pixels, pixel_buffer_size);

    float* d_intensities;
    cudaMalloc(&d_intensities, 5 * sizeof(float));
    cudaMemcpy(d_intensities, intensities, 5 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y );
    cuda_surface_raymarch_kernel<<<grid, block>>>(d_pixels, w, h, camera_orientation, camera_position,
            fov_rad, d_intensities, floor_distort,
            step_size, step_count,
            floor_y, ceiling_y, grid_opacity, zaxis);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pixel_buffer_size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    cudaFree(d_intensities);
}
