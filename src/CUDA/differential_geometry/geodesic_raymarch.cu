#include <cuda_runtime.h>
#include "../color.cuh"
#include "../../Host_Device_Shared/vec.h"
#include <stdint.h>
#include "../../Core/State/ResolvedStateEquationComponent.c"

// Store x equations and sizes in constant memory for fast access
#define MAX_EQ 256
__constant__ Cuda::ResolvedStateEquationComponent d_x_eq[MAX_EQ];
__constant__ Cuda::ResolvedStateEquationComponent d_y_eq[MAX_EQ];
__constant__ Cuda::ResolvedStateEquationComponent d_z_eq[MAX_EQ];
__constant__ Cuda::ResolvedStateEquationComponent d_w_eq[MAX_EQ];
__constant__ int d_x_size;
__constant__ int d_y_size;
__constant__ int d_z_size;
__constant__ int d_w_size;
__constant__ int special_case_code; // 0 is general case, 1 is x,y,z are identity, 2 is flat metric
__constant__ float     delta =        .01f;
__constant__ float inv_delta = 1.0f / .01f;

__device__ Cuda::vec4 surface_eval_general_case(float x, float y, float z) {
    int error_x = 0;
    int error_y = 0;
    int error_z = 0;
    int error_w = 0;
    // Create vector of x, y, and z for evaluation
    float cuda_tags[3] = { x, y, z };
    float ox = evaluate_resolved_state_equation(d_x_size, d_x_eq, cuda_tags, 3, error_x);
    float oy = evaluate_resolved_state_equation(d_y_size, d_y_eq, cuda_tags, 3, error_y);
    float oz = evaluate_resolved_state_equation(d_z_size, d_z_eq, cuda_tags, 3, error_z);
    float ow = evaluate_resolved_state_equation(d_w_size, d_w_eq, cuda_tags, 3, error_w);
    if(error_x) {
        printf("Error calculating manifold x at a=%f b=%f c=%f\n. Error code: %d\n", x, y, z, error_x);
        print_resolved_state_equation(d_x_size, d_x_eq);
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if(error_y) {
        printf("Error calculating manifold y at a=%f b=%f c=%f\n. Error code: %d\n", x, y, z, error_y);
        print_resolved_state_equation(d_y_size, d_y_eq);
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if(error_z) {
        printf("Error calculating manifold z at a=%f b=%f c=%f\n. Error code: %d\n", x, y, z, error_z);
        print_resolved_state_equation(d_z_size, d_z_eq);
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if(error_w) {
        printf("Error calculating manifold w at a=%f b=%f c=%f\n. Error code: %d\n", x, y, z, error_w);
        print_resolved_state_equation(d_w_size, d_w_eq);
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    return { ox, oy, oz, ow };
}

// Compute partial derivatives of the surface embedding wrt parameter axes
static __device__ void dsurface_dv_numerical_general_case(float x, float y, float z, Cuda::vec4& d_dx, Cuda::vec4& d_dy, Cuda::vec4& d_dz) {
    Cuda::vec4 here   = surface_eval_general_case(x, y, z);
    Cuda::vec4 plus_x = surface_eval_general_case(x + delta, y, z);
    Cuda::vec4 plus_y = surface_eval_general_case(x, y + delta, z);
    Cuda::vec4 plus_z = surface_eval_general_case(x, y, z + delta);
    d_dx = (plus_x - here) * inv_delta;
    d_dy = (plus_y - here) * inv_delta;
    d_dz = (plus_z - here) * inv_delta;
}

static __noinline__ __device__ void metric_tensor_general_case(float x, float y, float z, float g[3][3]) {
    Cuda::vec4 d_dx, d_dy, d_dz;
    dsurface_dv_numerical_general_case(x, y, z, d_dx, d_dy, d_dz);

    g[0][0] = dot(d_dx, d_dx);
    g[1][1] = dot(d_dy, d_dy);
    g[2][2] = dot(d_dz, d_dz);

    g[0][1] = g[1][0] = dot(d_dx, d_dy);
    g[0][2] = g[2][0] = dot(d_dx, d_dz);
    g[1][2] = g[2][1] = dot(d_dy, d_dz);
}

static __noinline__ __device__ void dmetric_dv_numerical_general_case(float x, float y, float z, float dg[3][3][3], float g[3][3]) {
    float g_pu[3][3], g_pv[3][3], g_pw[3][3];
    metric_tensor_general_case(x, y, z, g);
    metric_tensor_general_case(x + delta, y, z, g_pu);
    metric_tensor_general_case(x, y + delta, z, g_pv);
    metric_tensor_general_case(x, y, z + delta, g_pw);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            dg[0][i][j] = (g_pu[i][j] - g[i][j]) * inv_delta;
            dg[1][i][j] = (g_pv[i][j] - g[i][j]) * inv_delta;
            dg[2][i][j] = (g_pw[i][j] - g[i][j]) * inv_delta;
        }
}

__noinline__ __device__ float surface_eval_special_case(float x, float y, float z) {
    int error = 0;
    // Create vector of x, y, and z for evaluation
    float cuda_tags[3] = { x, y, z };
    float w = evaluate_resolved_state_equation(d_w_size, d_w_eq, cuda_tags, 3, error);
    if(error) {
        printf("Error calculating manifold w at a=%f b=%f c=%f\n. Error code: %d\n", x, y, z, error);
        print_resolved_state_equation(d_w_size, d_w_eq);
    }

    return w;
}

static __noinline__ __device__ void dmetric_dv_numerical_special_case(float x, float y, float z, float dg[3][3][3], float g[3][3]) {
    float xd = x + delta;
    float yd = y + delta;
    float zd = z + delta;
    float p000 = surface_eval_special_case(x, y, z);
    float p001 = surface_eval_special_case(x, y, zd);
    float p010 = surface_eval_special_case(x, yd, z);
    float p100 = surface_eval_special_case(xd, y, z);
    float p011 = surface_eval_special_case(x, yd, zd);
    float p101 = surface_eval_special_case(xd, y, zd);
    float p110 = surface_eval_special_case(xd, yd, z);
    float p200 = surface_eval_special_case(xd + delta, y, z);
    float p020 = surface_eval_special_case(x, yd + delta, z);
    float p002 = surface_eval_special_case(x, y, zd + delta);
    {
        float dw_dx = (p100 - p000) * inv_delta;
        float dw_dy = (p010 - p000) * inv_delta;
        float dw_dz = (p001 - p000) * inv_delta;

        g[0][0] = 1 + dw_dx * dw_dx;
        g[1][1] = 1 + dw_dy * dw_dy;
        g[2][2] = 1 + dw_dz * dw_dz;

        g[0][1] = g[1][0] = dw_dx * dw_dy;
        g[0][2] = g[2][0] = dw_dx * dw_dz;
        g[1][2] = g[2][1] = dw_dy * dw_dz;
    }
    {
        float dw_dx = (p200 - p100) * inv_delta;
        float dw_dy = (p110 - p100) * inv_delta;
        float dw_dz = (p101 - p100) * inv_delta;

        dg[0][0][0] = (1 + dw_dx * dw_dx - g[0][0]) * inv_delta;
        dg[0][1][1] = (1 + dw_dy * dw_dy - g[1][1]) * inv_delta;
        dg[0][2][2] = (1 + dw_dz * dw_dz - g[2][2]) * inv_delta;

        dg[0][0][1] = dg[0][1][0] = (dw_dx * dw_dy - g[0][1]) * inv_delta;
        dg[0][0][2] = dg[0][2][0] = (dw_dx * dw_dz - g[0][2]) * inv_delta;
        dg[0][1][2] = dg[0][2][1] = (dw_dy * dw_dz - g[1][2]) * inv_delta;
    }
    {
        float dw_dx = (p110 - p010) * inv_delta;
        float dw_dy = (p020 - p010) * inv_delta;
        float dw_dz = (p011 - p010) * inv_delta;

        dg[1][0][0] = (1 + dw_dx * dw_dx - g[0][0]) * inv_delta;
        dg[1][1][1] = (1 + dw_dy * dw_dy - g[1][1]) * inv_delta;
        dg[1][2][2] = (1 + dw_dz * dw_dz - g[2][2]) * inv_delta;

        dg[1][0][1] = dg[1][1][0] = (dw_dx * dw_dy - g[0][1]) * inv_delta;
        dg[1][0][2] = dg[1][2][0] = (dw_dx * dw_dz - g[0][2]) * inv_delta;
        dg[1][1][2] = dg[1][2][1] = (dw_dy * dw_dz - g[1][2]) * inv_delta;
    }
    {
        float dw_dx = (p101 - p001) * inv_delta;
        float dw_dy = (p011 - p001) * inv_delta;
        float dw_dz = (p002 - p001) * inv_delta;

        dg[2][0][0] = (1 + dw_dx * dw_dx - g[0][0]) * inv_delta;
        dg[2][1][1] = (1 + dw_dy * dw_dy - g[1][1]) * inv_delta;
        dg[2][2][2] = (1 + dw_dz * dw_dz - g[2][2]) * inv_delta;

        dg[2][0][1] = dg[2][1][0] = (dw_dx * dw_dy - g[0][1]) * inv_delta;
        dg[2][0][2] = dg[2][2][0] = (dw_dx * dw_dz - g[0][2]) * inv_delta;
        dg[2][1][2] = dg[2][2][1] = (dw_dy * dw_dz - g[1][2]) * inv_delta;
    }
}

static __device__ float determinant_3x3(const float m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

static __device__ bool invert3x3(const float m[3][3], float invOut[3][3]) {
    float det = determinant_3x3(m);
    float inv_det = 1.0f / det;
    if (fabs(det) < 1e-12) {
        return false; // Singular matrix
    }

    invOut[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    invOut[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
    invOut[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    invOut[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
    invOut[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    invOut[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;

    invOut[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    invOut[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
    invOut[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    return true;
}

// Christoffel symbols Γ^i_jk at parameter-space point v using central differences.
// Gamma is output as Gamma[i][j][k]
static __device__ bool christoffel_symbols(Cuda::vec3 v, float Gamma[3][3][3]) {
    // compute metric at center
    float g[3][3];
    float dg[3][3][3];
    if(special_case_code == 1) dmetric_dv_numerical_special_case(v.x, v.y, v.z, dg, g);
    else                       dmetric_dv_numerical_general_case(v.x, v.y, v.z, dg, g);

    // invert metric
    float g_inv[3][3];
    if (!invert3x3(g, g_inv)) return false;

    // Γ^i_jk = 1/2 g^{i l} ( ∂_j g_{l k} + ∂_k g_{l j} - ∂_l g_{j k} )
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                float sum = 0.0f;
                for (int l = 0; l < 3; ++l) {
                    sum += (dg[j][l][k] + dg[k][l][j] - dg[l][j][k]) * g_inv[i][l];
                }
                Gamma[i][j][k] = sum * 0.5f;
            }
        }
    }
    return true;
}

// Geodesic RHS: input Y[6] = [u, v, w, up, vp, wp] -> outputs dY[6]
static __noinline__ __device__ bool geodesic_rhs(const float Y[6], float dY[6]) {
    Cuda::vec3 pos = Cuda::vec3(Y[0], Y[1], Y[2]);
    float vel[3] = { Y[3], Y[4], Y[5] };

    float Gamma[3][3][3];
    if (!christoffel_symbols(pos, Gamma)) return false;

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
static __device__ bool rk4_step_geodesic(float Y[6], float Y2[6], float dt) {
    float k1[6], k2[6], k3[6], k4[6], temp[6];

    if (!geodesic_rhs(Y, k1)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k1[i];

    if (!geodesic_rhs(temp, k2)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + 0.5f * dt * k2[i];

    if (!geodesic_rhs(temp, k3)) return false;
    for (int i = 0; i < 6; ++i) temp[i] = Y[i] + dt * k3[i];

    if (!geodesic_rhs(temp, k4)) return false;
    for (int i = 0; i < 6; ++i)
        Y2[i] = Y[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    return true;
}

static __device__ bool collision_cube(float Y[6], uint32_t& color, float& dist) {
    const int wall_dist = 4;
    const int floor_ceiling_dist = 1;
    if (Y[0] < -wall_dist) { // Left Wall Pattern
        int square_num = floorf(floorf(Y[1]+.5) + floorf(Y[2]+.5));
        color = square_num % 2 ?
            0xffff0000 : // red
            0xff990000;  // dark red
        return true;
    }

    else if (Y[0] > wall_dist) { // Right Wall Pattern
        int square_num = floorf(floorf(Y[1]+.5) + floorf(Y[2]+.5));
        color = square_num % 2 ?
            0xffff8800 : // orange
            0xff994400;  // dark orange
        return true;
    }

    else if (Y[1] < -floor_ceiling_dist) { // Floor Pattern
        int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[2]+.5));
        color = square_num % 2 ?
            0xffffffff : // white
            0xffcccccc;  // light gray
        return true;
    }

    else if (Y[1] > floor_ceiling_dist) { // Ceiling Pattern
        int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[2]+.5));
        color = square_num % 2 ?
            0xffffff00 : // yellow
            0xffff9900;  // dark yellow
        return true;
    }

    else if (Y[2] < -wall_dist) { // Back Wall Pattern
        int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[1]+.5));
        color = square_num % 2 ?
            0xff008800 : // green
            0xff004400;  // dark green
        return true;
    }
    
    else if (Y[2] > wall_dist) { // Front Wall Pattern
        int square_num = floorf(floorf(Y[0]+.5) + floorf(Y[1]+.5));
        color = square_num % 2 ?
            0xff0022ff : // blue
            0xff000099;  // dark blue
        return true;
    }

    dist = fminf(fminf(
        fminf(1 - Y[0], Y[0] + 1),
        fminf(1 - Y[1], Y[1] + 1)),
        fminf(1 - Y[2], Y[2] + 1)
    );

    return false; // no collision
}

// Kernel: trace one ray per pixel, integrate geodesic in parameter-space
__global__ void cuda_surface_raymarch_kernel(uint32_t* d_pixels, int w, int h,
                                             Cuda::quat camera_orientation,
                                             Cuda::vec3 camera_position,
                                             float fov, float max_dist) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    d_pixels[py * w + px] = 0xff0000ff; // default to blue

    // NDC coordinates [-1,1]
    float ndc_x = ((px + 0.5f) / float(w)) * 2.0f - 1.0f;
    float ndc_y = ((py + 0.5f) / float(h)) * 2.0f - 1.0f;

    float aspect = float(w) / float(h);
    float px_cam = ndc_x * tanf(fov * 0.5f) * aspect;
    float py_cam = -ndc_y * tanf(fov * 0.5f); // negative to flip Y to image coords
    Cuda::vec3 dir_cam = Cuda::vec3(px_cam, py_cam, -1.0f);
    Cuda::vec3 dir_world(normalize(rotate_vector(dir_cam, camera_orientation)));

    // Initialize state in parameter-space (we treat param-space coords directly)
    float Y[6];
    float Y2[6];
    Y[0] = camera_position.x;
    Y[1] = camera_position.y;
    Y[2] = camera_position.z;
    Y[3] = dir_world.x;
    Y[4] = dir_world.y;
    Y[5] = dir_world.z;

    uint32_t out = 0xff00ff00u;
    float last_dist = 4.0f;
    float dist_traveled = 0.0f;
    while (dist_traveled < max_dist) {
        if(last_dist < 0.004f) last_dist = 0.004f; // prevent tiny steps
        float dt = fminf(0.4f, last_dist * 1.2f); // adaptive step size
        if(special_case_code == 2) {
            // Flat metric special case: straight line
            Y2[0] = Y[0] + Y[3] * dt;
            Y2[1] = Y[1] + Y[4] * dt;
            Y2[2] = Y[2] + Y[5] * dt;
            Y2[3] = Y[3];
            Y2[4] = Y[4];
            Y2[5] = Y[5];
        }
        else {
            bool ok = rk4_step_geodesic(Y, Y2, dt);
            if (!ok) {
                // numerical trouble (singular metric inversion)
                out = 0xffff0000; // red for failure
                break;
            }
        }

        float cube_dist;
        bool collided = collision_cube(Y2, out, cube_dist);
        if(collided) break;
        last_dist = cube_dist;

        dist_traveled += dt;

        Y[0] = Y2[0];
        Y[1] = Y2[1];
        Y[2] = Y2[2];
        Y[3] = Y2[3];
        Y[4] = Y2[4];
        Y[5] = Y2[5];
    }

    if(dist_traveled >= max_dist) {
        d_pixels[py * w + px] = 0xff000000;
    }

    // fade to black based on steps
    else d_pixels[py * w + px] = d_colorlerp(out, 0xff000000, dist_traveled / max_dist );
}

// Host-facing launcher
extern "C" void launch_cuda_surface_raymarch(uint32_t* h_pixels, int w, int h,
                                             int x_size, Cuda::ResolvedStateEquationComponent* x_eq,
                                             int y_size, Cuda::ResolvedStateEquationComponent* y_eq,
                                             int z_size, Cuda::ResolvedStateEquationComponent* z_eq,
                                             int w_size, Cuda::ResolvedStateEquationComponent* w_eq,
                                             int special,
                                             Cuda::quat camera_orientation, Cuda::vec3 camera_position,
                                             float fov_rad, float max_dist) {

    // Write equations to constant memory
    cudaMemcpyToSymbol(d_x_eq, x_eq, sizeof(Cuda::ResolvedStateEquationComponent) * x_size);
    cudaMemcpyToSymbol(d_y_eq, y_eq, sizeof(Cuda::ResolvedStateEquationComponent) * y_size);
    cudaMemcpyToSymbol(d_z_eq, z_eq, sizeof(Cuda::ResolvedStateEquationComponent) * z_size);
    cudaMemcpyToSymbol(d_w_eq, w_eq, sizeof(Cuda::ResolvedStateEquationComponent) * w_size);
    cudaMemcpyToSymbol(d_x_size, &x_size, sizeof(int));
    cudaMemcpyToSymbol(d_y_size, &y_size, sizeof(int));
    cudaMemcpyToSymbol(d_z_size, &z_size, sizeof(int));
    cudaMemcpyToSymbol(d_w_size, &w_size, sizeof(int));
    cudaMemcpyToSymbol(special_case_code, &special, sizeof(int));

    uint32_t* d_pixels;
    size_t pixel_buffer_size = w * h * sizeof(uint32_t);
    cudaMalloc(&d_pixels, pixel_buffer_size);

    dim3 block(16, 16);
    dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y );
    cuda_surface_raymarch_kernel<<<grid, block>>>(
        d_pixels, w, h, camera_orientation, camera_position,
        fov_rad, max_dist
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pixel_buffer_size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
