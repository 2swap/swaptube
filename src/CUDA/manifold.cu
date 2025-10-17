// This file renders a specified 3d manifold in CUDA.
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstring>
#include "calculator.cuh"
#include "color.cuh" // For complex_to_srgb
#include "common_graphics.cuh"
#include "edge_detect.cuh"
#include "../Host_Device_Shared/ManifoldData.h"

// Kernel
__global__ void render_manifold_kernel(
    uint32_t* pixels, const int w, const int h,
    const char* x_eq, const char* y_eq, const char* z_eq,
    const char* r_eq, const char* i_eq,
    const float u_min, const float u_max, const int u_steps,
    const float v_min, const float v_max, const int v_steps,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float* depth_buffer,
    const float ab_dilation, const float dot_radius
) {
    // Determine u, v from thread indices
    int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (u_idx >= u_steps || v_idx >= v_steps) return;
    float u = u_min + (u_max - u_min) * u_idx / (u_steps - 1);
    float v = v_min + (v_max - v_min) * v_idx / (v_steps - 1);

    // Evaluate manifold equations to get 3D point
    double x = 0;
    double y = 0;
    double z = 0;
    char x_inserted[256];
    char y_inserted[256];
    char z_inserted[256];
    insert_tags(x_eq, u, v, x_inserted, 256);
    insert_tags(y_eq, u, v, y_inserted, 256);
    insert_tags(z_eq, u, v, z_inserted, 256);
    if(!calculator(x_inserted, &x)) printf("Error calculating manifold x at u=%f v=%f: %s\n", u, v, x_inserted);
    if(!calculator(y_inserted, &y)) printf("Error calculating manifold y at u=%f v=%f: %s\n", u, v, y_inserted);
    if(!calculator(z_inserted, &z)) printf("Error calculating manifold z at u=%f v=%f: %s\n", u, v, z_inserted);

    // Project 3D point to 2D screen space
    bool behind_camera = false;
    float out_x, out_y, out_z;
    d_coordinate_to_pixel(
        glm::vec3(x, y, z),
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
    int pixel_x = static_cast<int>(out_x);
    int pixel_y = static_cast<int>(out_y);

    // Evaluate color equations to get color
    double r = 0;
    double i = 0;
    char r_inserted[256];
    char i_inserted[256];
    insert_tags(r_eq, u, v, r_inserted, 256);
    insert_tags(i_eq, u, v, i_inserted, 256);
    if(!calculator(r_inserted, &r)) printf("Error calculating color r at u=%f v=%f: %s\n", u, v, r_inserted);
    if(!calculator(i_inserted, &i)) printf("Error calculating color i at u=%f v=%f: %s\n", u, v, i_inserted);

    uint32_t color = d_complex_to_srgb(thrust::complex<float>(r, i), ab_dilation, dot_radius);

    // Depth test and write pixel
    if (pixel_x >= 0 && pixel_x < w && pixel_y >= 0 && pixel_y < h) {
        int pixel_index = pixel_y * w + pixel_x;
        float old_int = atomicMin((int*)&depth_buffer[pixel_index], __float_as_int(out_z));
        float old_depth = __int_as_float(old_int);
        if (out_z < old_depth - 3e-3) { // New pixel is closer with some epsilon to avoid z-fighting
            pixels[pixel_index] = color;
        }
        // TODO is this truly thread-safe / atomic?
    }
}

// Externed entry point
extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const ManifoldData* manifold, const int num_manifolds,
    const glm::vec3 camera_pos, const glm::quat camera_direction, const glm::quat conjugate_camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius,
    const float axes_length
) {
    // Allocate and copy pixels to device
    uint32_t* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(uint32_t));
    cudaMemcpy(d_pixels, pixels, w * h * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Allocate zeroized depth buffer on device (initialize to large values from host)
    float* d_depth_buffer;
    cudaMalloc(&d_depth_buffer, w * h * sizeof(float));
    float* h_depth = (float*)malloc(w * h * sizeof(float));
    for (int i = 0; i < w * h; ++i) h_depth[i] = 1e30f;
    cudaMemcpy(d_depth_buffer, h_depth, w * h * sizeof(float), cudaMemcpyHostToDevice);
    free(h_depth);

    ManifoldData axis_data = {
        "(v) 0 == (u) *",
        "(v) 1 == (u) *",
        "(v) 2 == (u) *",
        ".001",
        ".001",
        -axes_length, axes_length, (int)(2500.0f*axes_length),
        0, 2, 3
    };

    for(int m = 0; m < num_manifolds + 1; ++m) {
        const ManifoldData& manifold_data = (m < num_manifolds) ? manifold[m] : axis_data;
        if(axes_length <= 0 && m == num_manifolds) continue; // Skip axis if not needed

        // Copy string expressions to device
        char* d_x_eq = nullptr;
        char* d_y_eq = nullptr;
        char* d_z_eq = nullptr;
        char* d_r_eq = nullptr;
        char* d_i_eq = nullptr;

        size_t len;

        len = strlen(manifold_data.x_eq) + 1;
        cudaMalloc(&d_x_eq, len);
        cudaMemcpy(d_x_eq, manifold_data.x_eq, len, cudaMemcpyHostToDevice);

        len = strlen(manifold_data.y_eq) + 1;
        cudaMalloc(&d_y_eq, len);
        cudaMemcpy(d_y_eq, manifold_data.y_eq, len, cudaMemcpyHostToDevice);

        len = strlen(manifold_data.z_eq) + 1;
        cudaMalloc(&d_z_eq, len);
        cudaMemcpy(d_z_eq, manifold_data.z_eq, len, cudaMemcpyHostToDevice);

        len = strlen(manifold_data.r_eq) + 1;
        cudaMalloc(&d_r_eq, len);
        cudaMemcpy(d_r_eq, manifold_data.r_eq, len, cudaMemcpyHostToDevice);

        len = strlen(manifold_data.i_eq) + 1;
        cudaMalloc(&d_i_eq, len);
        cudaMemcpy(d_i_eq, manifold_data.i_eq, len, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((manifold_data.u_steps + blockSize.x - 1) / blockSize.x, (manifold_data.v_steps + blockSize.y - 1) / blockSize.y);
        render_manifold_kernel<<<gridSize, blockSize>>>(
            d_pixels, w, h,
            d_x_eq, d_y_eq, d_z_eq,
            d_r_eq, d_i_eq,
            manifold_data.u_min, manifold_data.u_max, manifold_data.u_steps,
            manifold_data.v_min, manifold_data.v_max, manifold_data.v_steps,
            camera_pos, camera_direction, conjugate_camera_direction,
            geom_mean_size, fov,
            d_depth_buffer,
            ab_dilation, dot_radius
        );
        cudaDeviceSynchronize();

        // Free device memory for this manifold
        cudaFree(d_x_eq);
        cudaFree(d_y_eq);
        cudaFree(d_z_eq);
        cudaFree(d_r_eq);
        cudaFree(d_i_eq);
    }

    cuda_edge_detect(d_pixels, d_depth_buffer, w, h, 0xffffffff);

    // Copy pixels back to host
    cudaMemcpy(pixels, d_pixels, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_depth_buffer);
}
