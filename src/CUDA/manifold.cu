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

// Kernel
__global__ void render_manifold_kernel(
    uint32_t* pixels, int w, int h,
    const char* manifold_x_eq, const char* manifold_y_eq, const char* manifold_z_eq,
    const char* color_r_eq, const char* color_i_eq,
    float u_min, float u_max, int u_steps,
    float v_min, float v_max, int v_steps,
    glm::vec3 camera_pos, glm::quat camera_direction, glm::quat conjugate_camera_direction,
    float geom_mean_size, float fov,
    float opacity,
    float* depth_buffer,
    float ab_dilation, float dot_radius
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
    insert_tags(manifold_x_eq, u, v, x_inserted, 256);
    insert_tags(manifold_y_eq, u, v, y_inserted, 256);
    insert_tags(manifold_z_eq, u, v, z_inserted, 256);
    if(!calculator(x_inserted, &x)) printf("Error calculating manifold x at u=%f v=%f\n", u, v);
    if(!calculator(y_inserted, &y)) printf("Error calculating manifold y at u=%f v=%f\n", u, v);
    if(!calculator(z_inserted, &z)) printf("Error calculating manifold z at u=%f v=%f\n", u, v);

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
    insert_tags(color_r_eq, u, v, r_inserted, 256);
    insert_tags(color_i_eq, u, v, i_inserted, 256);
    if(!calculator(r_inserted, &r)) printf("Error calculating color r at u=%f v=%f\n", u, v);
    if(!calculator(i_inserted, &i)) printf("Error calculating color i at u=%f v=%f\n", u, v);

    uint32_t color = d_complex_to_srgb(thrust::complex<float>(r, i), ab_dilation, dot_radius);

    // Depth test and write pixel
    if (pixel_x >= 0 && pixel_x < w && pixel_y >= 0 && pixel_y < h) {
        int pixel_index = pixel_y * w + pixel_x;
        float old_int = atomicMin((int*)&depth_buffer[pixel_index], __float_as_int(out_z));
        float old_depth = __int_as_float(old_int);
        if (out_z < old_depth - 1e-2) { // New pixel is closer with some epsilon to avoid z-fighting
            pixels[pixel_index] = color;
        }
        // TODO is this truly thread-safe / atomic?
    }
}

// Externed entry point
extern "C" void cuda_render_manifold(
    uint32_t* pixels, int w, int h,
    const char* manifold_x_eq, const char* manifold_y_eq, const char* manifold_z_eq,
    const char* color_r_eq, const char* color_i_eq,
    float u_min, float u_max, int u_steps,
    float v_min, float v_max, int v_steps,
    glm::vec3 camera_pos, glm::quat camera_direction, glm::quat conjugate_camera_direction,
    float geom_mean_size, float fov,
    float opacity,
    float ab_dilation, float dot_radius
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

    // Copy string expressions to device
    char* d_manifold_x_eq = nullptr;
    char* d_manifold_y_eq = nullptr;
    char* d_manifold_z_eq = nullptr;
    char* d_color_r_eq = nullptr;
    char* d_color_i_eq = nullptr;
    size_t len;

    len = strlen(manifold_x_eq) + 1;
    cudaMalloc(&d_manifold_x_eq, len);
    cudaMemcpy(d_manifold_x_eq, manifold_x_eq, len, cudaMemcpyHostToDevice);

    len = strlen(manifold_y_eq) + 1;
    cudaMalloc(&d_manifold_y_eq, len);
    cudaMemcpy(d_manifold_y_eq, manifold_y_eq, len, cudaMemcpyHostToDevice);

    len = strlen(manifold_z_eq) + 1;
    cudaMalloc(&d_manifold_z_eq, len);
    cudaMemcpy(d_manifold_z_eq, manifold_z_eq, len, cudaMemcpyHostToDevice);

    len = strlen(color_r_eq) + 1;
    cudaMalloc(&d_color_r_eq, len);
    cudaMemcpy(d_color_r_eq, color_r_eq, len, cudaMemcpyHostToDevice);

    len = strlen(color_i_eq) + 1;
    cudaMalloc(&d_color_i_eq, len);
    cudaMemcpy(d_color_i_eq, color_i_eq, len, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((u_steps + blockSize.x - 1) / blockSize.x, (v_steps + blockSize.y - 1) / blockSize.y);
    render_manifold_kernel<<<gridSize, blockSize>>>(
        d_pixels, w, h,
        d_manifold_x_eq, d_manifold_y_eq, d_manifold_z_eq,
        d_color_r_eq, d_color_i_eq,
        u_min, u_max, u_steps,
        v_min, v_max, v_steps,
        camera_pos, camera_direction, conjugate_camera_direction,
        geom_mean_size, fov,
        opacity,
        d_depth_buffer,
        ab_dilation, dot_radius
    );
    cudaDeviceSynchronize();

    // Now render 3-dimensional axes (for reference). 15,000 steps is plenty.
    blockSize = dim3(16, 16);
    gridSize = dim3((5000 + blockSize.x - 1) / blockSize.x, (3 + blockSize.y - 1) / blockSize.y);
    // Copy axis equations to device
    const char* axis_x_eq = "(v) 0 == (u) *";
    const char* axis_y_eq = "(v) 1 == (u) *";
    const char* axis_z_eq = "(v) 2 == (u) *";
    const char* axis_color_r_eq = ".001";
    const char* axis_color_i_eq = ".001";
    char* d_axis_x_eq;
    char* d_axis_y_eq;
    char* d_axis_z_eq;
    char* d_axis_color_r_eq;
    char* d_axis_color_i_eq;
    len = strlen(axis_x_eq) + 1;
    cudaMalloc(&d_axis_x_eq, len);
    cudaMalloc(&d_axis_y_eq, len);
    cudaMalloc(&d_axis_z_eq, len);
    cudaMemcpy(d_axis_x_eq, axis_x_eq, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_axis_y_eq, axis_y_eq, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_axis_z_eq, axis_z_eq, len, cudaMemcpyHostToDevice);
    len = strlen(axis_color_r_eq) + 1;
    cudaMalloc(&d_axis_color_r_eq, len);
    cudaMalloc(&d_axis_color_i_eq, len);
    cudaMemcpy(d_axis_color_r_eq, axis_color_r_eq, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_axis_color_i_eq, axis_color_i_eq, len, cudaMemcpyHostToDevice);
    render_manifold_kernel<<<gridSize, blockSize>>>(
        d_pixels, w, h,
        d_axis_x_eq, d_axis_y_eq, d_axis_z_eq,
        d_axis_color_r_eq, d_axis_color_i_eq,
        -2, 2, 5000,
        0, 2, 3,
        camera_pos, camera_direction, conjugate_camera_direction,
        geom_mean_size, fov,
        opacity,
        d_depth_buffer,
        ab_dilation, dot_radius
    );
    cudaDeviceSynchronize();

    cuda_edge_detect(d_pixels, d_depth_buffer, w, h, 0xffffffff);

    // Copy pixels back to host
    cudaMemcpy(pixels, d_pixels, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_depth_buffer);
    cudaFree(d_manifold_x_eq);
    cudaFree(d_manifold_y_eq);
    cudaFree(d_manifold_z_eq);
    cudaFree(d_color_r_eq);
    cudaFree(d_color_i_eq);
    cudaFree(d_axis_x_eq);
    cudaFree(d_axis_y_eq);
    cudaFree(d_axis_z_eq);
    cudaFree(d_axis_color_r_eq);
    cudaFree(d_axis_color_i_eq);
}
