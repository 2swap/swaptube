// This file renders a specified 3d manifold in CUDA.
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cstring>
#include "color.cuh" // For complex_to_srgb
#include "common_graphics.cuh"
#include "edge_detect.cuh"
#include "../Host_Device_Shared/ManifoldData.c"
#include "../Host_Device_Shared/vec.h"
#include "deepcopy_manifold.cuh"

// Kernel
__global__ void render_manifold_kernel(
    uint32_t* pixels, const int w, const int h,
    const Cuda::ManifoldData d_manifold,
    const Cuda::vec3 camera_pos, const Cuda::quat camera_direction,
    const float geom_mean_size, const float fov,
    float* depth_buffer,
    const float ab_dilation, const float dot_radius
) {
    // Determine u, v from thread indices
    int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (u_idx >= d_manifold.u_steps || v_idx >= d_manifold.v_steps) return;
    float u = d_manifold.u_min + (d_manifold.u_max - d_manifold.u_min) * u_idx / (d_manifold.u_steps - 1);
    float v = d_manifold.v_min + (d_manifold.v_max - d_manifold.v_min) * v_idx / (d_manifold.v_steps - 1);

    float cuda_tags[3] = { u, v, 0 };

    // Evaluate manifold equations to get 3D point
    int error = 0;
    float x = evaluate_resolved_state_equation(d_manifold.x_size, d_manifold.x_eq, cuda_tags, 2, error);
    float y = evaluate_resolved_state_equation(d_manifold.y_size, d_manifold.y_eq, cuda_tags, 2, error);
    float z = evaluate_resolved_state_equation(d_manifold.z_size, d_manifold.z_eq, cuda_tags, 2, error);

    // Project 3D point to 2D screen space
    bool behind_camera = false;
    float out_x, out_y, out_z;
    d_coordinate_to_pixel(
        {x, y, z},
        behind_camera,
        camera_direction,
        camera_pos,
        fov,
        geom_mean_size,
        w,
        h,
        out_x, out_y, out_z
    );
    //if(behind_camera) return; // Don't render points behind camera
    int pixel_x = out_x;
    int pixel_y = out_y;

    // Evaluate color equations to get color
    float r = evaluate_resolved_state_equation(d_manifold.r_size, d_manifold.r_eq, cuda_tags, 3, error);
    float i = evaluate_resolved_state_equation(d_manifold.i_size, d_manifold.i_eq, cuda_tags, 3, error);

    uint32_t color = d_complex_to_srgb(thrust::complex<float>(r, i), ab_dilation, dot_radius);

    // Depth test and write pixel
    if (pixel_x >= 0 && pixel_x < w && pixel_y >= 0 && pixel_y < h) {
        int pixel_index = pixel_y * w + pixel_x;
        // Atomically test and update depth using atomicCAS on float bit patterns
        const float eps = 3e-3f; // epsilon to avoid z-fighting
        unsigned int* depth_ui = (unsigned int*)(depth_buffer + pixel_index);
        unsigned int old_ui = *depth_ui;
        for (;;) {
            float old_f = __int_as_float(old_ui);
            if (!(out_z < old_f - eps)) break; // no update needed
            unsigned int new_ui = __float_as_int(out_z);
            unsigned int prev_ui = atomicCAS(depth_ui, old_ui, new_ui);
            if (prev_ui == old_ui) {
                // successfully updated depth, write pixel color
                pixels[pixel_index] = color;
                break;
            }
            old_ui = prev_ui; // try again with new observed value
        }
    }
}

// Externed entry point
extern "C" void cuda_render_manifold(
    uint32_t* pixels, const int w, const int h,
    const Cuda::ManifoldData* manifold, const int num_manifolds,
    const Cuda::vec3 camera_pos, const Cuda::quat camera_direction,
    const float geom_mean_size, const float fov,
    const float ab_dilation, const float dot_radius
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

    for(int m = 0; m < num_manifolds; ++m) {
        Cuda::ManifoldData d_manifold = deepcopy_manifold(manifold[m]);

        dim3 blockSize(16, 16);
        dim3 gridSize((manifold[m].u_steps + blockSize.x - 1) / blockSize.x, (manifold[m].v_steps + blockSize.y - 1) / blockSize.y);
        render_manifold_kernel<<<gridSize, blockSize>>>(
            d_pixels, w, h,
            d_manifold,
            camera_pos, camera_direction,
            geom_mean_size, fov,
            d_depth_buffer,
            ab_dilation, dot_radius
        );
        cudaDeviceSynchronize();

        free_manifold(d_manifold);
    }

    //cuda_edge_detect(d_pixels, d_depth_buffer, w, h, 0xffffbbbb);

    // Copy pixels back to host
    cudaMemcpy(pixels, d_pixels, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_depth_buffer);
}
