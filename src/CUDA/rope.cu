// draw nodes (everything)
// run physics
// allocate memory
// copy pins data to GPU
// initialize nodes
// free the memory IMPORTANT !
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "common_graphics.cuh"
#include "../Host_Device_Shared/helpers.h"


__global__ void physics_kernel(Cuda::vec2* rope, const int rope_length, const Cuda::vec2* pins,
    const int pins_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rope_length) return;

    Cuda::vec2 mean(0.0f, 0.0f);
    for (int j = -4; j <= 4; j++) {
        int delta = 1<<abs(j);
        if (j < 0) delta = -delta;
        mean += rope[(i+delta+rope_length)%rope_length]/9.0f;
    }
    rope[i] = mean;
    rope[i].y -= 0.0001f;


    // for across all pins and repell
    for (int j = 0; j < pins_length; ++j) {
        Cuda::vec2 delta = rope[i] - pins[j];
        float dist_squared = dot(delta, delta);
        float dist_quarted = dist_squared * dist_squared;
        Cuda::vec2 repel_force = delta / (dist_quarted) * 0.0000002f;
        //repel_force *= 5*(1.01+sin(i/1000.0f*3.14159f*2.0f)); // diff reppel
        rope[i] += repel_force;
    }
}

extern "C" void physics(Cuda::vec2* rope, const int rope_length, Cuda::vec2* pins,
    const int pins_length){
    int blockSize = 256;
    int gridSize = (rope_length + blockSize - 1) / blockSize;

    physics_kernel<<<gridSize, blockSize>>>(rope, rope_length, pins, pins_length);

}

extern "C" void allocate_rope_and_pins(Cuda::vec2** rope_pointer, Cuda::vec2** pins_pointer){
    //use cudamalloc here
    cudaMalloc(rope_pointer, 1000 * sizeof(Cuda::vec2));
    cudaMalloc(pins_pointer, 20 * sizeof(Cuda::vec2));
}

extern "C" void copy_pins(const Cuda::vec2* h_pins, Cuda::vec2* d_pins, const int pins_length){
    cudaMemcpy(d_pins, h_pins, pins_length * sizeof(Cuda::vec2), cudaMemcpyHostToDevice);
}

extern "C" void initialize_nodes_from_file(const std::string& file_name, Cuda::vec2* d_rope) {
    // 1. Lecture du fichier texte sur le CPU
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Erreur: Impossible d'ouvrir le fichier " << file_name << std::endl;
        return;
    }

    std::vector<Cuda::vec2> h_rope;
    Cuda::vec2 temp_point;

    // Lit les paires de float (x, y) ligne par ligne
    while (file >> temp_point.x >> temp_point.y) {
        h_rope.push_back(temp_point);
        // printf("Read point: (%f, %f)\n", temp_point.x, temp_point.y);
    }
    file.close();

    if (h_rope.empty()) {
        std::cerr << "Attention: Le fichier est vide ou mal formaté." << std::endl;
        return;
    }

    // 2. Transfert de la mémoire Host (CPU) vers Device (GPU)
    cudaError_t err = cudaMemcpy(
        d_rope, 
        h_rope.data(), 
        h_rope.size() * sizeof(Cuda::vec2), 
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        std::cerr << "Erreur CUDA Memcpy: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void free_memory(Cuda::vec2* rope_pointer, Cuda::vec2* pins_pointer){
    cudaFree(rope_pointer);
    cudaFree(pins_pointer);
}


__global__ void render_rope_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh, const Cuda::vec2* rope, const int rope_length, Cuda::vec2 lx_ty, Cuda::vec2 rx_by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rope_length) return;


    Cuda::vec2 pixel_1 = point_to_pixel_in_screen(rope[i], lx_ty, rx_by, wh);
    Cuda::vec2 pixel_2 = point_to_pixel_in_screen(rope[(i+1)%rope_length], lx_ty, rx_by, wh);

    bresenham(pixel_1.x, pixel_1.y, pixel_2.x, pixel_2.y, 0xFFFFFFFF, 1.0f, 2, pixels, wh, false);
}




extern "C" void cuda_render_rope(uint32_t* d_pixels, const Cuda::ivec2& wh, const Cuda::vec2* d_rope, const int rope_length,
    const Cuda::vec2& lx_ty, const Cuda::vec2& rx_by)
{
    int blockSize = 256;
    int gridSize = (rope_length + blockSize - 1) / blockSize;
    render_rope_kernel<<<gridSize, blockSize>>>(d_pixels, wh, d_rope, rope_length, lx_ty, rx_by);
    cudaDeviceSynchronize();
}