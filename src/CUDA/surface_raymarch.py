#pragma once

// Perform ray marching on a non-euclidean surface defined parametrically.
#include <cuda_runtime.h>
#include "color.cuh"

// Kernel over all pixels.
__global__ void cuda_surface_raymarch_kernel(
    uint32_t* d_pixels, int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
}

void cuda_surface_raymarch(
    uint32_t* d_pixels, int w, int h
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    cuda_surface_raymarch_kernel<<<gridSize, blockSize>>>(
        d_pixels, w, h
    );
    cudaDeviceSynchronize();
}


// Python inspiration

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Define your surface function
# -----------------------------
def surface(u, v):
    # Example: hyperbolic paraboloid
    # Replace with any surface(u,v) -> [x, y, z]
    x = u
    y = v
    z = np.cos(u) + np.cos(v)
    return np.array([x, y, z])

# -----------------------------
# Numerical derivatives
# -----------------------------
def derivative_u(u, v, h=1e-5):
    return (surface(u + h, v) - surface(u - h, v)) / (2*h)

def derivative_v(u, v, h=1e-5):
    return (surface(u, v + h) - surface(u, v - h)) / (2*h)

# Metric tensor g_ij = partial_i r · partial_j r
def metric(u, v):
    E = np.dot(derivative_u(u, v), derivative_u(u, v))
    F = np.dot(derivative_u(u, v), derivative_v(u, v))
    G = np.dot(derivative_v(u, v), derivative_v(u, v))
    return np.array([[E, F], [F, G]])

# -----------------------------
# Christoffel symbols Γ^i_jk
# -----------------------------
def christoffel_symbols(u, v, h=1e-5):
    g = metric(u, v)
    g_inv = np.linalg.inv(g)

    # Partial derivatives of the metric
    g_u = (metric(u + h, v) - metric(u - h, v)) / (2*h)
    g_v = (metric(u, v + h) - metric(u, v - h)) / (2*h)
    dg = [g_u, g_v]  # derivative w.r.t u and v

    Gamma = np.zeros((2,2,2))  # Γ^i_jk
    for i in range(2):
        for j in range(2):
            for k in range(2):
                Gamma[i,j,k] = 0.0
                for l in range(2):
                    Gamma[i,j,k] += 0.5 * g_inv[i,l] * (
