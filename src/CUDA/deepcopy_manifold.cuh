#include "../Core/State/ResolvedStateEquationComponent.c"
#include "../Host_Device_Shared/ManifoldData.c"

inline ManifoldData deepcopy_manifold(const ManifoldData& h_manifold) {
    ManifoldData d_manifold = h_manifold;

    size_t sz = sizeof(ResolvedStateEquationComponent);

    size_t x_sz = h_manifold.x_size * sz;
    size_t y_sz = h_manifold.y_size * sz;
    size_t z_sz = h_manifold.z_size * sz;
    size_t r_sz = h_manifold.r_size * sz;
    size_t i_sz = h_manifold.i_size * sz;

    cudaMalloc((void**)&d_manifold.x_eq, x_sz);
    cudaMalloc((void**)&d_manifold.y_eq, y_sz);
    cudaMalloc((void**)&d_manifold.z_eq, z_sz);
    cudaMalloc((void**)&d_manifold.r_eq, r_sz);
    cudaMalloc((void**)&d_manifold.i_eq, i_sz);

    cudaMemcpy(d_manifold.x_eq, h_manifold.x_eq, x_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifold.y_eq, h_manifold.y_eq, y_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifold.z_eq, h_manifold.z_eq, z_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifold.r_eq, h_manifold.r_eq, r_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifold.i_eq, h_manifold.i_eq, i_sz, cudaMemcpyHostToDevice);

    return d_manifold;
}

inline void free_manifold(const ManifoldData& d_manifold) {
    cudaFree(d_manifold.x_eq);
    cudaFree(d_manifold.y_eq);
    cudaFree(d_manifold.z_eq);
    cudaFree(d_manifold.r_eq);
    cudaFree(d_manifold.i_eq);
}
