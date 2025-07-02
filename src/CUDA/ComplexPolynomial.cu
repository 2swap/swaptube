#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ int complex_to_color(const thrust::complex<double>& c) {
    float hue = atan2(c.imag(), c.real()) * 180 / M_PI + 180;  // Convert [-π, π] to [0, 360]
    float saturation = 1.0f;
    float value = (2/M_PI) * atan(abs(c));

    return device_HSVtoRGB(hue, saturation, value);
}

__device__ thrust::complex<double> evaluate_polynomial_given_coefficients(const thrust::complex<double>* coefficients, int degree, const thrust::complex<double>& point) {
    thrust::complex<double> result(0.0, 0.0);
    thrust::complex<double> power_of_point(1.0, 0.0);
    for (int i = 0; i <= degree; i++) {
        result += coefficients[i] * power_of_point;
        power_of_point *= point;
    }
    return result;
}

__device__ thrust::complex<double> pixel_to_point(const thrust::complex<double>& pixel, const thrust::complex<double>& lx_ty, const thrust::complex<double>& rx_by, const thrust::complex<double>& wh_complex) {
    return pixel * (rx_by - lx_ty) / wh_complex + lx_ty;
}

__global__ void render_kernel(
    int* d_pixels,
    int w,
    int h,
    const thrust::complex<double>* d_coefficients,
    int degree,
    thrust::complex<double> lx_ty,
    thrust::complex<double> rx_by
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    thrust::complex<double> xy(x, y);
    thrust::complex<double> wh_complex(w, h);
    thrust::complex<double> point = pixel_to_point(xy, lx_ty, rx_by, wh_complex);

    thrust::complex<double> val = evaluate_polynomial_given_coefficients(d_coefficients, degree, point);
    int color = complex_to_color(val);

    d_pixels[y * w + x] = color;
}

extern "C" void color_complex_polynomial(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    const double* h_coefficients_real,
    const double* h_coefficients_imag,
    int degree,
    double lx, double ty,
    double rx, double by
) {
    // Allocate device memory for pixels
    int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(int));

    // Allocate device memory for coefficients
    thrust::complex<double>* d_coefficients;
    cudaMalloc(&d_coefficients, (degree + 1) * sizeof(thrust::complex<double>));

    // Create host array of complex coeffs
    thrust::complex<double>* h_coefficients = new thrust::complex<double>[degree + 1];
    for(int i = 0; i <= degree; i++){
        h_coefficients[i] = thrust::complex<double>(h_coefficients_real[i], h_coefficients_imag[i]);
    }

    // Copy coefficients to device
    cudaMemcpy(d_coefficients, h_coefficients, (degree + 1) * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    delete[] h_coefficients;

    // Define the region in complex plane
    thrust::complex<double> lx_ty(lx, ty);
    thrust::complex<double> rx_by(rx, by);

    // Kernel config
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    render_kernel<<<gridSize, blockSize>>>(d_pixels, w, h, d_coefficients, degree, lx_ty, rx_by);
    cudaDeviceSynchronize();

    // Copy pixels back to host
    cudaMemcpy(h_pixels, d_pixels, w * h * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_coefficients);
}
