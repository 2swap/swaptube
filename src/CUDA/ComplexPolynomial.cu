#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float lerp(float a, float b, float w){
    return w*b+(1-w)*a;
}

__device__ int complex_to_color(const thrust::complex<float>& c, float ab_dilation, float dot_radius) {
    float mag = abs(c);
    thrust::complex<float> norm = (c * ab_dilation / mag + thrust::complex<float>(1,1)) * .5;
    float am = 2*atan(mag/dot_radius)/M_PI;
    return device_OKLABtoRGB(255, (1-.8*am)*1, lerp(-.233888, .276216, norm.real()), lerp(-.311528, .198570, norm.imag()));
}

__device__ thrust::complex<float> evaluate_polynomial_given_coefficients(const thrust::complex<float>* coefficients, int degree, const thrust::complex<float>& point) {
    thrust::complex<float> result(0.0, 0.0);
    thrust::complex<float> power_of_point(1.0, 0.0);
    for (int i = 0; i <= degree; i++) {
        result += coefficients[i] * power_of_point;
        power_of_point *= point;
    }
    return result;
}

__device__ glm::vec2 pixel_to_point(const glm::vec2& pixel, const glm::vec2& lx_ty, const glm::vec2& rx_by, const glm::vec2& wh) {
    const glm::vec2 flip(pixel.x, wh.y-1-pixel.y);
    return flip * (rx_by - lx_ty) / wh + lx_ty;
}

__global__ void render_kernel(
    int* d_pixels,
    const thrust::complex<float>* d_coefficients,
    int degree,
    glm::vec2 wh,
    glm::vec2 lx_ty,
    glm::vec2 rx_by,
    float ab_dilation,
    float dot_radius
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wh.x || y >= wh.y) return;

    const glm::vec2 point = pixel_to_point(glm::vec2(x,y), lx_ty, rx_by, wh);
    const thrust::complex<float> val = evaluate_polynomial_given_coefficients(d_coefficients, degree, thrust::complex<float>(point.x, point.y));
    const int color = complex_to_color(val, ab_dilation, dot_radius);

    d_pixels[y * int(wh.x) + x] = color;
}

extern "C" void color_complex_polynomial(
    unsigned int* h_pixels, // to be overwritten with the result
    int w,
    int h,
    const float* h_coefficients_real,
    const float* h_coefficients_imag,
    int degree,
    float lx, float ty,
    float rx, float by,
    float ab_dilation,
    float dot_radius
) {
    // Allocate device memory for pixels
    int* d_pixels;
    cudaMalloc(&d_pixels, w * h * sizeof(int));

    // Allocate device memory for coefficients
    thrust::complex<float>* d_coefficients;
    cudaMalloc(&d_coefficients, (degree + 1) * sizeof(thrust::complex<float>));

    // Create host array of complex coeffs
    thrust::complex<float>* h_coefficients = new thrust::complex<float>[degree + 1];
    for(int i = 0; i <= degree; i++){
        h_coefficients[i] = thrust::complex<float>(h_coefficients_real[i], h_coefficients_imag[i]);
    }

    // Copy coefficients to device
    cudaMemcpy(d_coefficients, h_coefficients, (degree + 1) * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
    delete[] h_coefficients;

    // Define the region in complex plane
    glm::vec2 wh(w, h);
    glm::vec2 lx_ty(lx, ty);
    glm::vec2 rx_by(rx, by);

    // Kernel config
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    render_kernel<<<gridSize, blockSize>>>(d_pixels, d_coefficients, degree, wh, lx_ty, rx_by, ab_dilation, dot_radius);
    cudaDeviceSynchronize();

    // Copy pixels back to host
    cudaMemcpy(h_pixels, d_pixels, w * h * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_coefficients);
}
