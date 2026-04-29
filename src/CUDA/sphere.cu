#include "../Host_Device_Shared/vec.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include "../Host_Device_Shared/ThreeDimensionStructs.h"
#include "color.cuh" // Contains overlay_pixel and set_pixel
#include "common_graphics.cuh" // Contains fill_circle
#include <png.h>

extern "C" uint32_t* cuda_copy_map(const string& filename_with_or_without_suffix, int& out_width, int& out_height) {
    // Check if the filename already ends with ".png"
    string filename = filename_with_or_without_suffix;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".png") {
        filename += ".png";  // Append the ".png" suffix if it's not present
    }

    string fullpath = "io_in/" + filename;

    // Open the PNG file
    FILE* fp = fopen(fullpath.c_str(), "rb");
    if (!fp) {
        throw runtime_error("Failed to open PNG file " + fullpath);
    }

    // Create and initialize the png_struct
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw runtime_error("Failed to create png read struct.");
    }

    // Create and initialize the png_info
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        throw runtime_error("Failed to create png info struct.");
    }

    // Set up error handling (required without using the default error handlers)
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw runtime_error("Error during PNG creation.");
    }

    // Initialize input/output for libpng
    png_init_io(png, fp);
    png_read_info(png, info);

    // Get image info
    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    out_width = width;
    out_height = height;
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    if (bit_depth == 16) {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    // Prepare to read row by row and copy each row to the device without allocating the full image on the host
    png_size_t rowbytes = png_get_rowbytes(png, info);
    png_bytep row = (png_byte*)malloc(rowbytes);
    if (!row) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw runtime_error("Failed to allocate row buffer.");
    }

    uint32_t* d_map = nullptr;
    size_t map_sz = (size_t)width * (size_t)height * sizeof(uint32_t);
    cout << "Loading texture from " << fullpath << " with dimensions " << width << "x" << height << endl;
    cudaError_t cerr = cudaMalloc((void**)&d_map, map_sz);
    cout << "Attempting to allocate " << map_sz / (1024.0 * 1024.0 * 1024.0) << " GB for texture on GPU." << endl;
    if (cerr != cudaSuccess) {
        free(row);
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw runtime_error("cudaMalloc failed for PNG device buffer.");
    }

    size_t line_buf_sz = (size_t)width * sizeof(uint32_t);
    uint32_t* line_buf = (uint32_t*)malloc(line_buf_sz);
    if (!line_buf) {
        free(row);
        cudaFree(d_map);
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw runtime_error("Failed to allocate temporary line buffer.");
    }

    for (int y = 0; y < height; y++) {
        png_read_row(png, row, nullptr);
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            uint8_t r = px[0];
            uint8_t g = px[1];
            uint8_t b = px[2];
            uint8_t a = px[3];
            line_buf[x] = ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
        }
        cerr = cudaMemcpy(d_map + (y * width), line_buf, line_buf_sz, cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) {
            free(line_buf);
            free(row);
            cudaFree(d_map);
            png_destroy_read_struct(&png, &info, nullptr);
            fclose(fp);
            throw runtime_error("cudaMemcpy failed while copying PNG row to device.");
        }
    }

    free(line_buf);
    free(row);
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return d_map;
}

__device__ __forceinline__ bool ray_sphere_intersect(const Cuda::vec3& ray_origin, const Cuda::vec3& ray_dir, Cuda::vec3& out_hit_point) {
    Cuda::vec3 oc = ray_origin;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - 1;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;
    } else {
        float sqrt_disc = sqrtf(discriminant);
        float inv_2a = 0.5f / a;
        float t1 = (-b - sqrt_disc) * inv_2a;
        float t2 = (-b + sqrt_disc) * inv_2a;
        float t = (t1 > 0) ? t1 : t2; // Choose the closest positive intersection
        if (t > 0) {
            out_hit_point = ray_origin + ray_dir * t;
            return true;
        } else {
            return false;
        }
    }
}

__device__ __forceinline__ uint32_t cubic_interpolate(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, float t) {
    // Extract RGBA components
    uint8_t a0 = (v0 >> 24) & 0xFF;
    uint8_t r0 = (v0 >> 16) & 0xFF;
    uint8_t g0 = (v0 >> 8) & 0xFF;
    uint8_t b0 = v0 & 0xFF;

    uint8_t a1 = (v1 >> 24) & 0xFF;
    uint8_t r1 = (v1 >> 16) & 0xFF;
    uint8_t g1 = (v1 >> 8) & 0xFF;
    uint8_t b1 = v1 & 0xFF;

    uint8_t a2 = (v2 >> 24) & 0xFF;
    uint8_t r2 = (v2 >> 16) & 0xFF;
    uint8_t g2 = (v2 >> 8) & 0xFF;
    uint8_t b2 = v2 & 0xFF;

    uint8_t a3 = (v3 >> 24) & 0xFF;
    uint8_t r3 = (v3 >> 16) & 0xFF;
    uint8_t g3 = (v3 >> 8) & 0xFF;
    uint8_t b3 = v3 & 0xFF;

    // Cubic interpolation formula
    auto cubic_interp = [](uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3, float t) {
        return static_cast<uint8_t>(
            c1 + 0.5f * t * (c2 - c0 + t * (2.0f * c0 - 5.0f * c1 + 4.0f * c2 - c3 + t * (3.0f * (c1 - c2) + c3 - c0)))
        );
    };

    // Interpolate each channel
    uint8_t r = cubic_interp(r0, r1, r2, r3, t);
    uint8_t g = cubic_interp(g0, g1, g2, g3, t);
    uint8_t b = cubic_interp(b0, b1, b2, b3, t);
    uint8_t a = cubic_interp(a0, a1, a2, a3, t);
    return (a << 24) | (r << 16) | (g << 8) | b;
}

__device__ __forceinline__ uint32_t bicubic_sample(uint32_t* map, int map_width, int map_height, float u, float v) {
    // Convert to texture space
    float x = u * (map_width - 1);
    float y = v * (map_height - 1);
    int x0 = floorf(x);
    int y0 = floorf(y);
    float dx = x - x0;
    float dy = y - y0;

    uint32_t samples[4][4];
    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int sx = min(max(x0 + i, 0), map_width - 1);
            int sy = min(max(y0 + j, 0), map_height - 1);
            samples[j + 1][i + 1] = map[sy * map_width + sx];
        }
    }

    // Cubic interpolation in x direction
    uint32_t col[4];
    for (int j = 0; j < 4; j++) {
        col[j] = cubic_interpolate(samples[j][0], samples[j][1], samples[j][2], samples[j][3], dx);
    }
    // Cubic interpolation in y direction
    return cubic_interpolate(col[0], col[1], col[2], col[3], dy);
}

__global__ void render_sphere_kernel(
    uint32_t* pixels, int width, int height,
    float geom_mean_size,
    uint32_t* map, int map_width, int map_height,
    const Cuda::quat camera_direction, const Cuda::vec3 camera_pos, float fov, float opacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int px = idx % width;
    int py = idx / width;

    Cuda::vec3 ray_dir = get_raymarch_vector(Cuda::vec2(px, py), Cuda::vec2(width, height), fov, camera_direction);
    Cuda::vec3 hit_point;
    bool collision = ray_sphere_intersect(camera_pos, ray_dir, hit_point);
    if(!collision) return;

    float u = 0.5f + atan2f(hit_point.x, -hit_point.z) / (2.0f * M_PI);
    float v = 0.5f - asinf(hit_point.y) / M_PI;
    uint32_t color = bicubic_sample(map, map_width, map_height, u, v);
    pixels[idx] = (color & 0x00FFFFFF) | ((uint32_t)(opacity * 255) << 24);
}

extern "C" void cuda_render_sphere(
    uint32_t* h_pixels, int width, int height,
    float geom_mean_size,
    uint32_t* d_map, int map_width, int map_height,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov, float opacity)
{
    uint32_t* d_pixels = nullptr;
    size_t pix_sz = width * height * sizeof(uint32_t);

    cudaMalloc((void**)&d_pixels, pix_sz);
    cudaMemcpy(d_pixels, h_pixels, pix_sz, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (width * height + blockSize - 1) / blockSize;
    render_sphere_kernel<<<numBlocks, blockSize>>>(
        d_pixels, width, height,
        geom_mean_size,
        d_map, map_width, map_height,
        camera_direction, camera_pos, fov, opacity);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pixels, d_pixels, pix_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

extern "C" void cuda_free_map(uint32_t* d_map)
{
    cudaFree(d_map);
}
