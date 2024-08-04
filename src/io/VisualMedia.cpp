#pragma once
#include <librsvg/rsvg.h>
#include <png.h>
#include <vector>
#include <stdexcept>
#include <librsvg-2.0/librsvg/rsvg.h>
#include "../misc/pixels.h"

enum class ScalingMode {
    BoundingBox,
    ScaleFactor
};

struct ScalingParams {
    ScalingMode mode;
    int max_width;
    int max_height;
    double scale_factor;

    // Constructors for different modes
    ScalingParams(int width, int height) 
        : mode(ScalingMode::BoundingBox), max_width(width), max_height(height), scale_factor(0) {}

    ScalingParams(double factor) 
        : mode(ScalingMode::ScaleFactor), max_width(0), max_height(0), scale_factor(factor) {}
};

Pixels png_to_pix(const string& filename) {
    // Open the PNG file
    FILE* fp = fopen((PATH_MANAGER.this_project_media_dir + filename + ".png").c_str(), "rb");
    if (!fp) {
        throw runtime_error("Failed to open PNG file.");
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

    // Read image data
    vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }
    png_read_image(png, row_pointers.data());

    // Create a Pixels object
    Pixels ret(width, height);

    // Copy data to Pixels object
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            uint8_t r = px[0];
            uint8_t g = px[1];
            uint8_t b = px[2];
            uint8_t a = px[3];
            ret.set_pixel(x, y, argb_to_col(a, r, g, b));
        }
    }

    // Free memory and close file
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return ret;
}

Pixels svg_to_pix(const string& svg, ScalingParams& scaling_params) {
    // Open svg and get its dimensions
    RsvgHandle* handle = rsvg_handle_new_from_file(svg.c_str(), NULL);
    if (!handle) {
        fprintf(stderr, "Error loading SVG data from file \"%s\"\n", svg.c_str());
        exit(-1);
    }

    // Get the intrinsic dimensions of the SVG
    RsvgDimensionData dimension = { 0 };
    rsvg_handle_get_dimensions(handle, &dimension);

    //gdouble out_width, out_height;
    //rsvg_handle_get_intrinsic_size_in_pixels(handle, &out_width, &out_height);

    if (scaling_params.mode == ScalingMode::BoundingBox) {
        // Calculate the scale factor to fit within the bounding box
        scaling_params.scale_factor = min(static_cast<double>(scaling_params.max_width) / dimension.width, static_cast<double>(scaling_params.max_height) / dimension.height);
    }
    int width  = static_cast<int>(dimension.width  * scaling_params.scale_factor);
    int height = static_cast<int>(dimension.height * scaling_params.scale_factor);

    Pixels ret(width, height);

    // Create a uint8_t array to store the raw pixel data
    vector<uint8_t> raw_data(width * height * 4);

    // Render the SVG
    cairo_surface_t* surface = cairo_image_surface_create_for_data(raw_data.data(), CAIRO_FORMAT_ARGB32, width, height, width * 4);
    cairo_t* cr = cairo_create(surface);
    cairo_scale(cr, scaling_params.scale_factor, scaling_params.scale_factor);
    rsvg_handle_render_cairo(handle, cr);

    // Copy the uint8_t array to the Pixels object
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = (y * width + x) * 4;
            ret.set_pixel(x, y, argb_to_col(raw_data[offset + 3], raw_data[offset + 2], raw_data[offset + 1], raw_data[offset]));
        }
    }

    // Clean up
    g_object_unref(handle);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    ret.grayscale_to_alpha();
    return crop(ret);
}

// Create an unordered_map to store the cached results
unordered_map<string, Pixels> latex_cache;

/*
 * We use MicroTEX to convert LaTeX equations into svg files.
 */
Pixels eqn_to_pix(const string& eqn, ScalingParams& scaling_params) {
    // Check if the result is already in the cache
    auto it = latex_cache.find(eqn);
    if (it != latex_cache.end()) {
        return it->second; // Return the cached Pixels object
    }

    hash<string> hasher;
    char full_directory_path[PATH_MAX];
    realpath(PATH_MANAGER.latex_dir.c_str(), full_directory_path);
    string name = string(full_directory_path) + "/" + to_string(hasher(eqn)) + ".svg";

    if (access(name.c_str(), F_OK) != -1) {
        // File already exists, no need to generate LaTeX
        Pixels pixels = svg_to_pix(name, scaling_params);
        latex_cache[eqn] = pixels; // Cache the result before returning
        return pixels;
    }

    string command = "cd ../../MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + eqn + "\" -output=" + name + " >/dev/null 2>&1";
    int result = system(command.c_str());

    if (result == 0) {
        // System call successful, return the generated SVG
        Pixels pixels = svg_to_pix(name, scaling_params);
        latex_cache[eqn] = pixels; // Cache the result before returning
        return pixels;
    } else {
        // System call failed, handle the error
        throw runtime_error("Failed to generate LaTeX.");
    }
}
