#pragma once
#include <png.h>
#include <vector>
#include <stdexcept>
#include <librsvg-2.0/librsvg/rsvg.h>
#include "../misc/pixels.h"
#include <sys/stat.h>
#include <cmath>
#include <sstream>

enum class ScalingMode {
    BoundingBox,
    ScaleFactor
};

struct ScalingParams {
    ScalingMode mode;
    float max_width;
    float max_height;
    double scale_factor;

    // Constructors for different modes
    ScalingParams(float width, float height) 
        : mode(ScalingMode::BoundingBox), max_width(width), max_height(height), scale_factor(0) {}

    ScalingParams(double factor) 
        : mode(ScalingMode::ScaleFactor), max_width(0), max_height(0), scale_factor(factor) {}
};

void pix_to_png(const Pixels& pix, const string& filename) {
    if(pix.w * pix.h == 0) return; // cowardly exit.

    // Open the file for writing (binary mode)
    FILE* fp = fopen(("io_out/" + filename + ".png").c_str(), "wb");
    if (!fp) {
        throw runtime_error("Failed to open png file for writing: " + filename);
    }

    // Initialize write structure
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw runtime_error("Failed to create png write struct.");
    }

    // Initialize info structure
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        throw runtime_error("Failed to create png info struct.");
    }

    // Set up error handling (required without using the default error handlers)
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw runtime_error("Error during PNG creation.");
    }

    // Set up output control
    png_init_io(png, fp);

    // Write header (8 bit color depth)
    png_set_IHDR(png, info, pix.w, pix.h,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Allocate memory for one row
    png_bytep row = (png_bytep)malloc(4 * pix.w * sizeof(png_byte));
    if (!row) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw runtime_error("Failed to allocate memory for row.");
    }

    // Write image data
    for (int y = 0; y < pix.h; y++) {
        for (int x = 0; x < pix.w; x++) {
            int pixel = pix.get_pixel(x, y);
            uint8_t a = (pixel >> 24) & 0xFF;
            uint8_t r = (pixel >> 16) & 0xFF;
            uint8_t g = (pixel >> 8) & 0xFF;
            uint8_t b = pixel & 0xFF;
            row[x*4 + 0] = r;
            row[x*4 + 1] = g;
            row[x*4 + 2] = b;
            row[x*4 + 3] = a;
        }
        png_write_row(png, row);
    }

    // End write
    png_write_end(png, nullptr);

    // Free allocated memory
    free(row);

    // Cleanup
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

Pixels svg_to_pix(const string& filename_with_or_without_suffix, ScalingParams& scaling_params) {
    // Check if the filename already ends with ".svg"
    string filename = filename_with_or_without_suffix;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".svg") {
        filename += ".svg";  // Append the ".svg" suffix if it's not present
    }

    // Load SVG
    GError* error = nullptr;
    RsvgHandle* handle = rsvg_handle_new_from_file(filename.c_str(), &error);
    if (!handle) {
        string error_str = "Error loading SVG file " + filename + ": " + error->message;
        g_error_free(error);
        throw runtime_error(error_str);
    }

    gdouble gwidth, gheight;
    if (!rsvg_handle_get_intrinsic_size_in_pixels(handle, &gwidth, &gheight))
        throw runtime_error("Could not get intrinsic size of SVG file " + filename);

    // Calculate scale factor
    if (scaling_params.mode == ScalingMode::BoundingBox) {
        scaling_params.scale_factor = min(
            static_cast<double>(scaling_params.max_width) / gwidth,
            static_cast<double>(scaling_params.max_height) / gheight
        );
    } else if (scaling_params.scale_factor <= 0) {
        throw runtime_error("Invalid scale factor: " + to_string(scaling_params.scale_factor));
    }

    int width  = round(gwidth  * scaling_params.scale_factor);
    int height = round(gheight * scaling_params.scale_factor);

    if (width <= 0 || height <= 0) {
        g_object_unref(handle);
        throw runtime_error("Computed output size for SVG file " + filename + " is invalid: width=" + to_string(width) + ", height=" + to_string(height) + ", scaling factor=" + to_string(scaling_params.scale_factor));
    }

    Pixels ret(width, height);

    // Allocate pixel buffer
    vector<uint8_t> raw_data(width * height * 4, 0);

    // Create cairo surface and context
    cairo_surface_t* surface = cairo_image_surface_create_for_data(
        raw_data.data(), CAIRO_FORMAT_ARGB32, width, height, width * 4
    ); 
    cairo_t* cr = cairo_create(surface);

    // Set scale
    cairo_scale(cr, scaling_params.scale_factor, scaling_params.scale_factor);

    // Define viewport for rendering
    RsvgRectangle viewport = {
        .x = 0,
        .y = 0,
        .width = gwidth,
        .height = gheight
    };

    if (viewport.width <= 0 || viewport.height <= 0) {
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw runtime_error("Invalid viewport size for SVG file " + filename);
    }

    // Render SVG
    if (!rsvg_handle_render_document(handle, cr, &viewport, &error)) {
        g_error_free(error);
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw runtime_error("Failed to render SVG file " + filename + ": " + error->message);
    }

    // Copy pixels into Pixels object
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = (y * width + x) * 4;
            ret.set_pixel(x, y, argb(
                raw_data[offset + 3],  // Alpha
                raw_data[offset + 2],  // Red
                raw_data[offset + 1],  // Green
                raw_data[offset]       // Blue
            ));
        }
    }

    // Cleanup
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    g_object_unref(handle);

    //ret.grayscale_to_alpha();
    return crop(ret);
}

Pixels png_to_pix(const string& filename_with_or_without_suffix) {
    // Check if the filename already ends with ".png"
    string filename = filename_with_or_without_suffix;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".png") {
        filename += ".png";  // Append the ".png" suffix if it's not present
    }

    // Open the PNG file
    FILE* fp = fopen(("io_in/" + filename).c_str(), "rb");
    if (!fp) {
        throw runtime_error("Failed to open PNG file " + filename);
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
            ret.set_pixel(x, y, argb(a, r, g, b));
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

Pixels png_to_pix_bounding_box(const string& filename, int w, int h) {
    Pixels image = png_to_pix(filename);

    // Calculate the scaling factor based on the bounding box
    float scale = min(static_cast<float>(w) / image.w, static_cast<float>(h) / image.h);

    // Calculate the new dimensions
    int new_width = static_cast<int>(image.w * scale);
    int new_height = static_cast<int>(image.h * scale);

    // Scale the image using bicubic interpolation
    return image.bicubic_scale(new_width, new_height);
}

// Create an unordered_map to store the cached results
unordered_map<string, pair<Pixels, double>> latex_cache;

string generate_cache_key(const string& text, const ScalingParams& scaling_params) {
    hash<string> hasher;
    string key = text + "_" + to_string(static_cast<int>(scaling_params.mode)) + "_" + 
                 to_string(scaling_params.max_width) + "_" + 
                 to_string(scaling_params.max_height) + "_" + 
                 to_string(scaling_params.scale_factor);
    return to_string(hasher(key));
}

/*
 * We use MicroTEX to convert LaTeX equations into svg files.
 */
Pixels latex_to_pix(const string& latex, ScalingParams& scaling_params) {
    // Generate a cache key based on the equation and scaling parameters
    string cache_key = generate_cache_key(latex, scaling_params);

    // Check if the result is already in the cache
    auto it = latex_cache.find(cache_key);
    if (it != latex_cache.end()) {
        scaling_params.scale_factor = it->second.second;
        return it->second.first; // Return the cached Pixels object
    }

    hash<string> hasher;
    char full_directory_path[PATH_MAX];
    string latex_dir = "io_in/latex/";
    realpath(latex_dir.c_str(), full_directory_path);
    string name = string(full_directory_path) + "/" + to_string(hasher(latex)) + ".svg";

    if (access(name.c_str(), F_OK) == -1) {
        string command = "cd ../../MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + latex + "\" -output=" + name + " >/dev/null 2>&1";
        int result = system(command.c_str());
        if(result != 0) {
            cout << command << endl;
            throw runtime_error("Failed to generate LaTeX. Command printed above.");
        }
    }

    // System call successful, return the generated SVG
    Pixels pixels = svg_to_pix(name, scaling_params);
    latex_cache[cache_key] = make_pair(pixels, scaling_params.scale_factor); // Cache the result before returning
    return pixels;
}
