#pragma once

#include "pixels.h"

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

Pixels svg_to_pix(const string& svg, ScalingParams& scaling_params) {
    // Open svg and get its dimensions
    RsvgHandle* handle = rsvg_handle_new_from_file(svg.c_str(), NULL);
    if (!handle) {
        fprintf(stderr, "Error loading SVG data from file \"%s\"\n", svg.c_str());
        exit(-1);
    }

    // Get the intrinsic dimensions of the SVG
    RsvgDimensionData dimensions;
    rsvg_handle_get_dimensions(handle, &dimensions);

    if (scaling_params.mode == ScalingMode::BoundingBox) {
        // Calculate the scale factor to fit within the bounding box
        scaling_params.scale_factor = min(static_cast<double>(scaling_params.max_width) / dimensions.width, static_cast<double>(scaling_params.max_height) / dimensions.height);
    }
    int width  = static_cast<int>(dimensions.width  * scaling_params.scale_factor);
    int height = static_cast<int>(dimensions.height * scaling_params.scale_factor);

    Pixels ret(width, height);

    // Render the SVG
    cairo_surface_t* surface = cairo_image_surface_create_for_data(&(ret.pixels[0]), CAIRO_FORMAT_ARGB32, width, height, width * 4);
    cairo_t* cr = cairo_create(surface);
    cairo_scale(cr, scaling_params.scale_factor, scaling_params.scale_factor);
    rsvg_handle_render_cairo(handle, cr);
    g_object_unref(handle);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    ret.grayscale_to_alpha();
    return crop(ret);
}

// Create an unordered_map to store the cached results
std::unordered_map<std::string, Pixels> latex_cache;

Pixels eqn_to_pix(const string& eqn, ScalingParams& scaling_params) {
    // Check if the result is already in the cache
    auto it = latex_cache.find(eqn);
    if (it != latex_cache.end()) {
        return it->second; // Return the cached Pixels object
    }

    hash<string> hasher;
    string name = "/home/swap/CS/swaptube/out/latex/" + to_string(hasher(eqn)) + ".svg";

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
