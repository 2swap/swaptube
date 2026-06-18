#include "SVG.h"

#include <vector>
#include <stdexcept>
#include <librsvg-2.0/librsvg/rsvg.h>
#include <sys/stat.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <limits.h>
#include <unistd.h>
#include <cairo.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <iostream>

using namespace std;

string latex_color(uint32_t color, string text) {
    // Mask out the alpha channel
    uint32_t rgb = color & 0x00FFFFFF;

    // Convert to a hex string
    stringstream ss;
    ss << "\\textcolor{#" << hex << setw(6) << setfill('0') << rgb << "}{" << text << "}";

    return ss.str();
}

static gboolean get_svg_intrinsic_size(RsvgHandle *handle, gdouble* width, gdouble* height) {
    #if LIBRSVG_CHECK_VERSION(2, 52, 0)
        return rsvg_handle_get_intrinsic_size_in_pixels(handle, width, height);
    #else
        RsvgDimensionData dim;
        rsvg_handle_get_dimensions(handle, &dim);
        if (dim.width <= 0 || dim.height <= 0) return FALSE;
        *width  = dim.width;
        *height = dim.height;
        return TRUE;
    #endif
}

Pixels svg_to_pix(const string& filename_with_or_without_suffix, ScalingParams& scaling_params) {
    // Check if the filename already ends with ".svg"
    string filename = "io_in/" + filename_with_or_without_suffix;
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
    if (!get_svg_intrinsic_size(handle, &gwidth, &gheight))
        throw runtime_error("Could not get intrinsic size of SVG file " + filename);

    // Calculate scale factor
    if (scaling_params.mode == ScalingMode::BoundingBox) {
        scaling_params.scale_factor = min(
            static_cast<double>(scaling_params.bounding_box.x) / gwidth,
            static_cast<double>(scaling_params.bounding_box.y) / gheight
        );
    } else if (scaling_params.scale_factor <= 0) {
        throw runtime_error("Invalid scale factor: " + to_string(scaling_params.scale_factor));
    }

    ivec2 wh = floor(vec2(gwidth, gheight) * scaling_params.scale_factor);

    if (wh.x <= 0 || wh.y <= 0) {
        g_object_unref(handle);
        throw runtime_error("Computed output size for SVG file " + filename + " is invalid: width=" + to_string(wh.x) + ", height=" + to_string(wh.y) + ", scaling factor=" + to_string(scaling_params.scale_factor));
    }

    Pixels copy(wh);

    // Allocate pixel buffer
    vector<uint8_t> raw_data(wh.x * wh.y * 4, 0);

    // Create cairo surface and context
    cairo_surface_t* surface = cairo_image_surface_create_for_data(
        raw_data.data(), CAIRO_FORMAT_ARGB32, wh.x, wh.y, wh.x * 4
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
    for (int y = 0; y < wh.y; ++y) {
        for (int x = 0; x < wh.x; ++x) {
            int offset = (y * wh.x + x) * 4;
            copy.set_pixel_carelessly(x, y, argb(
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

    Pixels ret;
    copy.crop_by_alpha(ret);
    return ret;
}

// Custom hash and equality for pair<string, pair<int,int>>
struct StringIntPairHash {
    size_t operator()(const pair<string, pair<int, int>>& p) const noexcept {
        size_t h1 = std::hash<string>{}(p.first);
        size_t h2 = (static_cast<size_t>(p.second.first) << 32) ^ static_cast<size_t>(p.second.second);
        // boost-like mix
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
    }
};
struct StringIntPairEq {
    bool operator()(const pair<string, pair<int, int>>& a, const pair<string, pair<int, int>>& b) const noexcept {
        return a.first == b.first && a.second.first == b.second.first && a.second.second == b.second.second;
    }
};

// Create an unordered_map to store the cached results
unordered_map<string, pair<Pixels, double>> latex_cache;

static string generate_cache_key(const string& text, const ScalingParams& scaling_params) {
    hash<string> hasher;
    string key = text + "_" + to_string(static_cast<int>(scaling_params.mode)) + "_" + 
                 to_string(scaling_params.bounding_box.x) + "_" + 
                 to_string(scaling_params.bounding_box.y) + "_" +
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

    cout << "Generating LaTeX for: " << latex << endl;

    hash<string> hasher;
    char full_directory_path[PATH_MAX];
    string latex_dir = "io_in/latex/";
    realpath(latex_dir.c_str(), full_directory_path);
    string name_without_folder = to_string(hasher(latex)) + ".svg";
    string name = string(full_directory_path) + "/" + name_without_folder;

    if (access(name.c_str(), F_OK) == -1) {
        string command = "cd ../../MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + latex + "\" -output=" + name + " >/dev/null 2>&1";
        int result = system(command.c_str());
        if(result != 0) {
            cout << command << endl;
            throw runtime_error("Failed to generate LaTeX. Command printed above.");
        }
    }

    // System call successful, return the generated SVG
    Pixels pixels = svg_to_pix("latex/" + name_without_folder, scaling_params);
    latex_cache[cache_key] = make_pair(pixels, scaling_params.scale_factor); // Cache the result before returning
    return pixels;
}

void write_text(Pixels& pix, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle) {
    ScalingParams scaling_params(text_envelope);

    Pixels text = latex_to_pix(latex, scaling_params);
    if(abs(angle) > 0.001)
        pix.overlay_cpu_with_rotation(text, center, opacity, angle);
    else
        pix.overlay_cpu(text, center, opacity);
}
