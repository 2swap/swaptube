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
#include "../Core/Pixels.h"

using namespace std;

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

DevicePointer svg_to_gpu_pix(const string& filename_with_or_without_suffix, ScalingParams& scaling_params) {
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
        auto e = runtime_error("Failed to render SVG file " + filename + ": " + error->message);
        g_error_free(error);
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw e;
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

    DevicePointer dp(ret.wh);
    dp.copy_to_device(ret.pixels.data());

    return dp;
}
