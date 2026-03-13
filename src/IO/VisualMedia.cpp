#include "VisualMedia.h"

#include <filesystem>
#include <png.h>
#include <vector>
#include <stdexcept>
#include <librsvg-2.0/librsvg/rsvg.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <cairo.h>
#include <iostream>
#include <cstdint>

using namespace std;
namespace fs = std::filesystem;

void pix_to_png(const Pixels& pix, const string& filename) {
    if(pix.w * pix.h == 0) return; // cowardly exit.

#ifdef _WIN32
    cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, pix.w, pix.h);
    if (!surface || cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        if (surface) cairo_surface_destroy(surface);
        throw runtime_error("Failed to create cairo surface for PNG writing: " + filename);
    }

    unsigned char* data = cairo_image_surface_get_data(surface);
    const int stride = cairo_image_surface_get_stride(surface);
    for (int y = 0; y < pix.h; y++) {
        unsigned char* row = data + y * stride;
        for (int x = 0; x < pix.w; x++) {
            const int pixel = pix.get_pixel_carelessly(x, y);
            const uint8_t a = (pixel >> 24) & 0xFF;
            uint8_t r = (pixel >> 16) & 0xFF;
            uint8_t g = (pixel >> 8) & 0xFF;
            uint8_t b = pixel & 0xFF;

            // Cairo ARGB32 expects premultiplied channels.
            if (a != 255) {
                r = static_cast<uint8_t>((static_cast<int>(r) * a + 127) / 255);
                g = static_cast<uint8_t>((static_cast<int>(g) * a + 127) / 255);
                b = static_cast<uint8_t>((static_cast<int>(b) * a + 127) / 255);
            }

            unsigned char* px = row + x * 4;
            px[0] = b;
            px[1] = g;
            px[2] = r;
            px[3] = a;
        }
    }
    cairo_surface_mark_dirty(surface);

    const string outpath = "io_out/" + filename + ".png";
    const cairo_status_t status = cairo_surface_write_to_png(surface, outpath.c_str());
    cairo_surface_destroy(surface);
    if (status != CAIRO_STATUS_SUCCESS) {
        throw runtime_error("Failed to write PNG file " + outpath + " via cairo.");
    }
    return;
#endif

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
            int pixel = pix.get_pixel_carelessly(x, y);
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

    if (gwidth <= 0 || gheight <= 0) {
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw runtime_error("Invalid viewport size for SVG file " + filename);
    }

#if LIBRSVG_CHECK_VERSION(2, 52, 0)
    RsvgRectangle viewport;
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = gwidth;
    viewport.height = gheight;
    if (!rsvg_handle_render_document(handle, cr, &viewport, &error)) {
        string msg = (error != nullptr) ? error->message : "unknown error";
        if (error != nullptr) g_error_free(error);
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw runtime_error("Failed to render SVG file " + filename + ": " + msg);
    }
#else
    if (!rsvg_handle_render_cairo(handle, cr)) {
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        g_object_unref(handle);
        throw runtime_error("Failed to render SVG file " + filename + " using legacy librsvg API.");
    }
#endif

    // Copy pixels into Pixels object
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = (y * width + x) * 4;
            ret.set_pixel_carelessly(x, y, argb(
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

    return crop_by_alpha(ret);
}

void png_to_pix(Pixels& pix, const string& filename_with_or_without_suffix) {
    // Check if the filename already ends with ".png"
    string filename = filename_with_or_without_suffix;
    if (filename.length() < 4 || filename.substr(filename.length() - 4) != ".png") {
        filename += ".png";  // Append the ".png" suffix if it's not present
    }

    string fullpath = "io_in/" + filename;

    // Check cache
    static unordered_map<string, Pixels> png_cache;
    auto it = png_cache.find(fullpath);
    if (it != png_cache.end()) {
        pix = it->second;
        return;
    }

#ifdef _WIN32
    cairo_surface_t* surface = cairo_image_surface_create_from_png(fullpath.c_str());
    if (!surface || cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        string msg = surface ? cairo_status_to_string(cairo_surface_status(surface)) : "unknown error";
        if (surface) cairo_surface_destroy(surface);
        throw runtime_error("Failed to open PNG file " + fullpath + " via cairo: " + msg);
    }

    cairo_surface_flush(surface);
    const int surface_width = cairo_image_surface_get_width(surface);
    const int surface_height = cairo_image_surface_get_height(surface);
    const int stride = cairo_image_surface_get_stride(surface);
    unsigned char* data = cairo_image_surface_get_data(surface);
    if (surface_width <= 0 || surface_height <= 0 || data == nullptr) {
        cairo_surface_destroy(surface);
        throw runtime_error("Invalid PNG surface for " + fullpath);
    }

    pix = Pixels(surface_width, surface_height);
    for (int y = 0; y < surface_height; y++) {
        unsigned char* row = data + y * stride;
        for (int x = 0; x < surface_width; x++) {
            unsigned char* px = row + x * 4;
            const uint8_t b_premul = px[0];
            const uint8_t g_premul = px[1];
            const uint8_t r_premul = px[2];
            const uint8_t a = px[3];

            uint8_t r = r_premul;
            uint8_t g = g_premul;
            uint8_t b = b_premul;
            if (a != 0 && a != 255) {
                r = static_cast<uint8_t>(min(255, (static_cast<int>(r_premul) * 255 + a / 2) / a));
                g = static_cast<uint8_t>(min(255, (static_cast<int>(g_premul) * 255 + a / 2) / a));
                b = static_cast<uint8_t>(min(255, (static_cast<int>(b_premul) * 255 + a / 2) / a));
            }
            pix.set_pixel_carelessly(x, y, argb(a, r, g, b));
        }
    }

    cairo_surface_destroy(surface);
    png_cache[fullpath] = pix;
    return;
#endif

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
    pix = Pixels(width, height);

    // Copy data to Pixels object
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            uint8_t r = px[0];
            uint8_t g = px[1];
            uint8_t b = px[2];
            uint8_t a = px[3];
            pix.set_pixel_carelessly(x, y, argb(a, r, g, b));
        }
    }

    // Free memory and close file
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    // Store in cache
    png_cache[fullpath] = pix;
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

void png_to_pix_bounding_box(Pixels& pix, const string& filename, int w, int h) {
    static unordered_map<pair<string, pair<int, int>>, Pixels, StringIntPairHash, StringIntPairEq> png_bounding_box_cache;
    auto key = make_pair(filename, make_pair(w, h));
    auto it = png_bounding_box_cache.find(key);
    if (it != png_bounding_box_cache.end()) {
        pix = it->second;
        return;
    }

    Pixels image;
    png_to_pix(image, filename);

    image.scale_to_bounding_box(w, h, pix);

    // Store in cache with scale
    png_bounding_box_cache[key] = pix;
}

// Create an unordered_map to store the cached results
unordered_map<string, pair<Pixels, double>> latex_cache;

static string generate_cache_key(const string& text, const ScalingParams& scaling_params) {
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

    cout << "Generating LaTeX for: " << latex << endl;

    hash<string> hasher;
    string latex_dir = "io_in/latex/";
    string name_without_folder = to_string(hasher(latex)) + ".svg";
    string name = (fs::absolute(latex_dir) / name_without_folder).string();

    if (!fs::exists(name)) {
#ifdef _WIN32
        string output_name = name;
        for (char& c : output_name) {
            if (c == '\\') {
                c = '/';
            }
        }
        string microtex_build_dir = fs::absolute("../../MicroTeX-master/build-mingw-headless").string();
        bool use_mingw_headless = fs::exists(microtex_build_dir);
        if (!use_mingw_headless) {
            microtex_build_dir = fs::absolute("../../MicroTeX-master/build").string();
        }
        string command =
            "cd /d \"" + microtex_build_dir + "\" && "
            ".\\LaTeX.exe -headless -foreground=#ffffffff "
            "\"-input=" + latex + "\" "
            "\"-output=" + output_name + "\"";
#else
        string command =
            "cd ../../MicroTeX-master/build/ && "
            "./LaTeX -headless -foreground=#ffffffff "
            "\"-input=" + latex + "\" "
            "\"-output=" + name + "\" >/dev/null 2>&1";
#endif
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

void pdf_page_to_pix(Pixels& pix, const string& pdf_filename_without_suffix, const int page_number) {
    if (page_number < 1) {
        throw runtime_error("PDF page number is 1-indexed and should be positive.");
    }
    if (page_number >= 100) {
        throw runtime_error("PDF page number too large; pdf_page_to_pix only supports up to 99 pages. (TODO)");
    }

    // HOW TO MAKE PAGES:
    // pdftocairo -png -f 1 -l 3 -r 300 paper.pdf prefix
    // (This makes 3 pages at 300 DPI, named prefix-01.png, prefix-02.png, prefix-03.png)
    const string resolved_filename_without_suffix = "io_in/" + pdf_filename_without_suffix;
    if (resolved_filename_without_suffix.length() >= 4 && resolved_filename_without_suffix.substr(resolved_filename_without_suffix.length() - 4) == ".pdf") {
        throw runtime_error("pdf_page_to_pix: please provide the pdf filename without the .pdf suffix.");
    }
    const string resolved_filename_with_suffix = resolved_filename_without_suffix + ".pdf";

    const string png_filename = resolved_filename_without_suffix + "-" + (page_number < 10 ? "0" : "") + to_string(page_number) + ".png";

    bool png_file_exists = fs::exists(png_filename);

    // Execute pdftocairo command to convert the specified page to PNG
    if (!png_file_exists) {
        cout << "Converting PDF page " << page_number << " to PNG..." << endl;
        const string page_number_str = to_string(page_number);
        const string command = "pdftocairo -png -f " + page_number_str + " -l " + page_number_str + " -r 300 " + resolved_filename_with_suffix + " " + resolved_filename_without_suffix;
        int result = system(command.c_str());
        if (result != 0) {
            throw runtime_error("Failed to convert PDF page to PNG using pdftocairo.");
        }
    }

    // Verify that the PNG file was created
    if (!fs::exists(png_filename)) {
        throw runtime_error("PNG file was not created after pdftocairo command.");
    }

    // Load the generated PNG into the provided Pixels object
    const string png_filename_without_prefix = png_filename.substr(6);
    png_to_pix(pix, png_filename_without_prefix);
}
