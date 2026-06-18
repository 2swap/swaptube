#include "PNG.h"

#include <png.h>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <limits.h>
#include <unistd.h>
#include <cairo.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <iostream>
#include "ScalingParams.h"

using namespace std;

void pix_to_png(const Pixels& pix, const string& filename) {
    if(pix.wh.x * pix.wh.y == 0) return; // cowardly exit.

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
    png_set_IHDR(png, info, pix.wh.x, pix.wh.y,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Allocate memory for one row
    png_bytep row = (png_bytep)malloc(4 * pix.wh.x * sizeof(png_byte));
    if (!row) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw runtime_error("Failed to allocate memory for row.");
    }

    // Write image data
    for (int y = 0; y < pix.wh.y; y++) {
        for (int x = 0; x < pix.wh.x; x++) {
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
    pix = Pixels(ivec2(width, height));

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

    struct stat buffer;
    bool png_file_exists = false;
    if (stat(png_filename.c_str(), &buffer) == 0) {
        png_file_exists = true;
    }

    // Execute pdftocairo command to convert the specified page to PNG
    if (!png_file_exists) {
        cout << "Converting PDF page " << page_number << " to PNG..." << endl;
        const string page_number_str = to_string(page_number);
        const string command = "pdftocairo -png -f " + page_number_str + " -l " + page_number_str + " -r 300 " + resolved_filename_with_suffix + " " + resolved_filename_without_suffix;
        int result = system(command.c_str());
        if (result != 0) {
            throw runtime_error("Failed to convert PDF page to PNG using pdftocairo. Command executed:\n" + command);
        }
    }

    // Verify that the PNG file was created
    if (stat(png_filename.c_str(), &buffer) != 0) {
        throw runtime_error("PNG file was not created after pdftocairo command.");
    }

    // Load the generated PNG into the provided Pixels object
    const string png_filename_without_prefix = png_filename.substr(6);
    png_to_pix(pix, png_filename_without_prefix);
}
