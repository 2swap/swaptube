#pragma once

#include <string>
#include "../Core/Pixels.h"
#include "../Host_Device_Shared/vec.h"
#include <cstdint>

enum class ScalingMode {
    BoundingBox,
    ScaleFactor
};

struct ScalingParams {
    ScalingMode mode;
    vec2 bounding_box;
    double scale_factor;

    // Constructors for different modes
    ScalingParams(vec2 bb)
        : mode(ScalingMode::BoundingBox), bounding_box(bb), scale_factor(0) {}

    ScalingParams(double factor) 
        : mode(ScalingMode::ScaleFactor), bounding_box(0, 0), scale_factor(factor) {}
};

void pix_to_png(const Pixels& pix, const std::string& filename);
Pixels svg_to_pix(const std::string& filename_with_or_without_suffix, ScalingParams& scaling_params);
void png_to_pix(Pixels& pix, const std::string& filename_with_or_without_suffix);
void png_to_raw_data(uint32_t*& unallocated_data, int& width, int& height, const string& filename_with_or_without_suffix);
void png_to_pix_bounding_box(Pixels& pix, const std::string& filename, int w, int h);
Pixels latex_to_pix(const std::string& latex, ScalingParams& scaling_params);
void write_text(Pixels& pix, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle = 0);
void pdf_page_to_pix(Pixels& pix, const std::string& pdf_filename_without_suffix, const int page_number);
