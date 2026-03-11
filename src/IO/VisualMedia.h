#pragma once

#include <string>
#include "../Core/Pixels.h"

enum class ScalingMode {
    BoundingBox,
    ScaleFactor
};

struct ScalingParams {
    ScalingMode mode;
    vec2 max_dimensions;
    double scale_factor;

    // Constructors for different modes
    ScalingParams(const vec2& dimensions)
        : mode(ScalingMode::BoundingBox), max_dimensions(dimensions), scale_factor(0) {}

    ScalingParams(double factor) 
        : mode(ScalingMode::ScaleFactor), max_dimensions(), scale_factor(factor) {}
};

void pix_to_png(const Pixels& pix, const std::string& filename);
Pixels svg_to_pix(const std::string& filename_with_or_without_suffix, ScalingParams& scaling_params);
void png_to_pix(Pixels& pix, const std::string& filename_with_or_without_suffix);
void png_to_pix_bounding_box(Pixels& pix, const string& filename, const vec2& box);
Pixels latex_to_pix(const std::string& latex, ScalingParams& scaling_params);
void pdf_page_to_pix(Pixels& pix, const std::string& pdf_filename_without_suffix, const int page_number);
