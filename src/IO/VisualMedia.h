#pragma once

#include <string>
#include "../Core/Pixels.h"

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

void pix_to_png(const Pixels& pix, const std::string& filename);
Pixels svg_to_pix(const std::string& filename_with_or_without_suffix, ScalingParams& scaling_params);
void png_to_pix(Pixels& pix, const std::string& filename_with_or_without_suffix);
void png_to_pix_bounding_box(Pixels& pix, const std::string& filename, int w, int h);
Pixels latex_to_pix(const std::string& latex, ScalingParams& scaling_params);
void pdf_page_to_pix(Pixels& pix, const std::string& pdf_filename_without_suffix, const int page_number);
