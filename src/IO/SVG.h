#pragma once

#include <string>
#include "../Core/Pixels.h"
#include "../Host_Device_Shared/vec.h"
#include "ScalingParams.h"
#include <cstdint>

string latex_color(uint32_t color, string text);

Pixels svg_to_pix(const std::string& filename_with_or_without_suffix, ScalingParams& scaling_params);
Pixels latex_to_pix(const std::string& latex, ScalingParams& scaling_params);
void write_text(Pixels& pix, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle = 0);
