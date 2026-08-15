#pragma once

#include <string>
#include "SVG.h"
#include "../Host_Device_Shared/vec.h"
#include "ScalingParams.h"
#include <memory>
#include <cstdint>

string latex_color(uint32_t color, string text);

shared_ptr<DevicePointer> latex_to_gpu_pix(const std::string& latex, ScalingParams& scaling_params);
void write_text(uint32_t* gpu_pix, const ivec2& canvas_wh, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle_rad);
