#pragma once

#include <string>
#include "SVG.h"
#include "../Host_Device_Shared/vec.h"
#include "ScalingParams.h"
#include <memory>
#include <cstdint>
#include <vector>

string latex_color(uint32_t color, string text);

shared_ptr<DevicePointer> latex_to_gpu_pix(const std::string& latex, ScalingParams& scaling_params);

inline constexpr uint32_t GLYPH_ID_RULE_FLAG = 0x800000;
Pixels latex_to_glyph_ids(const std::string& latex, const ScalingParams& scaling_params);
vector<uint32_t> latex_glyph_shapes(const std::string& latex);
void write_text(uint32_t* gpu_pix, const ivec2& canvas_wh, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle_rad);
void write_text_linear(uint32_t* gpu_pix, const ivec2& canvas_wh, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const vec4& inv_matrix);
