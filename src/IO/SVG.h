#pragma once

#include <string>
#include "../DataObjects/DevicePointer.h"
#include "../Host_Device_Shared/vec.h"
#include "ScalingParams.h"
#include <memory>
#include <cstdint>

shared_ptr<DevicePointer> svg_to_gpu_pix(const std::string& filename_with_or_without_suffix, ScalingParams& scaling_params);
