#pragma once

#include <string>
#include "../Core/Pixels.h"
#include "../Host_Device_Shared/vec.h"
#include <cstdint>

void pix_to_png(const Pixels& pix, const std::string& full_filename);
void png_to_pix(Pixels& pix, const std::string& filename_with_or_without_suffix);
void png_to_raw_data(uint32_t*& unallocated_data, int& width, int& height, const string& filename_with_or_without_suffix);
void pdf_page_to_pix(Pixels& pix, const string& pdf_filename_without_suffix, const int page_number);
