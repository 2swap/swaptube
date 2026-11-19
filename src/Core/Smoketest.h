#pragma once

#include <string>

// Accessors for the process-wide flags which indicate whether we are running a smoketest,
// and whether the current macroblock is to be rendered.

bool is_smoketest();
void set_smoketest(bool smoketest);
bool macroblock_in_range();
bool rendering_on();

void parse_macroblock_range(const std::string& arg);
void enter_macroblock(int macroblock_index);
