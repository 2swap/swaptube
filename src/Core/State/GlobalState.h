#pragma once

#include <string>
#include <unordered_map>

using namespace std;

static unordered_map<string, double> global_state{
    {"frame_number", 0},
    {"t", 0},
    {"macroblock_number", 0},
    {"microblock_number", 0},
    {"macroblock_fraction", 0},
    {"microblock_fraction", 0},
    {"voice", 0}, // How loud the voice is right now (max L-channel sample as float for this frame)
};
void print_global_state();
double get_global_state(const std::string& key);
void set_global_state(const std::string& key, double value);
bool global_state_exists(const std::string& key);
