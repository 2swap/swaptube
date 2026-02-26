#pragma once

#include <string>

void print_global_state();
double get_global_state(const std::string& key);
void set_global_state(const std::string& key, double value);
bool global_state_exists(const std::string& key);
