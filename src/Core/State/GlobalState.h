#pragma once

#include <string>
#include <unordered_map>

using namespace std;

void print_global_state();
double get_global_state(const std::string& key);
void set_global_state(const std::string& key, double value);
bool global_state_exists(const std::string& key);
const unordered_map<string, double>& get_all_global_state();
