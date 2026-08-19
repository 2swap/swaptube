#pragma once

#include <optional>
#include <string>

void initialize_timeline_plan(const std::string& path, bool record);
bool is_recording_microblock_plan();
int begin_macroblock_plan_entry(const std::string& blurb, std::optional<int> declared_count = std::nullopt);
void record_planned_microblock();
void finalize_timeline_plan();
