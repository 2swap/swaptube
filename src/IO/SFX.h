#pragma once

#include <string>

void sfx_boink(double time, double freq, double halflife_seconds, double volume, const std::string& voice = "boink");
void sfx_clap(double time, double halflife_seconds, double volume, const std::string& voice = "clap");
