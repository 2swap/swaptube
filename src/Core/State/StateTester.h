#pragma once
#include <string>
#include "../../Scenes/Scene.h"

void parse_command(const std::string& command, std::string& state_value, std::string& operation);
void open_ui(Scene& scene);
