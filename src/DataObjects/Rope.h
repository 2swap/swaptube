#pragma once

#include <array>
#include <string>
#include <vector>
#include "../Host_Device_Shared/vec.h"

class Rope {
    public:

        vec2* d_nodes; //this is the first node's position actually
        vec2* d_pins;

        std::vector<vec2> h_pins;

        void tick();
        void add_pin(const vec2& pos);
        void remove_pin(int pin_index);

        Rope(const std::string& file_name);
};
