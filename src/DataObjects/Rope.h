#pragma once

#include <array>
#include <string>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../DataObjects/DataObject.h"

class Rope : public DataObject {
    public:

        vec2* d_nodes; //this is the first node's position actually
        vec2* d_pins;

        vector<vec2> h_pins;

        void tick(const StateReturn& state);
        void add_pin(const vec2& pos);
        void remove_pin(int pin_index);

        Rope(const std::string& file_name);


};