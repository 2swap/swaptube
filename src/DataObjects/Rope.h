#pragma once

#include <array>
#include <string>
#include "../Host_Device_Shared/vec.h"
#include "../DataObjects/DataObject.h"

class Rope : public DataObject {
    public:

        vec2* d_nodes; //this is the first node's position actually
        vec2* d_pins;

        void tick(const StateReturn& state);

        Rope(std::string file_name);


};